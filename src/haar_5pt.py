from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    MP_IMPORT_ERROR = e

from .utils import (
    align_face_5pt,
    bbox_from_5pt,
    clip_box_xyxy,
    ema,
    kps_span_ok,
    order_landmarks_5pt,
)


@dataclass
class FaceKpsBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5, 2) float32


from .config import Config

class Haar5ptDetector:

    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = Config.MIN_FACE_SIZE,
        smooth_alpha: float = Config.SMOOTH_ALPHA,
        debug: bool = True,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)

        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + Config.HAAR_CASCADE
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        if mp is None:
            raise RuntimeError(
                f"MediaPipe import failed ({MP_IMPORT_ERROR})\nInstall: pip install mediapipe"
            )

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

    def haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)

    def facemesh_5pt(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        idxs = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]
        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])
        kps = np.array(pts, dtype=np.float32)
        return order_landmarks_5pt(kps)

    def detect(
        self, frame_bgr: np.ndarray, max_faces: int = 1
    ) -> List[FaceKpsBox]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar_faces(gray)
        if faces.shape[0] == 0:
            return []

        areas = faces[:, 2] * faces[:, 3]
        i = int(np.argmax(areas))
        x, y, w, h = faces[i].tolist()

        kps = self.facemesh_5pt(frame_bgr)
        if kps is None:
            if self.debug:
                print(
                    "[haar_5pt] Haar face found but FaceMesh returned none -> reject"
                )
            return []

        margin = 0.35
        x1m = x - margin * w
        y1m = y - margin * h
        x2m = x + (1.0 + margin) * w
        y2m = y + (1.0 + margin) * h
        inside = (
            (kps[:, 0] >= x1m)
            & (kps[:, 0] <= x2m)
            & (kps[:, 1] >= y1m)
            & (kps[:, 1] <= y2m)
        )
        if inside.mean() < 0.60:
            if self.debug:
                print(
                    "[haar_5pt] FaceMesh points not consistent with Haar box -> reject"
                )
            return []

        if not kps_span_ok(kps, min_eye_dist=max(Config.EYE_DIST_THRESHOLD, 0.18 * w)):
            if self.debug:
                print("[haar_5pt] 5pt geometry sanity failed -> reject")
            return []

        box = bbox_from_5pt(
            kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15
        )
        box = clip_box_xyxy(box, W, H)

        box_s = ema(self._prev_box, box, self.smooth_alpha)
        kps_s = ema(self._prev_kps, kps, self.smooth_alpha)
        self._prev_box = box_s.copy()
        self._prev_kps = kps_s.copy()

        x1, y1, x2, y2 = box_s
        score = 1.0

        return [
            FaceKpsBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                score=float(score),
                kps=kps_s.astype(np.float32),
            )
        ][:max_faces]


def main():
    cap = cv2.VideoCapture(0)
    det = Haar5ptDetector(
        min_size=Config.MIN_FACE_SIZE,
        smooth_alpha=Config.SMOOTH_ALPHA,
        debug=True,
    )
    print("Haar + 5pt (FaceMesh) test. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det.detect(frame, max_faces=1)
        vis = frame.copy()
        if faces:
            f = faces[0]
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for x, y in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(
                vis,
                "OK",
                (f.x1, max(0, f.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                vis,
                "no face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        cv2.imshow("haar_5pt", vis)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()