from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2


ARC_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def estimate_affine_5pt(
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    k = kps_5x2.astype(np.float32)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    dst = ARC_TEMPLATE_112.copy()
    if (out_w, out_h) != (112, 112):
        sx = out_w / 112.0
        sy = out_h / 112.0
        dst = dst * np.array([sx, sy], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(k, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(
            np.array([k[0], k[1], k[2]], dtype=np.float32),
            np.array([dst[0], dst[1], dst[2]], dtype=np.float32)
        )
    return M.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112)
) -> Tuple[np.ndarray, np.ndarray]:
    M = estimate_affine_5pt(kps_5x2, out_size=out_size)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    aligned = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return aligned, M


def clip_box_xyxy(box: np.ndarray, W: int, H: int) -> np.ndarray:
    bb = box.astype(np.float32).copy()
    bb[0] = np.clip(bb[0], 0, W - 1)
    bb[1] = np.clip(bb[1], 0, H - 1)
    bb[2] = np.clip(bb[2], 0, W - 1)
    bb[3] = np.clip(bb[3], 0, H - 1)
    return bb


def clip_xyxy(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    rx1 = int(max(0, min(W - 1, round(x1))))
    ry1 = int(max(0, min(H - 1, round(y1))))
    rx2 = int(max(0, min(W - 1, round(x2))))
    ry2 = int(max(0, min(H - 1, round(y2))))
    if rx2 < rx1:
        rx1, rx2 = rx2, rx1
    if ry2 < ry1:
        ry1, ry2 = ry2, ry1
    return rx1, ry1, rx2, ry2


def bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15
) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def ema(
    prev: Optional[np.ndarray], cur: np.ndarray, alpha: float
) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)


def kps_span_ok(kps: np.ndarray, min_eye_dist: float = 12.0) -> bool:
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


def order_landmarks_5pt(kps: np.ndarray) -> np.ndarray:
    k = kps.copy()
    if k[0, 0] > k[1, 0]:
        k[[0, 1]] = k[[1, 0]]
    if k[3, 0] > k[4, 0]:
        k[[3, 4]] = k[[4, 3]]
    return k


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32).reshape(-1)
    n = float(np.linalg.norm(v) + eps)
    return (v / n).astype(np.float32)


def preprocess_arcface(
    aligned_bgr: np.ndarray,
    input_size: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    in_w, in_h = input_size
    if aligned_bgr.shape[1] != in_w or aligned_bgr.shape[0] != in_h:
        aligned_bgr = cv2.resize(aligned_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0
    x = np.transpose(rgb, (2, 0, 1))[None, ...]
    return x.astype(np.float32)