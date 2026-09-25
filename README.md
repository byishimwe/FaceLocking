# FaceLocking: Face Recognition with ArcFace ONNX and 5-Point Alignment (Part 1 & Part 2)

This project implements a complete face recognition and tracking pipeline built around:

- **Part 1**: Face detection, 5-point landmark alignment, ArcFace ONNX embedding, enrollment, and live recognition
- **Part 2**: Identity-locked tracking, smile detection, blink counting, closed-eye state, and normalized position errors for motor control

## Overview

| Part | Description |
|------|-------------|
| **Part 1** | Face recognition pipeline: detect → align → embed → enroll → recognize |
| **Part 2** | Identity-locked tracking with facial signals: lock → track → smile/blink/position |

The final output is a stable software signal (`error_x`, `error_y`, `smile`, `blink`, `eyes_closed`) ready for Part 3 motor control.

## Requirements

- Windows, macOS, or Linux
- Python 3.10+
- A working webcam
- ArcFace ONNX model at `models/embedder_arcface.onnx` (~30MB)

## Setup

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install opencv-python numpy onnxruntime scipy tqdm mediapipe==0.10.35
```

> **Note**: Use `mediapipe==0.10.35` (not 1.x) for `mp.solutions.face_mesh` compatibility.

## Download ArcFace Model

```powershell
cd models
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
unzip -o buffalo_l.zip
cp w600k_r50.onnx embedder_arcface.onnx
```

## Part 1: Face Recognition Pipeline

### Quick Test Camera
```powershell
python -m src.camera
```

### Test Face Detection
```powershell
python -m src.detect
```

### Test 5-Point Landmarks
```powershell
python -m src.landmarks
```

### Test Face Alignment
```powershell
python -m src.align
# Press 's' to save aligned face, 'q' to quit
```

### Test Face Embeddings
```powershell
python -m src.embed
```

### Enroll a Person
```powershell
python -m src.enroll
# Enter name (e.g., "Prince")
# Press SPACE to capture ~15 samples
# Press 's' to save to database
```

### Live Recognition
```powershell
python -m src.recognize
# Controls: q=quit, r=reload DB, +/-=threshold, d=debug overlay
```

### Evaluate Threshold
```powershell
python -m src.evaluate
# Requires at least 2 enrolled people with 5+ samples each
```

## Part 2: Identity-Locked Tracking with Facial Signals

### Run Tracking
```powershell
python -m src.face_tracking --target "Prince"
```

### What Part 2 Does

1. **Identity Lock**: Only the specified `--target` can acquire the lock
2. **Three States**: `SEARCHING` → `LOCKED` ↔ `LOST` → `SEARCHING` (after timeout)
3. **Geometric Tracking**: IoU + center displacement association with EMA smoothing (α=0.30)
4. **Periodic Verification**: ArcFace re-check every 10 frames while locked
5. **Facial Signals** (only on locked face):
   - **Smile**: Mouth width / face width with hysteresis (on=0.38, off=0.35)
   - **Blink**: EAR-based event detection (2-7 frames closure)
   - **Eyes Closed**: Sustained closure state (≥8 frames)
   - **Position**: Normalized error_x, error_y with 0.07 dead zone

### Display Output
```
LOCKED: Prince
H=LEFT V=UP error=(-0.24, -0.11)
SMILE | EYES OPEN blinks=3
EAR=0.287 smile=0.412
```

### Controls
- `q` - Quit

## Project Layout

```
FaceLocking/
├── data/
│   ├── db/                  face_db.npz, face_db.json
│   ├── debug_aligned/       Saved alignment previews
│   └── enroll/              Captured enrollment faces per person
├── models/
│   └── embedder_arcface.onnx  ArcFace ONNX model (~30MB)
├── src/
│   ├── align.py             Five-point alignment demo
│   ├── camera.py            Camera test
│   ├── detect.py            Haar detection demo
│   ├── embed.py             ArcFace embedding demo
│   ├── enroll.py            Build local face database
│   ├── evaluate.py          Threshold evaluation
│   ├── face_signals.py      Part 2: Smile, blink, eye-closed logic
│   ├── face_tracking.py     Part 2: Identity lock, tracking, position
│   ├── haar_5pt.py          Shared Haar + FaceMesh detection/alignment
│   ├── landmarks.py         5-point landmark preview
│   └── recognize.py         Real-time multi-face recognition
├── init_project.py          Folder bootstrap helper
├── README.md
├── .gitignore
└── .venv/                   Virtual environment (not committed)
```

## Key Files

| File | Responsibility |
|------|----------------|
| `src/haar_5pt.py` | Haar detection + MediaPipe FaceMesh 5pt landmarks |
| `src/align.py` | Similarity transform alignment to 112×112 |
| `src/embed.py` | ArcFace ONNX embedding extraction |
| `src/enroll.py` | Capture samples, compute mean embedding, save DB |
| `src/recognize.py` | Multi-face live recognition |
| `src/evaluate.py` | Genuine/impostor distance analysis |
| `src/face_signals.py` | EAR, blink, closed-eye, smile detection |
| `src/face_tracking.py` | Identity lock, tracking, position signals |

## Troubleshooting

### Camera does not open
Close other apps using webcam. Change `cv2.VideoCapture(0)` index if needed.

### MediaPipe import error
```powershell
python -m pip install --force-reinstall mediapipe==0.10.35
```

### Haar cascade not found
```powershell
curl -L -o "$env:USERPROFILE\AppData\Roaming\Python\Python314\site-packages\cv2\data\haarcascade_frontalface_default.xml" https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

### ONNX model not found
Ensure `models/embedder_arcface.onnx` exists and is ~30MB.

## What This Project Does Well

- Modular, explainable face recognition pipeline
- CPU-only, real-time on standard hardware
- Identity lock prevents distractor transfer
- Geometric tracking with periodic identity verification
- Facial signals (smile, blink, position) on locked target only
- Normalized errors ready for motor control (Part 3)

## Limitations

- Requires compatible ArcFace ONNX model
- Depends on good webcam and lighting
- MediaPipe 0.10.x required (not 1.x)
- Not a production biometric system
- Local matching only, no network/distributed features

## References

- ArcFace: Deng et al., CVPR 2019
- MediaPipe Face Mesh: Lugaresi et al., 2019
- Eye Aspect Ratio: Soukupová & Čech, 2016
- ONNX Runtime: Microsoft