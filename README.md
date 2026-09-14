# Face Recognition with ArcFace ONNX and 5-Point Alignment

This project is a local, camera-based face-recognition demo built around:

- OpenCV for camera capture and image processing
- Haar cascade face detection
- MediaPipe FaceMesh landmarks for 5-point alignment
- A 112x112 alignment step for ArcFace-style matching
- An ONNX face embedder and a small local face database

This is a research/demo pipeline, not a production biometric system. It is intended to run on a developer machine with a webcam and a compatible ArcFace ONNX model.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 to 3.13
- A working webcam
- A compatible ArcFace ONNX model at:

```text
models/embedder_arcface.onnx
```

The model is not included in this repository. Download or place a compatible model at that exact path before running the embedding, enrollment, evaluation, or recognition scripts.

## Setup

From the repository root, create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install opencv-python numpy onnxruntime mediapipe==0.10.21
```

The `.venv` directory is local to your machine and is already ignored by `.gitignore`.

> Important: this repository does not install dependencies for you automatically. The scripts will fail at import time until the packages above are installed.

## Usage

Run the commands from the project root with the virtual environment active.

### Check the camera

```powershell
python -m src.camera
```

### Test face detection

```powershell
python -m src.detect
```

### Preview five-point alignment

```powershell
python -m src.align
```

Press `q` to quit, or `s` to save the current aligned face under `data/debug_aligned/`.

### Enroll a person

```powershell
python -m src.enroll
```

Enter a name when prompted and follow the on-screen capture instructions. Enrollment creates:

```text
data/db/face_db.npz
data/db/face_db.json
data/enroll/<person-name>/
```

### Recognize enrolled people

```powershell
python -m src.recognize
```

The recognition window supports:

- `q`: quit
- `r`: reload the face database
- `+` / `-`: adjust the match threshold
- `d`: toggle debug overlays

### Evaluate the enrolled database

```powershell
python -m src.evaluate
```

This evaluates distances between enrolled crops and prints a threshold sweep. It expects aligned crops in `data/enroll` and uses the model from `models/embedder_arcface.onnx`.

## Project layout

```text
face-recognition-5pt/
├── data/
│   ├── db/              Generated face database files
│   ├── debug_aligned/   Saved alignment previews
│   └── enroll/          Captured enrollment faces
├── models/              Local ONNX model files
├── src/
│   ├── align.py         Five-point alignment demo
│   ├── camera.py        Camera test script
│   ├── detect.py        Haar detection demo
│   ├── embed.py         ArcFace embedding demo
│   ├── enroll.py        Build the local face database
│   ├── evaluate.py      Evaluate enrolled samples
│   ├── haar_5pt.py      Shared Haar + FaceMesh detection/alignment logic
│   ├── landmark.py      Minimal 5-point landmark preview
│   └── recognize.py     Real-time recognition
├── init_project.py      Optional folder bootstrap helper
├── README.md
├── .gitignore
└── .venv/               Local virtual environment (not committed)
```

> Note: `init_project.py` is a simple scaffold helper. It creates the folder structure and does not install dependencies or download the model. It also includes a placeholder `book/` directory and a `landmarks.py` name in the generated scaffold, but the actual source file used by the project is `src/landmark.py`.

## Known runtime assumptions

- The scripts assume a webcam is available and use `cv2.VideoCapture(0)` by default.
- If your machine uses another camera index, change the `VideoCapture(...)` argument in the relevant script.
- The code expects to be run from the project root so Python can import the `src` package correctly.
- The project is optimized for a developer workflow and is not a packaged install.

## Troubleshooting

### Camera does not open

Close other applications using the webcam. The scripts use camera index `0` in the main recognition and enrollment flows. If your camera uses another index, update the `cv2.VideoCapture(...)` value in the relevant script.

### MediaPipe cannot be imported

Confirm that the virtual environment is active, then reinstall the pinned version:

```powershell
python -m pip install --force-reinstall mediapipe==0.10.21
```

### The ONNX model cannot be found

Run the commands from the repository root and confirm this file exists:

```text
models/embedder_arcface.onnx
```

### Import errors for `cv2` or `onnxruntime`

This usually means the virtual environment has not been activated or the packages were not installed. Reinstall the setup from the section above.

## What this repository does well

- Demo face detection + alignment with a webcam
- Real-time multi-face recognition loop using Haar + FaceMesh + ArcFace-style embeddings
- Local DB creation and reload for enrollment / recognition
- Threshold evaluation for practical matching decisions

## Limitations

- Requires a compatible ONNX ArcFace model to function
- Depends on a good webcam and stable lighting
- Uses basic local matching logic, not a full production identity system
- Works as a local research/demo project rather than packaged software
