from __future__ import annotations
from pathlib import Path


class Config:
    MODEL_PATH = "models/embedder_arcface.onnx"
    INPUT_SIZE = (112, 112)
    DB_NPZ = Path("data/db/face_db.npz")
    DB_JSON = Path("data/db/face_db.json")
    ENROLL_DIR = Path("data/enroll")
    DEBUG_ALIGNED_DIR = Path("data/debug_aligned")
    HAAR_CASCADE = "haarcascade_frontalface_default.xml"
    DEFAULT_DIST_THRESH = 0.34
    MIN_FACE_SIZE = (70, 70)
    SMOOTH_ALPHA = 0.80
    EYE_DIST_THRESHOLD = 12.0


def get_model_path() -> str:
    return Config.MODEL_PATH


def get_db_path() -> Path:
    return Config.DB_NPZ


def get_enroll_dir() -> Path:
    return Config.ENROLL_DIR