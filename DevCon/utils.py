"""Paths, precision, and device defaults for the DL Streamer car demo (Linux server paths)."""

from __future__ import annotations

import os
from pathlib import Path

# --- Layout (India-DevCon-2026 + dlstreamer-demo) ---
WORKDIR = Path("/home/ubuntu/openvino/work/India-DevCon-2026/2")
VIDEO_DIR = WORKDIR / "Videos"
DEFAULT_VIDEO = "1900-151662242_medium.mp4"
MODEL_ROOT = Path("/home/ubuntu/dlstreamer-demo/car-model")

DEFAULT_PRECISION = "FP16"
DETECTION_DEVICE = "GPU"
CLASSIFICATION_DEVICE = "NPU"
RECLASSIFY_INTERVAL = 2


def model_paths(precision: str) -> tuple[Path, Path]:
    prec = precision.upper()
    base = MODEL_ROOT / prec / "deployment"
    det = base / "Detection" / "model" / "model.xml"
    cls = base / "Classification" / "model" / "model.xml"
    return det, cls


def apply_environment(
    *,
    video_name: str | None = None,
    precision: str | None = None,
) -> None:
    """Populate os.environ for variables used by %%bash cells in the notebook."""
    video = video_name or DEFAULT_VIDEO
    prec = (precision or DEFAULT_PRECISION).upper()
    det, cls = model_paths(prec)
    video_src = VIDEO_DIR / video

    os.environ["WORKDIR"] = str(WORKDIR)
    os.environ["VIDEO_DIR"] = str(VIDEO_DIR)
    os.environ["VIDEO_SRC"] = str(video_src)
    os.environ["MODEL_DIR"] = str(MODEL_ROOT)
    os.environ["DETECTION_MODEL"] = str(det)
    os.environ["CLASSIFICATION_MODEL"] = str(cls)
    os.environ["DETECTION_DEVICE"] = DETECTION_DEVICE
    os.environ["CLASSIFICATION_DEVICE"] = CLASSIFICATION_DEVICE
    os.environ["PRECISION"] = prec

    for label in ("FP16", "FP32", "INT8"):
        d, c = model_paths(label)
        os.environ[f"DETECTION_MODEL_{label}"] = str(d)
        os.environ[f"CLASSIFICATION_MODEL_{label}"] = str(c)


def validate_default_assets() -> None:
    """Fail fast if the default video and FP16 models are not on disk."""
    apply_environment()
    for key in ("VIDEO_SRC", "DETECTION_MODEL", "CLASSIFICATION_MODEL"):
        p = Path(os.environ[key])
        if not p.is_file():
            raise FileNotFoundError(f"Missing {key}: {p}")
