"""Paths, pipelines, and runners for the DL Streamer car demo (Linux server paths)."""

from __future__ import annotations

import os
import shlex
import subprocess
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

DLSTREAMER_GITHUB = "https://github.com/dlstreamer/dlstreamer"

# Quieter GStreamer logs in Jupyter (overlay still draws on the video window)
_GST_VISUAL_ENV = {
    "GST_DEBUG": "0",
    "GST_DEBUG_NO_COLOR": "1",
}


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
    detection_device: str | None = None,
    classification_device: str | None = None,
) -> None:
    """Populate os.environ for pipeline builders and runners."""
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
    os.environ["DETECTION_DEVICE"] = detection_device or DETECTION_DEVICE
    os.environ["CLASSIFICATION_DEVICE"] = classification_device or CLASSIFICATION_DEVICE
    os.environ["PRECISION"] = prec
    os.environ["RECLASSIFY_INTERVAL"] = str(RECLASSIFY_INTERVAL)

    for label in ("FP16", "FP32", "INT8"):
        d, c = model_paths(label)
        os.environ[f"DETECTION_MODEL_{label}"] = str(d)
        os.environ[f"CLASSIFICATION_MODEL_{label}"] = str(c)


def validate_selected_assets() -> None:
    for key in ("VIDEO_SRC", "DETECTION_MODEL", "CLASSIFICATION_MODEL"):
        p = Path(os.environ[key])
        if not p.is_file():
            raise FileNotFoundError(f"Missing {key}: {p}")


def get_available_devices() -> list[str]:
    """OpenVINO device short names (CPU, GPU, NPU) when available."""
    try:
        import openvino as ov

        raw = ov.Core().available_devices
        seen: set[str] = set()
        out: list[str] = []
        for d in raw:
            short = d.split(".")[0]
            if short not in seen:
                seen.add(short)
                out.append(short)
        return out if out else ["CPU", "GPU", "NPU"]
    except Exception:
        return ["CPU", "GPU", "NPU"]


def get_device_dropdown_options() -> list[str]:
    """CPU, GPU, NPU first, then any extra device types OpenVINO reports (e.g. GNA)."""
    preferred = ["CPU", "GPU", "NPU"]
    extra = sorted(set(get_available_devices()) - set(preferred))
    return preferred + extra


def _q(path: str) -> str:
    return shlex.quote(path)


def pipeline_raw_video() -> str:
    v = _q(os.environ["VIDEO_SRC"])
    return f"filesrc location={v} ! decodebin3 ! videoconvert ! autovideosink sync=true"


def pipeline_fps() -> str:
    v = _q(os.environ["VIDEO_SRC"])
    return f"filesrc location={v} ! decodebin3 ! videoconvert ! gvafpscounter ! autovideosink sync=true"


def pipeline_detection() -> str:
    v = _q(os.environ["VIDEO_SRC"])
    m = _q(os.environ["DETECTION_MODEL"])
    d = os.environ["DETECTION_DEVICE"]
    return (
        f"filesrc location={v} ! decodebin3 ! "
        f"gvadetect model={m} device={d} pre-process-backend=opencv ! "
        f"gvawatermark ! gvafpscounter ! videoconvert ! autovideosink sync=true"
    )


def pipeline_detect_classify() -> str:
    v = _q(os.environ["VIDEO_SRC"])
    det = _q(os.environ["DETECTION_MODEL"])
    cls = _q(os.environ["CLASSIFICATION_MODEL"])
    dd = os.environ["DETECTION_DEVICE"]
    cd = os.environ["CLASSIFICATION_DEVICE"]
    ri = os.environ["RECLASSIFY_INTERVAL"]
    return (
        f"filesrc location={v} ! decodebin3 ! "
        f"gvadetect model={det} device={dd} pre-process-backend=opencv ! "
        f"gvatrack ! "
        f"gvaclassify model={cls} device={cd} pre-process-backend=opencv reclassify-interval={ri} ! "
        f"queue ! gvawatermark ! gvafpscounter ! videoconvert ! autovideosink sync=true"
    )


def _branch() -> str:
    v = _q(os.environ["VIDEO_SRC"])
    det = _q(os.environ["DETECTION_MODEL"])
    cls = _q(os.environ["CLASSIFICATION_MODEL"])
    dd = os.environ["DETECTION_DEVICE"]
    cd = os.environ["CLASSIFICATION_DEVICE"]
    ri = os.environ["RECLASSIFY_INTERVAL"]
    return (
        f"filesrc location={v} ! decodebin3 ! "
        f"gvadetect model={det} device={dd} pre-process-backend=opencv ! "
        f"gvatrack ! "
        f"gvaclassify model={cls} device={cd} pre-process-backend=opencv reclassify-interval={ri} ! "
        f"queue ! gvafpscounter ! fakesink sync=false"
    )


def pipeline_benchmark_2() -> str:
    b = _branch()
    return f"{b} {b}"


def pipeline_benchmark_4() -> str:
    b = _branch()
    return f"{b} {b} {b} {b}"


def run_visual(pipeline: str) -> None:
    """Run a pipeline with display; do not fail on window close (exit often nonzero)."""
    env = {**os.environ, **_GST_VISUAL_ENV}
    subprocess.run(["gst-launch-1.0", pipeline], env=env, check=False)
    print("Pipeline ended.")


def run_benchmark(pipeline: str) -> None:
    """Run headless benchmark; fail on error."""
    env = {**os.environ, "GST_DEBUG": "0"}
    subprocess.run(["gst-launch-1.0", pipeline], env=env, check=True)


def show_config_widgets() -> None:
    """Jupyter: precision + device dropdowns and Apply; sets os.environ for pipelines."""
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError:
        apply_environment()
        validate_selected_assets()
        print("ipywidgets missing; defaults from utils.")
        return

    dev_opts = get_device_dropdown_options()
    det_default = DETECTION_DEVICE if DETECTION_DEVICE in dev_opts else dev_opts[0]
    cls_default = CLASSIFICATION_DEVICE if CLASSIFICATION_DEVICE in dev_opts else dev_opts[0]

    prec_w = w.Dropdown(
        options=["FP16", "FP32", "INT8"],
        value=DEFAULT_PRECISION,
        description="Precision:",
    )
    det_w = w.Dropdown(options=dev_opts, value=det_default, description="Detect dev:")
    cls_w = w.Dropdown(options=dev_opts, value=cls_default, description="Classify dev:")

    def _apply(_=None):
        apply_environment(
            precision=prec_w.value,
            detection_device=det_w.value,
            classification_device=cls_w.value,
        )
        validate_selected_assets()
        print(
            os.environ["PRECISION"],
            os.environ["DETECTION_DEVICE"],
            os.environ["CLASSIFICATION_DEVICE"],
        )
        print(os.environ["VIDEO_SRC"])

    btn = w.Button(description="Apply configuration")
    btn.on_click(_apply)
    display(prec_w, det_w, cls_w, btn)
    _apply()


def check_gvadetect() -> bool:
    """Return True if the DL Streamer detect plugin is registered."""
    r = subprocess.run(
        ["gst-inspect-1.0", "gvadetect"],
        capture_output=True,
    )
    ok = r.returncode == 0
    print("DL Streamer plugins available (gvadetect found)." if ok else "WARNING: gvadetect not found.")
    return ok


def print_bash_exports() -> None:
    """Emit export lines for shell sessions."""
    keys = [
        "WORKDIR",
        "VIDEO_DIR",
        "VIDEO_SRC",
        "MODEL_DIR",
        "DETECTION_MODEL",
        "CLASSIFICATION_MODEL",
        "DETECTION_DEVICE",
        "CLASSIFICATION_DEVICE",
        "PRECISION",
        "RECLASSIFY_INTERVAL",
    ]
    for k in keys:
        if k in os.environ:
            print(f"export {k}={shlex.quote(os.environ[k])}")
    for label in ("FP16", "FP32", "INT8"):
        for prefix in ("DETECTION_MODEL_", "CLASSIFICATION_MODEL_"):
            k = f"{prefix}{label}"
            if k in os.environ:
                print(f"export {k}={shlex.quote(os.environ[k])}")


def validate_default_assets() -> None:
    apply_environment()
    validate_selected_assets()
