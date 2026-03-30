"""
Geti Lab 4 — unified ``bm_utils`` (single file for easy copy to remote).

Includes: preprocess/postprocess/two-stage timing (formerly ``benchmark_utils``),
notebook helpers (formerly ``ov_lab4_utils``), and ``benchmark_app`` CLI helpers
(formerly ``lab4_benchmark_app``). Gradio UIs live in ``app_gradio.py``.

Teaching-oriented names: ``detection_input_hw``, ``draw_detection_boxes``,
``draw_labeled_boxes``, ``plot_detection_classification_benchmark``.
Older aliases remain (e.g. ``get_detection_input_hw``).

For ``Inference with OpenVINO_Original.ipynb`` (face / MobileNet / U-Net sample paths),
use ``bm_utils_original_tutorial_models.py`` instead of this file (or copy it over as
``bm_utils.py`` when running that notebook only).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_labels(model_dir: Union[str, Path]) -> List[str]:
    """Labels from ``model_dir/config.json`` ``model_parameters.labels`` (space-separated)."""
    model_dir = Path(model_dir)
    p = model_dir / "config.json"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    raw = cfg.get("model_parameters", {}).get("labels", "") or ""
    return [s.strip() for s in raw.split()]


def list_available_precisions(models_dir: Path) -> List[str]:
    """
    Precision names under ``models_dir`` that have both Detection and Classification IRs
    (``FP32``, ``FP16``, ``INT8`` checked in order).
    """
    out: List[str] = []
    for name in ("FP32", "FP16", "INT8"):
        det = models_dir / name / "Detection" / "model.xml"
        cls = models_dir / name / "Classification" / "model.xml"
        if det.is_file() and cls.is_file():
            out.append(name)
    return out


# --- Detection: from model rt_info mean=0, scale=255, resize_type=standard ---
def preprocess_detection(image, input_h=800, input_w=992):
    """Detection preprocess: RGB, resize to input size, /255 (matches Geti/OTX model_api)."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h))
    normalized = resized.astype(np.float32) / 255.0
    nchw = np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
    return nchw


def postprocess_detection(output, orig_h, orig_w, input_w, input_h, thresh=0.225):
    """Detection postprocess: boxes [N,5] -> (boxes, scores)."""
    out_by_name = {}
    for k, v in output.items():
        name = k.get_any_name() if hasattr(k, "get_any_name") else str(k)
        out_by_name[name.lower()] = np.array(v)
    boxes_arr = out_by_name.get("boxes")
    if boxes_arr is None:
        boxes_arr = np.squeeze(list(output.values())[0])
    boxes_arr = np.squeeze(boxes_arr)
    if boxes_arr.ndim == 3:
        boxes_arr = boxes_arr.reshape(-1, 5)
    elif boxes_arr.ndim != 2 or boxes_arr.shape[1] != 5:
        return np.zeros((0, 4)), np.zeros(0)
    result_boxes, result_scores = [], []
    for i in range(len(boxes_arr)):
        x1, y1, x2, y2, conf = boxes_arr[i]
        if conf < thresh:
            continue
        norm_x1, norm_y1 = x1 / input_w, y1 / input_h
        norm_x2, norm_y2 = x2 / input_w, y2 / input_h
        bx1 = int(np.clip(norm_x1 * orig_w, 0, orig_w))
        by1 = int(np.clip(norm_y1 * orig_h, 0, orig_h))
        bx2 = int(np.clip(norm_x2 * orig_w, 0, orig_w))
        by2 = int(np.clip(norm_y2 * orig_h, 0, orig_h))
        if (bx2 - bx1) * (by2 - by1) > 1.0 and bx2 > bx1 and by2 > by1:
            result_boxes.append([bx1, by1, bx2, by2])
            result_scores.append(float(conf))
    return (np.array(result_boxes) if result_boxes else np.zeros((0, 4)),
            np.array(result_scores) if result_scores else np.zeros(0))


def detection_input_hw(compiled_detection) -> Tuple[int, int]:
    """Height and width expected by the detection model input (defaults 800×992 if dynamic)."""
    inp = compiled_detection.input(0)
    sh = inp.get_partial_shape()
    input_h = int(sh[2].get_length()) if sh[2].is_static else 800
    input_w = int(sh[3].get_length()) if sh[3].is_static else 992
    return input_h, input_w


def run_detection_ms(compiled_detection, input_tensor) -> Tuple[float, Any]:
    """Run the detection model once. Returns (time_ms, model_output)."""
    t0 = time.perf_counter()
    det_out = compiled_detection([input_tensor])
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, det_out


def draw_detection_boxes(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    label: str = "CAR",
    box_color: Tuple[int, int, int] = (0, 220, 255),
) -> np.ndarray:
    """BGR image copy with rectangles and a small label chip per box (detection preview)."""
    det_label_draw = str(label).upper() if label else "CAR"
    vis = image_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(det_label_draw, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(vis, det_label_draw, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    return vis


def draw_labeled_boxes(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    labels: Sequence[str],
    box_color: Tuple[int, int, int] = (0, 220, 255),
) -> np.ndarray:
    """BGR copy with per-box text (e.g. classification: CAR — color)."""
    vis = image_bgr.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        text = labels[i] if i < len(labels) else ""
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    return vis


# --- Classification: from model rt_info mean=[123.675,116.28,103.53], scale=[58.395,57.12,57.375], resize_type=standard ---
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SCALE = np.array([58.395, 57.12, 57.375], dtype=np.float32)
MEAN_RGB = MEAN
SCALE_RGB = SCALE


def preprocess_classification(crop, input_h=224, input_w=224):
    """Classification preprocess: RGB, resize to 224x224, (pixel - mean) / scale (matches Geti/OTX)."""
    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h))
    normalized = (resized.astype(np.float32) - MEAN) / SCALE
    nchw = np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
    return nchw


def classify_crops_with_labels(
    compiled_classification,
    image_bgr,
    detection_boxes,
    classification_labels,
    detection_label="CAR",
):
    """
    Classify each detection crop only (detection must already be done).
    Returns (display_labels, total_cls_ms). Empty crops get \"Unknown\".
    Display format: \"{DET_LABEL} - {color}\".
    """
    cin = compiled_classification.input(0)
    csh = cin.get_partial_shape()
    cls_h = int(csh[2].get_length()) if csh[2].is_static else 224
    cls_w = int(csh[3].get_length()) if csh[3].is_static else 224
    prefix = (detection_label or "CAR").upper()
    color_names = []
    total_cls_ms = 0.0
    for box in detection_boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            color_names.append("Unknown")
            continue
        cls_tensor = preprocess_classification(crop, cls_h, cls_w)
        t0 = time.perf_counter()
        cls_out = compiled_classification([cls_tensor])
        t1 = time.perf_counter()
        total_cls_ms += (t1 - t0) * 1000.0
        logits = np.squeeze(list(cls_out.values())[0])
        idx = int(np.argmax(logits))
        label = classification_labels[idx] if 0 <= idx < len(classification_labels) else "Unknown"
        color_names.append(f"{prefix} - {label}")
    return color_names, total_cls_ms


def timed_two_stage_pipeline_ms(det_compiled, cls_compiled, image_bgr, thresh=0.225):
    """
    One timed run: detection, then classification for each box.

    Returns (det_ms, cls_ms, n_boxes). cls_ms adds up every crop.
    """
    input_h, input_w = detection_input_hw(det_compiled)
    orig_h, orig_w = image_bgr.shape[:2]

    tensor = preprocess_detection(image_bgr, input_h, input_w)
    t0 = time.perf_counter()
    det_out = det_compiled([tensor])
    t1 = time.perf_counter()
    det_ms = (t1 - t0) * 1000.0

    boxes, _ = postprocess_detection(det_out, orig_h, orig_w, input_w, input_h, thresh)

    cin = cls_compiled.input(0)
    csh = cin.get_partial_shape()
    ch = int(csh[2].get_length()) if csh[2].is_static else 224
    cw = int(csh[3].get_length()) if csh[3].is_static else 224

    cls_ms = 0.0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        cls_tensor = preprocess_classification(crop, ch, cw)
        t0 = time.perf_counter()
        cls_compiled([cls_tensor])
        t1 = time.perf_counter()
        cls_ms += (t1 - t0) * 1000.0

    return det_ms, cls_ms, len(boxes)


def warmup_two_stage_pipeline(det_compiled, cls_compiled, image_bgr, thresh=0.225):
    """One full detection + classification run to warm devices; timings ignored."""
    timed_two_stage_pipeline_ms(det_compiled, cls_compiled, image_bgr, thresh)


def benchmark_two_stage_per_device(
    core: Any,
    detection_xml: Union[str, Path],
    classification_xml: Union[str, Path],
    image_bgr,
    thresh: float = 0.225,
    devices: Optional[Sequence[str]] = None,
    shared_detection_device: Optional[str] = None,
) -> List[dict]:
    """
    Times one image: detection always uses ``shared_detection_device`` (default GPU if present, else CPU).
    Each row runs classification on a different device (from ``devices``).

    Default ``devices`` is CPU and GPU only. Pass more names (e.g. NPU) if you want extra rows.

    Returns one dict per row: device, det_device, cls_device, det_ms, cls_ms, total_ms, n_boxes.
    """
    detection_xml = Path(detection_xml)
    classification_xml = Path(classification_xml)
    if devices is None:
        devices = [d for d in ("CPU", "GPU") if d in core.available_devices]
    if shared_detection_device is None:
        shared_detection_device = "GPU" if "GPU" in core.available_devices else "CPU"
    if shared_detection_device not in core.available_devices:
        raise ValueError(
            f"shared_detection_device={shared_detection_device!r} not in {core.available_devices}"
        )
    rows: List[dict] = []
    dm = core.read_model(str(detection_xml))
    cm = core.read_model(str(classification_xml))
    det_dev = shared_detection_device
    cd = core.compile_model(dm, det_dev)
    # One untimed end-to-end pass warms the shared detector (GPU/NPU init, caches).
    _cw = core.compile_model(cm, "CPU")
    timed_two_stage_pipeline_ms(cd, _cw, image_bgr, thresh)
    for dev in devices:
        dev_u = str(dev).upper()
        if dev_u == "NPU":
            try:
                cc = core.compile_model(cm, "NPU")
                cls_dev = "NPU"
            except Exception:
                cc = core.compile_model(cm, "CPU")
                cls_dev = "CPU"
        else:
            cc = core.compile_model(cm, dev)
            cls_dev = dev
        dms, cms, nbox = timed_two_stage_pipeline_ms(cd, cc, image_bgr, thresh)
        rows.append(
            {
                "device": dev,
                "det_device": det_dev,
                "cls_device": cls_dev,
                "det_ms": dms,
                "cls_ms": cms,
                "total_ms": dms + cms,
                "n_boxes": nbox,
            }
        )
    return rows


def print_two_stage_benchmark_table(rows: List[dict], precision_label: str) -> None:
    """Print a text table from benchmark_two_stage_per_device rows."""
    print(f"Precision: {precision_label}")
    hdr = f"{'Classify':<8} {'Det':>6} {'Cls':>6} {'Det_ms':>10} {'Cls_ms':>10} {'Total_ms':>10} {'Boxes':>5}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        print(
            f"{r['device']:<8} {str(r['det_device']):>6} {str(r['cls_device']):>6} "
            f"{r['det_ms']:10.2f} {r['cls_ms']:10.2f} {r['total_ms']:10.2f} {r['n_boxes']:5d}"
        )


def plot_detection_classification_benchmark(
    rows: List[dict],
    precision_label: str,
    device_colors: Optional[Sequence[str]] = None,
    figsize: Optional[tuple] = None,
) -> None:
    """Two bar charts: detection time and classification time per row (one bar group per device)."""
    import matplotlib.pyplot as plt

    if device_colors is None:
        device_colors = ["#0072B2", "#E69F00", "#009E73"]
    devices = [r["device"] for r in rows]
    n_dev = len(devices)
    if n_dev == 0:
        print("Nothing to plot (no devices).")
        return
    x_pos = np.arange(n_dev)

    def col(key: str):
        return [float(r[key]) for r in rows]

    det_v = col("det_ms")
    cls_v = col("cls_ms")
    bar_colors = [device_colors[i % len(device_colors)] for i in range(n_dev)]
    w = 0.62
    if figsize is None:
        figsize = (max(6.0, 0.65 * n_dev + 3.5), 7.2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.bar(x_pos, det_v, width=w, color=bar_colors, edgecolor="#bbbbbb", linewidth=0.6)
    ax1.set_ylabel("Detection time (ms)")
    ax1.set_title(f"Detection time (ms) — {precision_label}")
    ax1.grid(axis="y", alpha=0.35)

    ax2.bar(x_pos, cls_v, width=w, color=bar_colors, edgecolor="#bbbbbb", linewidth=0.6)
    ax2.set_ylabel("Classification time (ms)")
    ax2.set_xlabel("Device")
    ax2.set_title(f"Classification time (ms) — {precision_label}")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(devices)
    ax2.grid(axis="y", alpha=0.35)
    if n_dev:
        ax2.set_xlim(x_pos[0] - 0.55, x_pos[-1] + 0.55)
    plt.tight_layout()
    plt.show()


# --- Backward-compatible names (older notebooks and imports) ---
get_detection_input_hw = detection_input_hw
visualize_detection_boxes = draw_detection_boxes
visualize_labeled_boxes = draw_labeled_boxes
plot_benchmark_det_cls = plot_detection_classification_benchmark


DEFAULT_DETECTION_CONFIDENCE_THRESHOLD = 0.225

# --- Device introspection (same idea as original OpenVINO notebook) ---


def print_device_details(core) -> None:
    """Print FULL_DEVICE_NAME, OPTIMIZATION_CAPABILITIES, RANGE_FOR_STREAMS per device."""
    for device in core.available_devices:
        print("Device full name: ", core.get_property(device, "FULL_DEVICE_NAME"))
        print("Supported formats: ", core.get_property(device, "OPTIMIZATION_CAPABILITIES"))
        print("Range for multiple streams: ", core.get_property(device, "RANGE_FOR_STREAMS"))
        print("-------------------------")


def model_xml_paths(models_root: str | Path, precision: str) -> Tuple[Path, Path]:
    """Detection and Classification IR paths for a precision folder (FP32/FP16/INT8)."""
    root = Path(models_root)
    det = root / precision / "Detection" / "model.xml"
    cls = root / precision / "Classification" / "model.xml"
    return det, cls


# --- Input prep (boss-style name: returns tensor + original H,W) ---


def prepare_detection_tensor(image_bgr: np.ndarray, compiled_detection) -> Tuple[np.ndarray, int, int]:
    """
    Resize/normalize for the detection model (Geti/OTX convention).
    Returns ``(input_tensor_nchw, orig_h, orig_w)``.
    """
    input_h, input_w = detection_input_hw(compiled_detection)
    orig_h, orig_w = image_bgr.shape[:2]
    tensor = preprocess_detection(image_bgr, input_h, input_w)
    return tensor, orig_h, orig_w


# Alias for notebooks that used ``resize_image`` + ``draw`` naming from bm_utils ---
resize_image = prepare_detection_tensor


def draw(image: np.ndarray, boxes: np.ndarray, labels: Sequence[str] | None = None) -> np.ndarray:
    """
    Draw boxes on image. If ``labels`` is None, draws a single class chip (CAR);
    otherwise per-box strings (detection + classification).
    """
    if labels is None:
        return draw_detection_boxes(image, boxes, label="CAR")
    return draw_labeled_boxes(image, boxes, list(labels))


# --- Benchmarks (Python timing; optional ``benchmark_app`` is separate) ---


def benchmark_latency_ms(compiled, input_tensor, num_iter: int = 50) -> float:
    """Mean inference latency (ms) over ``num_iter`` runs after one warmup."""
    compiled([input_tensor])
    times: List[float] = []
    for _ in range(num_iter):
        t0 = time.perf_counter()
        compiled([input_tensor])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times))


def benchmark_throughput_ips(compiled, input_tensor, duration_sec: float = 2.0) -> Tuple[float, int]:
    """
    Approximate inferences per second (IPS) by running for ``duration_sec`` after warmup.
    Returns (ips, count).
    """
    compiled([input_tensor])
    t_end = time.perf_counter() + duration_sec
    n = 0
    while time.perf_counter() < t_end:
        compiled([input_tensor])
        n += 1
    ips = n / duration_sec
    return float(ips), n


def run_pipeline_latency_ms(
    det_compiled,
    cls_compiled,
    image_bgr: np.ndarray,
    thresh: float = DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
) -> Tuple[float, float, int]:
    """
    One end-to-end timed run: detection time (ms), classification time (sum of crops, ms), box count.
    ``thresh`` is the detection confidence threshold (see ``DEFAULT_DETECTION_CONFIDENCE_THRESHOLD``).
    """
    return timed_two_stage_pipeline_ms(det_compiled, cls_compiled, image_bgr, thresh)


# --- Gradio-oriented: list media images ---


def list_media_images(media_dir: str | Path) -> List[str]:
    """Basenames of images under ``media_dir``."""
    media_dir = Path(media_dir)
    if not media_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for p in sorted(media_dir.iterdir()):
        if p.suffix.lower() in exts:
            out.append(p.name)
    return out


# Globals (same pattern as bm_utils.py)
results_data = pd.DataFrame(
    columns=["model", "precision", "device", "goal", "latency", "throughput", "marker", "color"]
)

reports_dir = "."

_models_root = "models"
_model_key = "car_detection"

# One logical model (Geti detection), three precisions — marker/color for scatter plot
model_dict = {
    "car_detection-FP32": ("o", "tab:blue", f"{_models_root}/FP32/Detection/model.xml"),
    "car_detection-FP16": ("s", "tab:orange", f"{_models_root}/FP16/Detection/model.xml"),
    "car_detection-INT8": ("^", "tab:green", f"{_models_root}/INT8/Detection/model.xml"),
}


def set_models_root(path: str) -> None:
    """Point IR paths at your ``models`` folder (default ``models`` next to this file)."""
    global _models_root, model_dict
    _models_root = path.rstrip("/\\")
    model_dict = {
        "car_detection-FP32": ("o", "tab:blue", f"{_models_root}/FP32/Detection/model.xml"),
        "car_detection-FP16": ("s", "tab:orange", f"{_models_root}/FP16/Detection/model.xml"),
        "car_detection-INT8": ("^", "tab:green", f"{_models_root}/INT8/Detection/model.xml"),
    }


_re_median = re.compile(r"Median:\s*([\d.]+(?:[eE][+-]?\d+)?)")
_re_throughput = re.compile(r"Throughput:\s*([\d.]+(?:[eE][+-]?\d+)?)")


def parse_output(file_path, goal):
    median_latency = np.nan
    throughput = np.nan

    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            text = file.read()
        for line in text.splitlines():
            if "Median:" in line and "[ INFO ]" in line:
                m = _re_median.search(line)
                if m:
                    median_latency = float(m.group(1))
            if "Throughput:" in line and "[ INFO ]" in line:
                m = _re_throughput.search(line)
                if m:
                    throughput = float(m.group(1))
    else:
        raise FileNotFoundError(f"{file_path} could not be found.")

    return median_latency, throughput


def update_plot(model_name, precision, device, goal, results):
    global results_data, model_dict

    key_name = f"{model_name}-{precision}"
    marker, color, _ = model_dict[key_name]
    new_row = pd.DataFrame(
        {
            "model": [model_name],
            "precision": [precision],
            "device": [device],
            "goal": [goal],
            "latency": [results[0]],
            "throughput": [results[1]],
            "marker": marker,
            "color": color,
        }
    )
    results_data = pd.concat([results_data, new_row], ignore_index=True)
    fig, ax = plt.subplots()

    for _, row in results_data.iterrows():
        if row["goal"] == goal and row["model"] == model_name:
            y = row[goal]
            if pd.isna(y):
                continue
            plt.scatter(row["device"], y, marker=row["marker"], color=row["color"])

    results_models = results_data.drop_duplicates(subset=["model", "precision", "goal"])
    for _, row in results_models.iterrows():
        if row["goal"] == goal and row["model"] == model_name:
            plt.scatter([], [], label=row["precision"], marker=row["marker"], color=row["color"])

    ax.set_xlabel("Device")
    if goal == "latency":
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{model_name} Latency by Device")
    else:
        ax.set_ylabel("Throughput (FPS)")
        ax.set_title(f"{model_name} Throughput by Device")

    ax.legend(loc="best")
    return fig


def run_benchmark(model_name, precision, device, hint):
    global model_dict, reports_dir

    key_name = f"{model_name}-{precision}"
    entry = model_dict.get(key_name)
    if entry is None:
        raise KeyError(f"Unknown {key_name!r}; check model_dict and models_root.")
    _, _, model_path = entry
    model_path = os.path.normpath(model_path)

    if device == "AUTO throughput":
        bm_cmd_hint = "cumulative_throughput"
        device = "AUTO:GPU,NPU,CPU"
    else:
        bm_cmd_hint = hint

    file_path = f"{reports_dir}/{model_name}-{precision}-{device}-{hint}.txt"
    file_path = file_path.replace(",", "-").replace(":", "-")
    os.makedirs(reports_dir, exist_ok=True)

    # OpenVINO 2025+/2026 CPU (and typical GPU) plugins no longer accept int8 for
    # INFERENCE_PRECISION_HINT; use undefined so the compiled model follows the IR.
    if precision == "INT8":
        if device == "NPU":
            precision_arg = "-infer_precision i8"
        else:
            precision_arg = "-infer_precision undefined"
    elif precision == "FP16":
        precision_arg = "-infer_precision f16"
    else:
        precision_arg = "-infer_precision f32"

    mp = model_path.replace("\\", "/")
    bm_command = (
        f'benchmark_app -m "{mp}" -hint {bm_cmd_hint} -d {device} {precision_arg} -t 30'
    )
    os.system(f"{bm_command} 1> {file_path} 2>&1")

    results = parse_output(file_path, hint)
    cmd_out = bm_command
    y_metric = results[0] if hint == "latency" else results[1]
    if pd.isna(y_metric) and os.path.isfile(file_path):
        err_hint = ""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "[ ERROR ]" in line or line.strip().startswith("RuntimeError"):
                    err_hint = line.strip()[:500]
                    break
        if err_hint:
            cmd_out = (
                f"{bm_command}\n\n"
                f"(No Median/Throughput in log — benchmark may have failed.)\n{err_hint}"
            )

    fig = update_plot(model_name, precision, device, hint, results)
    return fig, cmd_out


def clear_plot_results(model_name, goal):
    global results_data

    results_data = pd.DataFrame(
        columns=["model", "precision", "device", "goal", "latency", "throughput", "marker", "color"]
    )

    fig, ax = plt.subplots()
    ax.set_xlabel("Device")
    if goal == "latency":
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{model_name} Latency by Device")
    else:
        ax.set_ylabel("Throughput (FPS)")
        ax.set_title(f"{model_name} Throughput by Device")

    ax.legend(loc="best")
    return fig

