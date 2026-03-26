"""
Lab / benchmark helpers for the detection + classification notebooks.
Preprocess, two-stage timing, and fair cross-device benchmark (shared detector).
"""
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import cv2
import numpy as np


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
    One timed synchronous run: detection then classification per box.
    Uses preprocess_detection / postprocess_detection / preprocess_classification.

    Returns:
        (det_ms, cls_ms, n_boxes) — cls_ms is the sum over crops.
    """
    inp = det_compiled.input(0)
    sh = inp.get_partial_shape()
    input_h = int(sh[2].get_length()) if sh[2].is_static else 800
    input_w = int(sh[3].get_length()) if sh[3].is_static else 992
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
    """One full det+cls run to warm NPU/driver caches; ignores returned timings."""
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
    Fair cross-device timing: **detection** is always compiled on ``shared_detection_device``
    (default **GPU** if listed, else **CPU**). Each row only changes **where classification runs**.

    Default ``devices`` is **CPU** and **GPU** only (no NPU). Pass e.g. ``devices=(\"CPU\", \"GPU\", \"NPU\")``
    filtered by availability to include NPU.

    Returns list of dicts with: device (cls target), det_device, cls_device, det_ms, cls_ms, total_ms, n_boxes.
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
    print(f"Benchmark precision: {precision_label}")
    hdr = f"{'Cls_on':<8} {'Det@':>6} {'Cls@':>6} {'Det_ms':>10} {'Cls_ms':>10} {'Total_ms':>10} {'Boxes':>5}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        print(
            f"{r['device']:<8} {str(r['det_device']):>6} {str(r['cls_device']):>6} "
            f"{r['det_ms']:10.2f} {r['cls_ms']:10.2f} {r['total_ms']:10.2f} {r['n_boxes']:5d}"
        )


def plot_benchmark_det_cls(
    rows: List[dict],
    precision_label: str,
    device_colors: Optional[Sequence[str]] = None,
    figsize: Optional[tuple] = None,
) -> None:
    """Two stacked bar charts: detection ms and classification ms per device."""
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
    ax1.set_ylabel("Detection (ms)")
    ax1.set_title(f"Detection latency (shared Det@) — {precision_label}")
    ax1.grid(axis="y", alpha=0.35)

    ax2.bar(x_pos, cls_v, width=w, color=bar_colors, edgecolor="#bbbbbb", linewidth=0.6)
    ax2.set_ylabel("Classification (ms)")
    ax2.set_xlabel("Device")
    ax2.set_title(f"Classification latency per Cls_on row — {precision_label}")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(devices)
    ax2.grid(axis="y", alpha=0.35)
    if n_dev:
        ax2.set_xlim(x_pos[0] - 0.55, x_pos[-1] + 0.55)
    plt.tight_layout()
    plt.show()
