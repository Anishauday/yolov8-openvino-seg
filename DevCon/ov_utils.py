"""
Car detection + color classification — Intel geti deployment.
Two-stage pipeline: detect cars, classify color per crop.
"""
import os
import glob
import json
import textwrap
import time
import numpy as np
import cv2


def get_sample_images_by_prefix(media_dir, prefix="sample_image_", max_count=3):
    """
    Return up to max_count sorted paths to images in media_dir whose filename
    starts with prefix (e.g. sample_image_1.jpg). Uses glob sample_image_*.
    """
    if not os.path.isdir(media_dir):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for p in glob.glob(os.path.join(media_dir, f"{prefix}*")):
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
            paths.append(os.path.normpath(os.path.abspath(p)))
    paths = sorted(paths)[:max_count]
    return paths


def get_sample_images(base_dir=".", exts=None):
    """Return list of (label, path) for Gradio. Scans base_dir (e.g. media/)."""
    if exts is None:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    for d in (base_dir, os.path.join(base_dir, "media")):
        if not os.path.exists(d):
            continue
        for ext in exts:
            paths.extend(glob.glob(os.path.join(d, ext)))
    seen = set()
    result = []
    for p in sorted(paths):
        if not os.path.isfile(p):
            continue
        name = os.path.basename(p)
        if name in seen:
            continue
        seen.add(name)
        result.append((name, os.path.normpath(os.path.abspath(p))))
    return result


def load_config(model_dir):
    """Load config.json for labels."""
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {"model_parameters": {"labels": "object", "label_ids": ""}}


def get_available_devices(exclude=None):
    """Return list of available OpenVINO devices. exclude: optional list to filter out."""
    import openvino as ov
    devs = list(ov.Core().available_devices)
    if exclude:
        devs = [d for d in devs if d.upper() not in {e.upper() for e in exclude}]
    return devs


def is_transient_npu_device_error(exc: BaseException) -> bool:
    """Match Intel NPU / Level Zero runtime errors (inference or driver messages)."""
    s = str(exc).lower()
    needles = (
        "device_lost",
        "ze_result_error_device_lost",
        "0x70000001",
        "device hung",
        "was removed",
        "driver update",
        "memallochost",
    )
    return any(n in s for n in needles)


def _device_name_is_npu(device: str) -> bool:
    return "NPU" in (device or "").upper()


def load_model(model_dir, device="CPU", return_device=False):
    """
    Load OpenVINO IR. Returns (compiled, config), or (compiled, config, effective_device) if return_device=True.
    If compile fails on an NPU device name, retries once on CPU (same as Gradio fallback intent).
    """
    import openvino as ov
    xml_path = os.path.join(model_dir, "model.xml")
    xml_path = os.path.normpath(os.path.abspath(xml_path))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model not found: {xml_path}")
    core = ov.Core()
    model = core.read_model(xml_path)
    effective = device
    try:
        compiled = core.compile_model(model, device)
    except Exception as first_err:
        if _device_name_is_npu(device):
            try:
                compiled = core.compile_model(model, "CPU")
                effective = "CPU"
            except Exception:
                raise first_err from None
        else:
            raise first_err
    config = load_config(model_dir)
    if return_device:
        return compiled, config, effective
    return compiled, config


def get_model_name(model_dir):
    """Return model name from model.json if present."""
    for d in (model_dir, os.path.dirname(model_dir)):
        model_json = os.path.join(d, "model.json")
        if os.path.exists(model_json):
            with open(model_json) as f:
                data = json.load(f)
                return data.get("name", os.path.basename(model_dir))
    return os.path.basename(model_dir)


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


def preprocess_classification(crop, input_h=224, input_w=224):
    """Classification preprocess: RGB, resize to 224x224, (pixel - mean) / scale (matches Geti/OTX)."""
    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h))
    normalized = (resized.astype(np.float32) - MEAN) / SCALE
    nchw = np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
    return nchw


def run_classification(compiled, crop, labels_list):
    """Run classification on crop. Returns color label string."""
    tensor = preprocess_classification(crop)
    out = compiled([tensor])
    logits = np.squeeze(list(out.values())[0])
    idx = int(np.argmax(logits))
    if 0 <= idx < len(labels_list):
        return labels_list[idx].strip()
    return "Unknown"


def draw_boxes_labels(image, boxes, labels_list, color=(220, 220, 180), thickness=2):
    """Draw boxes with 'CAR - Color' labels. No percentage. Pale-yellow style."""
    font_scale = 0.75
    font_thickness = 2
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = labels_list[i] if i < len(labels_list) else "CAR"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), font_thickness
        )
    return img


def run_detect_and_classify(det_compiled, cls_compiled, image, det_label="CAR", thresh=0.225,
                            cls_labels=None, return_latency=False):
    """
    Two-stage pipeline: detect cars, classify color per crop.
    Returns (boxes, scores, color_labels, result_img) or with lat_ms if return_latency.
    Per-box label format: "CAR - [Color]" (no percentage).
    """
    if cls_labels is None:
        cls_labels = ["Silver", "Green", "Brown", "Blue", "White", "Black", "Yellow", "Red"]
    if isinstance(cls_labels, str):
        cls_labels = [s.strip() for s in cls_labels.split()]

    inp = det_compiled.input(0)
    sh = inp.get_partial_shape()
    d2, d3 = sh[2], sh[3]
    input_h = int(d2.get_length()) if d2.is_static else 800
    input_w = int(d3.get_length()) if d3.is_static else 992
    orig_h, orig_w = image.shape[:2]

    t0 = time.perf_counter()
    tensor = preprocess_detection(image, input_h, input_w)
    det_out = det_compiled([tensor])
    boxes, scores = postprocess_detection(det_out, orig_h, orig_w, input_w, input_h, thresh)

    color_labels = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            color_labels.append("Unknown")
            continue
        color = run_classification(cls_compiled, crop, cls_labels)
        color_labels.append(f"{det_label.upper()} - {color}")

    lat_ms = (time.perf_counter() - t0) * 1000
    result = draw_boxes_labels(image, boxes, color_labels)
    if return_latency:
        return boxes, scores, color_labels, result, lat_ms
    return boxes, scores, color_labels, result


def run_detect_and_classify_video(det_compiled, cls_compiled, video_path, output_path=None,
                                  thresh=0.225, det_label="CAR", cls_labels=None, device="CPU"):
    """Process video frame-by-frame. Write output with device + latency overlay."""
    if cls_labels is None:
        cls_labels = ["Silver", "Green", "Brown", "Blue", "White", "Black", "Yellow", "Red"]
    if isinstance(cls_labels, str):
        cls_labels = [s.strip() for s in cls_labels.split()]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if str(output_path).lower().endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, scores, color_labels, result, lat_ms = run_detect_and_classify(
            det_compiled, cls_compiled, frame, det_label=det_label, thresh=thresh,
            cls_labels=cls_labels, return_latency=True
        )
        txt = f"{device} {lat_ms:.0f} ms"
        cv2.putText(result, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if writer:
            writer.write(result)
        frame_count += 1
    cap.release()
    if writer:
        writer.release()
    return output_path, frame_count, fps


def get_notebook_config(base_dir=None):
    """
    Return config dict for notebook: BASE, paths, precisions, devices, defaults, THRESH.
    base_dir: folder containing models/, media/, output/. If None, infer from cwd.
    """
    if base_dir is None:
        base_dir = os.getcwd() if os.path.basename(os.getcwd()) == "1" else os.path.join(os.getcwd(), "1")
    models_dir = os.path.join(base_dir, "models")
    precisions = get_available_model_precisions(models_dir)
    devices = get_available_devices()
    default_precision = precisions[0] if precisions else "INT8"
    default_device = "CPU" if "CPU" in devices else (devices[0] if devices else "CPU")
    samples = get_sample_images(os.path.join(base_dir, "media"))
    image_path = os.path.join(base_dir, "media", "sample_image.jpg")
    if not os.path.exists(image_path) and samples:
        image_path = samples[0][1]
    return {
        "BASE": base_dir,
        "MODEL_DIR_DET": lambda p: os.path.join(base_dir, "models", p, "Detection"),
        "MODEL_DIR_CLS": lambda p: os.path.join(base_dir, "models", p, "Classification"),
        "IMAGE_PATH": image_path,
        "VIDEO_PATH": os.path.join(base_dir, "media", "Car_video.mp4"),
        "OUTPUT_VIDEO_PATH": os.path.join(base_dir, "output", "Car_video_out.avi"),
        "OUTPUT_IMAGE_PATH": os.path.join(base_dir, "output", "sample_image_out.jpg"),
        "precisions": precisions,
        "devices": devices,
        "default_precision": default_precision,
        "default_device": default_device,
        "samples": samples,
        "THRESH": 0.15,
    }


def load_models_for_notebook(precision, device, base_dir=None):
    """
    Load detection + classification models. Returns (det_compiled, cls_compiled, det_label, cls_labels).
    NPU compile failures fall back to CPU inside load_model.
    """
    cfg = get_notebook_config(base_dir)
    if precision not in cfg["precisions"]:
        raise FileNotFoundError(f"Precision {precision} not in {cfg['precisions']}")
    det_dir = cfg["MODEL_DIR_DET"](precision)
    cls_dir = cfg["MODEL_DIR_CLS"](precision)
    det_compiled, det_cfg, dev_det = load_model(det_dir, device=device, return_device=True)
    cls_compiled, cls_cfg, dev_cls = load_model(cls_dir, device=device, return_device=True)
    if device.upper() != "CPU" and (dev_det == "CPU" or dev_cls == "CPU"):
        print(f"Note: models compiled on CPU (NPU compile for '{device}' failed).")
    det_label = det_cfg.get("model_parameters", {}).get("labels", "CAR")
    lbl = cls_cfg.get("model_parameters", {}).get("labels", "")
    cls_labels = [s.strip() for s in lbl.split()] if lbl else []
    return det_compiled, cls_compiled, det_label, cls_labels


# --- Jupyter: precision + device + Apply (single row), shared session for downstream cells ---
_notebook_session = None  # dict or None


def get_notebook_session():
    """Return dict from last Apply in show_openvino_config_widgets; keys: det_compiled, cls_compiled, det_label, cls_labels, precision, device, base_dir."""
    global _notebook_session
    if _notebook_session is None:
        raise RuntimeError("Run the configuration cell and click Apply (or wait for auto-apply).")
    return _notebook_session


def show_openvino_config_widgets(base_dir=None, auto_apply=True):
    """
    Jupyter: precision + device dropdowns and Apply in one row; loads models and sets OV_PRECISION / OV_DEVICE.
    Call get_notebook_session() in later cells for compiled models and labels.
    """
    global _notebook_session
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError:
        cfg = get_notebook_config(base_dir)
        os.environ["OV_PRECISION"] = cfg["default_precision"]
        os.environ["OV_DEVICE"] = cfg["default_device"]
        p, d = cfg["default_precision"], cfg["default_device"]
        det, cls_, dl, cl = load_models_for_notebook(p, d, base_dir)
        _notebook_session = {
            "det_compiled": det,
            "cls_compiled": cls_,
            "det_label": dl,
            "cls_labels": cl,
            "precision": p,
            "device": d,
            "base_dir": cfg["BASE"],
        }
        print("ipywidgets missing; loaded defaults:", p, d)
        return

    cfg = get_notebook_config(base_dir)
    prec_opts = cfg["precisions"] or ["INT8"]
    dev_opts = cfg["devices"] or ["CPU"]
    prec_w = w.Dropdown(
        options=prec_opts,
        value=cfg["default_precision"] if cfg["default_precision"] in prec_opts else prec_opts[0],
        description="Precision:",
        layout=w.Layout(width="200px"),
    )
    dev_w = w.Dropdown(
        options=dev_opts,
        value=cfg["default_device"] if cfg["default_device"] in dev_opts else dev_opts[0],
        description="Device:",
        layout=w.Layout(width="200px"),
    )
    btn = w.Button(description="Apply", button_style="primary", layout=w.Layout(width="100px"))

    def _apply(_=None):
        global _notebook_session
        p, d = prec_w.value, dev_w.value
        os.environ["OV_PRECISION"] = p
        os.environ["OV_DEVICE"] = d
        det, cls_, dl, cl = load_models_for_notebook(p, d, cfg["BASE"])
        _notebook_session = {
            "det_compiled": det,
            "cls_compiled": cls_,
            "det_label": dl,
            "cls_labels": cl,
            "precision": p,
            "device": d,
            "base_dir": cfg["BASE"],
        }
        print(
            get_model_name(cfg["MODEL_DIR_DET"](p)),
            "|",
            get_model_name(cfg["MODEL_DIR_CLS"](p)),
            "|",
            dl,
            cl,
        )

    btn.on_click(_apply)
    row = w.HBox([prec_w, dev_w, btn], layout=w.Layout(flex_flow="row wrap", align_items="center", gap="8px"))
    display(row)
    if auto_apply:
        _apply()


def letterbox_rgb_display(rgb: np.ndarray, tw: int, th: int, pad_rgb=(32, 32, 32)) -> np.ndarray:
    """Resize RGB image to fit inside tw x th, preserve aspect, pad."""
    h, w = rgb.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.empty((th, tw, 3), dtype=np.uint8)
    out[:, :] = pad_rgb
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def format_grid_caption_line(line: str, width: int = 88, max_lines: int = 2) -> str:
    """Wrap caption to max_lines; prefer break before 'Details:'."""
    if len(line) <= width:
        return line
    if "Details:" in line and len(line) > width:
        head, sep, tail = line.partition("Details:")
        head = head.rstrip()
        if len(head) + len(sep) + len(tail) > width and len(head) < width:
            return head + "\n" + sep + tail.strip()
    lines = textwrap.wrap(line, width=width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[: max_lines - 1]) + "\n" + lines[max_lines - 1][: width - 3] + "..."


def plot_sample_inference_grid(
    det_compiled,
    cls_compiled,
    det_label,
    cls_labels,
    thresh,
    media_dir,
    device_label: str,
    prefix: str = "sample_image_",
    max_count: int = 3,
    canvas_w: int = 720,
    canvas_h: int = 540,
):
    """
    Run inference on sample_image_* under media_dir and show a single row of letterboxed panels
    with one caption line (wrapped) under each image.
    """
    import matplotlib.pyplot as plt

    sample_paths = get_sample_images_by_prefix(media_dir, prefix=prefix, max_count=max_count)
    if not sample_paths:
        print("No sample_image_* under media/ — add up to 3 files.")
        return

    panels = []
    for path in sample_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        boxes, _scores, color_labels, result_img, lat_ms = run_detect_and_classify(
            det_compiled,
            cls_compiled,
            img,
            det_label=det_label,
            cls_labels=cls_labels,
            thresh=thresh,
            return_latency=True,
        )
        details = "; ".join(color_labels) if color_labels else "(none)"
        line = (
            f"Detections: {len(boxes)} | Latency: {lat_ms:.2f} ms | "
            f"Device: {device_label} | Details: {details}"
        )
        rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        rgb_box = letterbox_rgb_display(rgb, canvas_w, canvas_h)
        panels.append((rgb_box, format_grid_caption_line(line)))

    if not panels:
        print("No readable sample images.")
        return

    n = len(panels)
    fig_w = max(14.0, 4.2 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 7.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (rgb_box, cap) in zip(axes, panels):
        ax.imshow(rgb_box, aspect="equal")
        ax.axis("off")
        ax.text(
            0.5,
            -0.02,
            cap,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            linespacing=1.35,
        )
    plt.show()


def get_available_model_precisions(models_dir):
    """Return list of precision folders (e.g. INT8, FP16) that contain Detection and Classification."""
    if not os.path.exists(models_dir):
        return []
    precisions = []
    for name in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, name)
        if not os.path.isdir(path):
            continue
        det_path = os.path.join(path, "Detection", "model.xml")
        cls_path = os.path.join(path, "Classification", "model.xml")
        if os.path.exists(det_path) and os.path.exists(cls_path):
            precisions.append(name)
    return precisions


def benchmark_int8_across_devices(models_dir, image, devices=None, thresh=0.15):
    """
    Benchmark INT8 pipeline (detect + classify) across CPU, GPU, NPU.
    Returns (results_dict, fig): dict device->latency_ms, matplotlib figure.
    Skips devices that fail.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if devices is None:
        devices = get_available_devices()
    if "INT8" not in get_available_model_precisions(models_dir):
        return {}, None

    det_dir = os.path.join(models_dir, "INT8", "Detection")
    cls_dir = os.path.join(models_dir, "INT8", "Classification")
    cls_cfg = load_config(cls_dir)
    cls_labels = [s.strip() for s in (cls_cfg.get("model_parameters", {}).get("labels", "") or "").split()]

    results = {}
    for dev in devices:
        try:
            det_c, det_cfg = load_model(det_dir, device=dev)
            cls_c, _ = load_model(cls_dir, device=dev)
            _, _, _, _, lat_ms = run_detect_and_classify(
                det_c, cls_c, image, cls_labels=cls_labels, thresh=thresh, return_latency=True
            )
            results[dev] = lat_ms
        except Exception:
            pass

    if not results:
        return results, None

    fig, ax = plt.subplots(figsize=(6, 4))
    devs = list(results.keys())
    lats = [results[d] for d in devs]
    colors = ["#0071C5", "#00A3E0", "#6FBD44"][: len(devs)]
    ax.bar(devs, lats, color=colors)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("INT8 Inference Time by Device")
    for i, v in enumerate(lats):
        ax.text(devs[i], v + 1, f"{v:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    return results, fig


def display_output_video(output_video_path, max_frames=60, scale=0.5, fps=10):
    """
    Convert output video to GIF for notebook display. Handles .avi and .mp4.
    Returns (gif_path, message). Use: gif_path, msg = display_output_video(OUTPUT_VIDEO_PATH)
    """
    avi = output_video_path
    mp4 = str(output_video_path).replace(".avi", ".mp4") if output_video_path else ""
    if avi and os.path.exists(avi):
        p = convert_video_to_gif(avi, max_frames=max_frames, scale=scale, fps=fps)
        return p, f"Display: {p}"
    if mp4 and os.path.exists(mp4):
        p = convert_video_to_gif(mp4, max_frames=max_frames, scale=scale, fps=fps)
        return p, f"Display: {p}"
    return None, "Run video inference first (section 6)."


def convert_video_to_gif(input_path, output_path=None, max_frames=60, scale=0.5, fps=10):
    """Convert video to GIF for browser display."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install Pillow for GIF export")
    out = output_path or str(input_path).rsplit(".", 1)[0] + ".gif"
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(vid_fps / fps))
    frames = []
    n = 0
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        if n % frame_interval == 0:
            if scale != 1.0:
                w, h = int(f.shape[1] * scale), int(f.shape[0] * scale)
                f = cv2.resize(f, (w, h))
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        n += 1
    cap.release()
    if frames:
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
    return out
