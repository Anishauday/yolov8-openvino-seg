"""
Lab 1 inference utilities: preprocessing, postprocessing, model loading.
Keeps the notebook minimal and logic easy to debug.
"""
import os
import glob
import time
import subprocess
import tarfile
import urllib.request
import numpy as np
import cv2

# YOLOX model URLs (Megvii GitHub releases)
YOLOX_URLS = {
    "yolox_s": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s_openvino.tar.gz",
    "yolox_l": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l_openvino.tar.gz",
}


def _ssdlite_model_exists(output_dir):
    """Check if ssdlite_mobilenet_v2 FP32 and FP16 IR already exist."""
    pattern = os.path.join(output_dir, "**", "FP32", "*.xml")
    fp32 = glob.glob(pattern, recursive=True)
    pattern = os.path.join(output_dir, "**", "FP16", "*.xml")
    fp16 = glob.glob(pattern, recursive=True)
    return len(fp32) > 0 and len(fp16) > 0


def download_ssdlite_mobilenet_v2(output_dir="models_test"):
    """Download and convert ssdlite_mobilenet_v2 to FP32 and FP16.
    Skips download/conversion if model already exists."""
    if _ssdlite_model_exists(output_dir):
        return output_dir  # Model already available, skip download
    subprocess.run(
        ["omz_downloader", "--name", "ssdlite_mobilenet_v2", "-o", output_dir],
        check=True,
    )
    subprocess.run(
        [
            "omz_converter",
            "--name", "ssdlite_mobilenet_v2",
            "--precisions", "FP32,FP16",
            "-d", output_dir,
            "-o", output_dir,
        ],
        check=True,
    )
    return output_dir


def get_available_devices():
    """Return list of available OpenVINO devices (e.g. ['CPU', 'GPU'])."""
    import openvino as ov
    return ov.Core().available_devices


# --- YOLOX (yolox_s, yolox_l) ---

def _yolox_model_exists(output_dir, model_name):
    """Check if YOLOX model xml exists."""
    xml_path = os.path.join(output_dir, model_name, f"{model_name}.xml")
    return os.path.exists(xml_path)


def download_yolox(model_name, output_dir="models_test"):
    """Download YOLOX-S or YOLOX-L OpenVINO IR from Megvii. Skips if already exists."""
    if model_name not in YOLOX_URLS:
        raise ValueError(f"Unknown model {model_name}. Use yolox_s or yolox_l.")
    if _yolox_model_exists(output_dir, model_name):
        return output_dir
    os.makedirs(output_dir, exist_ok=True)
    url = YOLOX_URLS[model_name]
    tar_path = os.path.join(output_dir, f"{model_name}_openvino.tar.gz")
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as t:
        t.extractall(output_dir)
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    for f in [f"{model_name}.xml", f"{model_name}.bin", f"{model_name}.mapping"]:
        src = os.path.join(output_dir, f)
        if os.path.exists(src):
            os.rename(src, os.path.join(model_dir, f))
    if os.path.exists(tar_path):
        os.remove(tar_path)
    return output_dir


def preprocess_yolox(img, input_size=(640, 640)):
    """Letterbox to input_size, 114 padding, NCHW float32. Returns (tensor, ratio)."""
    h, w = input_size[0], input_size[1]
    padded = np.ones((h, w, 3), dtype=np.uint8) * 114
    r = min(h / img.shape[0], w / img.shape[1])
    new_w, new_h = int(img.shape[1] * r), int(img.shape[0] * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded[:new_h, :new_w] = resized
    tensor = padded.transpose(2, 0, 1)
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
    return tensor, r


def _yolox_decode(output, img_size, strides=(8, 16, 32)):
    """Decode YOLOX raw output to [x_center, y_center, w, h, box_score, class_probs...]."""
    grids, expanded_strides = [], []
    for stride in strides:
        hsize, wsize = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    out = output.copy()
    out[..., :2] = (out[..., :2] + grids) * expanded_strides
    out[..., 2:4] = np.exp(np.clip(out[..., 2:4], -50, 50)) * expanded_strides
    return out


def _nms_single(boxes, scores, nms_thr=0.45):
    """Single-class NMS."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def _multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.35):
    """Class-agnostic multiclass NMS."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    valid = cls_scores > score_thr
    if valid.sum() == 0:
        return None
    boxes_v = boxes[valid]
    scores_v = cls_scores[valid]
    cls_v = cls_inds[valid]
    keep = _nms_single(boxes_v, scores_v, nms_thr)
    if not keep:
        return None
    return np.column_stack([boxes_v[keep], scores_v[keep], cls_v[keep]])


def postprocess_yolox(output, img_size, ratio, thresh=0.3, target_classes=None):
    """Postprocess YOLOX output. Returns (boxes_xyxy, scores, class_ids).
    target_classes: e.g. (0, 2) for person, car; when set, only those classes are considered."""
    out = np.squeeze(output)
    if out.ndim == 2:
        out = out.reshape(1, -1, out.shape[-1])
    out = _yolox_decode(out, img_size)
    boxes = out[0, :, :4]  # x_center, y_center, w, h
    box_scores = 1 / (1 + np.exp(-np.clip(out[0, :, 4], -50, 50)))
    class_logits = out[0, :, 5:]
    class_probs = 1 / (1 + np.exp(-np.clip(class_logits, -50, 50)))
    scores = box_scores[:, None] * class_probs
    if target_classes is not None:
        mask = np.zeros_like(scores)
        mask[:, target_classes] = 1.0
        scores = scores * mask
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy /= ratio
    dets = _multiclass_nms(boxes_xyxy, scores, nms_thr=0.35, score_thr=0.35)
    if dets is None:
        return np.zeros((0, 4)), np.array([]), np.array([])
    boxes_out = dets[:, :4].astype(int)
    scores_out = dets[:, 4]
    cls_out = dets[:, 5].astype(int)
    keep = scores_out >= thresh
    return boxes_out[keep], scores_out[keep], cls_out[keep]


def _ensure_yolox_fp16(model_dir, model_name):
    """Convert YOLOX FP32 IR to FP16 and cache. Returns path to FP16 xml."""
    import openvino as ov
    model_dir = os.path.normpath(os.path.abspath(model_dir))
    fp16_dir = os.path.join(model_dir, model_name, "FP16")
    xml_fp16 = os.path.join(fp16_dir, f"{model_name}.xml")
    if os.path.exists(xml_fp16):
        return os.path.normpath(os.path.abspath(xml_fp16))
    xml_fp32 = os.path.join(model_dir, model_name, f"{model_name}.xml")
    if not os.path.exists(xml_fp32):
        raise FileNotFoundError(f"YOLOX model not found: {xml_fp32}")
    os.makedirs(fp16_dir, exist_ok=True)
    model = ov.Core().read_model(xml_fp32)
    ov.save_model(model, xml_fp16, compress_to_fp16=True)
    return xml_fp16


def load_yolox_model(model_dir, model_name, device="CPU", precision="FP32"):
    """Load YOLOX IR and compile. precision: FP32 or FP16."""
    import openvino as ov
    if precision == "FP32":
        xml_path = os.path.join(model_dir, model_name, f"{model_name}.xml")
    else:
        xml_path = _ensure_yolox_fp16(model_dir, model_name)
    xml_path = os.path.normpath(os.path.abspath(xml_path))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"YOLOX model not found: {xml_path}")
    core = ov.Core()
    model = core.read_model(xml_path)
    return core.compile_model(model, device), xml_path


def run_detection_yolox(compiled, image, target_classes=None, thresh=0.3):
    """Run YOLOX inference. Returns (boxes, labels, input_tensor). target_classes: None=all, or (0,2) for person,car (COCO80)."""
    input_shape = compiled.input(0).shape
    _, _, inp_h, inp_w = input_shape
    input_tensor, ratio = preprocess_yolox(image, (inp_h, inp_w))
    output = compiled([input_tensor])
    out_arr = list(output.values())[0]
    boxes, scores, labels = postprocess_yolox(out_arr, (inp_h, inp_w), ratio, thresh, target_classes)
    if target_classes is not None:
        keep = np.isin(labels, target_classes)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    orig_h, orig_w = image.shape[:2]
    boxes = boxes[:, [0, 1, 2, 3]].clip(0, [orig_w, orig_h, orig_w, orig_h]).astype(int)
    return boxes, labels, input_tensor


def preprocess_ssdlite(img, input_shape):
    """ssdlite_mobilenet_v2 expects NHWC (1,300,300,3), BGR."""
    s = tuple(input_shape)
    if len(s) == 4 and s[-1] == 3:
        _, h, w, _ = s
        resized = cv2.resize(img, (w, h))
        return np.expand_dims(resized, axis=0).astype(np.float32)
    return preprocess_image(img, input_shape)


def load_compiled_model(model_dir, precision, device="CPU"):
    """Load ssdlite_mobilenet_v2 IR and compile for given device."""
    import openvino as ov
    pattern = os.path.join(model_dir, "**", precision, "*.xml")
    xml_path = glob.glob(pattern, recursive=True)
    if not xml_path:
        raise FileNotFoundError(f"No {precision} model in {model_dir}")
    core = ov.Core()
    model = core.read_model(xml_path[0])
    return core.compile_model(model, device), xml_path[0]


def get_model_size_mb(xml_path):
    """Return model size in MB (bin file). Resolves path relative to xml dir."""
    xml_path = os.path.normpath(os.path.abspath(xml_path))
    bin_path = xml_path.replace(".xml", ".bin")
    if os.path.exists(bin_path):
        return os.path.getsize(bin_path) / (1024 * 1024)
    return 0.0


def plot_yolox_fp32_fp16_charts(model_dir, model_name, device, image_path, target_classes=(0, 2), thresh=0.3):
    """Plot FP32 vs FP16 latency and model size for YOLOX. Returns matplotlib figure."""
    import matplotlib.pyplot as plt
    img_path = image_path if os.path.isabs(image_path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
    img = cv2.imread(img_path)
    if img is None:
        img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    compiled_fp32, xml_fp32 = load_yolox_model(model_dir, model_name, device, "FP32")
    compiled_fp16, xml_fp16 = load_yolox_model(model_dir, model_name, device, "FP16")
    _, _, inp = run_detection_yolox(compiled_fp32, img, target_classes, thresh)
    lat_fp32 = benchmark_latency_ms(compiled_fp32, inp, num_iter=50)
    lat_fp16 = benchmark_latency_ms(compiled_fp16, inp, num_iter=50)
    size_fp32 = get_model_size_mb(xml_fp32)
    size_fp16 = get_model_size_mb(xml_fp16)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["FP32", "FP16"], [lat_fp32, lat_fp16], color=["#2ecc71", "#3498db"])
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Inference Latency")
    axes[1].bar(["FP32", "FP16"], [size_fp32, size_fp16], color=["#2ecc71", "#3498db"])
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title("Model Size")
    plt.suptitle(f"{model_name} on {device}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_ssdlite_fp32_fp16_charts(model_dir, device, image_path, target_classes=(1, 3), thresh=0.3):
    """Plot FP32 vs FP16 latency and model size for ssdlite_mobilenet_v2. Returns matplotlib figure."""
    import matplotlib.pyplot as plt
    img_path = image_path if os.path.isabs(image_path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
    img = cv2.imread(img_path)
    if img is None:
        img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    compiled_fp32, xml_fp32 = load_compiled_model(model_dir, "FP32", device)
    compiled_fp16, xml_fp16 = load_compiled_model(model_dir, "FP16", device)
    input_shape = compiled_fp32.input(0).shape
    _, _, inp = run_detection_ssdlite(compiled_fp32, img, input_shape, target_classes, thresh)
    lat_fp32 = benchmark_latency_ms(compiled_fp32, inp, num_iter=50)
    lat_fp16 = benchmark_latency_ms(compiled_fp16, inp, num_iter=50)
    size_fp32 = get_model_size_mb(xml_fp32)
    size_fp16 = get_model_size_mb(xml_fp16)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["FP32", "FP16"], [lat_fp32, lat_fp16], color=["#2ecc71", "#3498db"])
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Inference Latency")
    axes[1].bar(["FP32", "FP16"], [size_fp32, size_fp16], color=["#2ecc71", "#3498db"])
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title("Model Size")
    plt.suptitle(f"ssdlite_mobilenet_v2 on {device}", fontsize=12)
    plt.tight_layout()
    return fig


def run_detection_ssdlite(compiled, image, input_shape, target_classes=(1, 3), thresh=0.5):
    """Run inference and return (boxes, labels) for person(1)+car(3)."""
    orig_h, orig_w = image.shape[:2]
    input_tensor = preprocess_ssdlite(image, input_shape)
    output = compiled([input_tensor])
    out_arr = np.squeeze(list(output.values())[0])
    if out_arr.ndim == 3:
        out_arr = out_arr.reshape(-1, 7)
    boxes, labels = [], []
    for det in out_arr:
        batch_id, class_id, conf = float(det[0]), int(det[1]), float(det[2])
        if batch_id < 0:
            break
        if class_id not in target_classes or conf < thresh:
            continue
        x1, y1, x2, y2 = det[3], det[4], det[5], det[6]
        if max(x1, y1, x2, y2) <= 1.0:
            x1, y1, x2, y2 = x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        labels.append(class_id)
    return np.array(boxes) if boxes else np.zeros((0, 4)), labels, input_tensor


def find_model_xml(path):
    """Find .xml file in the given path."""
    files = glob.glob(os.path.join(path, "*.xml"))
    if not files:
        raise FileNotFoundError(f"No .xml model found in {path}")
    return files[0]


def preprocess_image(img, input_shape, model_name=""):
    """Resize and convert to NCHW format. BGR order (OpenCV default).
    Most OpenVINO IR models accept raw [0,255] float32."""
    _, _, h, w = input_shape
    resized = cv2.resize(img, (w, h))
    nchw = np.expand_dims(resized.transpose(2, 0, 1), axis=0).astype(np.float32)
    return nchw


def run_inference(compiled_model, input_tensor):
    """Run synchronous inference."""
    return compiled_model(input_tensor)


def postprocess(output, orig_h, orig_w, input_h, input_w, model_name, thresh=0.5):
    """Model-specific postprocessing. Supports:
    - DetectionOutput (1,1,N,7): [batch_id, class_id, conf, x_min, y_min, x_max, y_max], coords normalized 0-1
    - ATSS boxes (N,5): [x_min, y_min, x_max, y_max, conf], coords in input resolution
    - Deim-DFine-X: same as DetectionOutput if NMS in model, else boxes+scores."""
    # Find boxes and labels tensors (ATSS: boxes + labels, label 0 = background)
    # OpenVINO output keys can be ConstOutput objects, not strings
    def _key_name(k):
        return k.get_any_name() if hasattr(k, 'get_any_name') else str(k)
    arr = None
    labels_arr = None
    for k, v in output.items():
        name = _key_name(k)
        a = np.squeeze(np.array(v))
        if a.ndim == 1:
            a = np.expand_dims(a, 0)
        if name.lower() == "labels" and a.size > 0:
            labels_arr = a.flatten()
        elif arr is None and a.ndim >= 2 and a.shape[-1] in (5, 6, 7):
            arr = a.reshape(-1, a.shape[-1])
    if arr is None:
        arr = np.squeeze(list(output.values())[0])
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        arr = arr.reshape(-1, arr.shape[-1]) if arr.size > 0 else np.zeros((0, 7))
    boxes = []
    # DetectionOutput: [batch_id, class_id, conf, x_min, y_min, x_max, y_max], normalized 0-1
    if arr.shape[-1] == 7:
        for det in arr:
            batch_id, conf = float(det[0]), float(det[2])
            if batch_id < 0:
                break
            if conf <= thresh:
                continue
            x1, y1, x2, y2 = det[3], det[4], det[5], det[6]
            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    # ATSS/Deim boxes (N,5): [x_min, y_min, x_max, y_max, conf] or [cx, cy, w, h, conf] for DETR-based.
    elif arr.shape[-1] in (5, 6):
        use_label_filter = labels_arr is not None and np.any(labels_arr > 0)
        # Deim-DFine-X (DETR-based) outputs [cx, cy, w, h, conf]; convert to corner format
        use_center_format = "deim" in model_name.lower() or "dfine" in model_name.lower()
        for i, det in enumerate(arr):
            if use_label_filter and i < len(labels_arr) and int(labels_arr[i]) <= 0:
                continue
            conf_val = float(det[4])
            if 0 < conf_val <= 1 and conf_val < thresh:
                continue
            if use_label_filter and conf_val <= 0:
                continue
            cx, cy, w_or_x2, h_or_y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
            if use_center_format:
                # [cx, cy, w, h] -> [x1, y1, x2, y2]
                x1 = cx - w_or_x2 / 2
                y1 = cy - h_or_y2 / 2
                x2 = cx + w_or_x2 / 2
                y2 = cy + h_or_y2 / 2
            else:
                x1, y1, x2, y2 = cx, cy, w_or_x2, h_or_y2
            # Scale from input resolution (or normalized 0-1) to original image
            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h
            else:
                scale_x, scale_y = orig_w / input_w, orig_h / input_h
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
            x1 = max(0, min(int(x1), orig_w))
            y1 = max(0, min(int(y1), orig_h))
            x2 = max(0, min(int(x2), orig_w))
            y2 = max(0, min(int(y2), orig_h))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.zeros((0, 4))


def draw_boxes(img, boxes, color=(0, 200, 0), thickness=2):
    """Draw bounding boxes on image."""
    out = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def check_device_latency(device, latency_ms, model_name="yolox_s"):
    """Warn if GPU selected but latency suggests CPU (GPU typically 5-20ms, CPU 60-250ms)."""
    if device.upper() != "GPU":
        return
    # GPU typically 5-25ms for yolox_s; CPU 60-250ms
    if latency_ms > 50:
        print(f"[!] GPU selected but latency {latency_ms:.1f}ms suggests CPU. Change device and wait for auto-reload (or re-run device cell).")


def benchmark_latency_ms(compiled_model, input_tensor, num_iter=100):
    """Single-image inference; return average latency in ms."""
    for _ in range(10):
        compiled_model(input_tensor)
    times = []
    for _ in range(num_iter):
        start = time.perf_counter()
        compiled_model(input_tensor)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)
