"""
Lab 1 inference utilities: preprocessing, postprocessing, model loading.
Keeps the notebook minimal and logic easy to debug.
"""
import os
import glob
import time
import numpy as np
import cv2


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


def postprocess(output, orig_h, orig_w, input_h, input_w, model_name, thresh=0.3):
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
    # ATSS/Deim boxes (N,5): [x_min, y_min, x_max, y_max, conf]. Filter by labels>0 when present.
    elif arr.shape[-1] in (5, 6):
        use_label_filter = labels_arr is not None and np.any(labels_arr > 0)
        for i, det in enumerate(arr):
            if use_label_filter and i < len(labels_arr) and int(labels_arr[i]) <= 0:
                continue
            conf_val = float(det[4])
            if 0 < conf_val <= 1 and conf_val < thresh:
                continue
            if use_label_filter and conf_val <= 0:
                continue
            x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
            # Scale from input resolution to original image
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
