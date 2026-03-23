"""
Car detection inference utils — Intel exported model (deployment folder).
Pure OpenVINO pipeline. Optional geti_sdk for comparison.
"""
import os
import glob
import json
import numpy as np
import cv2

USE_SDK = False
try:
    from geti_sdk.deployment import Deployment
    USE_SDK = True
except ImportError:
    pass


def get_sample_images(base_dir=".", exts=None):
    """Return list of (label, path) for Gradio. Scans base_dir (e.g. media/)."""
    if exts is None:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    for d in (base_dir, os.path.join(base_dir, "media"), os.path.join(base_dir, "example_code")):
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
    """Return list of available OpenVINO devices (e.g. ['CPU', 'GPU']).
    exclude: list of device names to filter out (e.g. ['NPU'])."""
    import openvino as ov
    devs = list(ov.Core().available_devices)
    if exclude:
        devs = [d for d in devs if d.upper() not in {e.upper() for e in exclude}]
    return devs


def load_model(model_dir, device="CPU", num_streams=None):
    """Load OpenVINO IR. Returns (compiled, config).
    num_streams: set for throughput (e.g. 2, 4); uses PERFORMANCE_HINT=THROUGHPUT."""
    import openvino as ov
    xml_path = os.path.join(model_dir, "model.xml")
    xml_path = os.path.normpath(os.path.abspath(xml_path))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Model not found: {xml_path}")
    config = load_config(model_dir)
    core = ov.Core()
    model = core.read_model(xml_path)
    if num_streams is not None:
        config_ov = {"PERFORMANCE_HINT": "THROUGHPUT", "NUM_STREAMS": str(num_streams)}
        compiled = core.compile_model(model, device, config_ov)
    else:
        compiled = core.compile_model(model, device)
    return compiled, config


def get_model_name(model_dir):
    """Return model name from model.json if present, else folder name.
    Checks model_dir and parent (Detection) folder."""
    for d in (model_dir, os.path.dirname(model_dir)):
        model_json = os.path.join(d, "model.json")
        if os.path.exists(model_json):
            with open(model_json) as f:
                data = json.load(f)
                return data.get("name", os.path.basename(model_dir))
    return os.path.basename(model_dir)


def get_inference_tensor(compiled, image):
    """Return preprocessed NCHW tensor for benchmarking."""
    inp = compiled.input(0)
    sh = inp.get_partial_shape()
    d2, d3 = sh[2], sh[3]
    input_h = int(d2.get_length()) if d2.is_static else 800
    input_w = int(d3.get_length()) if d3.is_static else 992
    return preprocess_sdk_style(image, input_h, input_w)


def benchmark_latency_ms(compiled, input_tensor, num_iter=50):
    """Average inference latency in ms. Warmup + measured iterations."""
    for _ in range(10):
        compiled([input_tensor])
    import time
    t0 = time.perf_counter()
    for _ in range(num_iter):
        compiled([input_tensor])
    return (time.perf_counter() - t0) / num_iter * 1000


def benchmark_throughput_streams(model_dir, image, device="CPU", stream_counts=(1, 2, 4), num_frames=200):
    """
    Benchmark throughput (inferences/sec) for different stream counts.
    Returns list of (num_streams, throughput_fps, latency_ms).
    """
    import time
    tensor = None
    results = []
    for n in stream_counts:
        compiled, _ = load_model(model_dir, device, num_streams=n)
        if tensor is None:
            tensor = get_inference_tensor(compiled, image)
        for _ in range(10):
            compiled([tensor])
        t0 = time.perf_counter()
        for _ in range(num_frames):
            compiled([tensor])
        elapsed = time.perf_counter() - t0
        throughput = num_frames / elapsed
        latency_ms = (elapsed / num_frames) * 1000
        results.append((n, throughput, latency_ms))
    return results


def preprocess_sdk_style(image, input_h, input_w):
    """
    Replicate model_api/ImageModel preprocess with resize_type='standard'.
    - Direct resize to (input_w, input_h)
    - Normalize /255 (mean=0, scale=255)
    - Input: RGB (match geti_sdk which uses RGB)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h))
    normalized = resized.astype(np.float32) / 255.0
    nchw = np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)
    return nchw


def postprocess_sdk_style(output, orig_h, orig_w, input_w, input_h, thresh=0.225):
    """
    Replicate model_api BoxesLabelsParser + _resize_detections (resize_type='standard').
    Model outputs: boxes [N,5] = (xmin, ymin, xmax, ymax, score) in input pixel space.
    Normalize: x /= input_w, y /= input_h.
    Scale to orig: orig_x = norm_x * orig_w, orig_y = norm_y * orig_h.
    """
    out_by_name = {}
    for k, v in output.items():
        name = k.get_any_name() if hasattr(k, "get_any_name") else str(k)
        out_by_name[name.lower()] = np.array(v)
    boxes_arr = out_by_name.get("boxes")
    if boxes_arr is None:
        boxes_arr = np.squeeze(list(output.values())[0])
    labels_arr = out_by_name.get("labels")
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
        if labels_arr is not None and i < len(labels_arr):
            try:
                lab = int(np.asarray(labels_arr[i]).item())
                if lab < 0:
                    continue
            except (ValueError, TypeError):
                pass
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


def run_detection_ov(compiled, image, thresh=0.225, label="", return_latency=False):
    """
    Pure OpenVINO pipeline - no geti_sdk. Replicates SDK preprocessing/postprocessing
    from model_api (standard resize, BoxesLabelsParser). For workshop use.
    If return_latency=True, returns (boxes, result, lat_ms).
    """
    import time
    inp = compiled.input(0)
    sh = inp.get_partial_shape()
    d2, d3 = sh[2], sh[3]
    input_h = int(d2.get_length()) if d2.is_static else 800
    input_w = int(d3.get_length()) if d3.is_static else 992
    orig_h, orig_w = image.shape[:2]
    tensor = preprocess_sdk_style(image, input_h, input_w)
    t0 = time.perf_counter()
    out = compiled([tensor])
    lat_ms = (time.perf_counter() - t0) * 1000
    boxes, scores = postprocess_sdk_style(out, orig_h, orig_w, input_w, input_h, thresh)
    result = draw_boxes(image, boxes, scores=scores, label=label)
    if return_latency:
        return boxes, result, lat_ms
    return boxes, result


def run_detection_sdk(deployment_path, image, label=""):
    """Run inference via geti_sdk (correct preprocessing). Returns (boxes, result_image)."""
    deployment = Deployment.from_folder(deployment_path)
    deployment.load_inference_models(device="CPU")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction = deployment.infer(image_rgb)
    boxes, scores = [], []
    for ann in prediction.annotations:
        rect = ann.shape
        x1 = rect.x
        y1 = rect.y
        x2 = rect.x + rect.width
        y2 = rect.y + rect.height
        boxes.append([x1, y1, x2, y2])
        scores.append(ann.labels[0].probability if ann.labels else 0)
    boxes = np.array(boxes) if boxes else np.zeros((0, 4))
    scores = np.array(scores) if scores else np.zeros(0)
    result = draw_boxes(image, boxes, scores=scores, label=label or "car")
    return boxes, result


def run_detection(compiled, image, thresh=0.225, label="", use_sdk=False, deployment_path=None):
    """
    Run inference. Returns (boxes, result_image).
    Default: pure OpenVINO pipeline (run_detection_ov) - no SDK needed.
    Set use_sdk=True and deployment_path to use geti_sdk instead (for comparison).
    """
    if use_sdk and USE_SDK and deployment_path:
        if not os.path.exists(os.path.join(deployment_path, "project.json")):
            deployment_path = os.path.join(deployment_path, "deployment")
        return run_detection_sdk(deployment_path, image, label=label)
    return run_detection_ov(compiled, image, thresh=thresh, label=label)


def convert_video_to_avi(input_path, output_path=None):
    """
    Convert video to AVI with MJPEG (plays in notebook/browser).
    Returns output_path. Use when mp4 doesn't display.
    """
    out = output_path or str(input_path).rsplit(".", 1)[0] + ".avi"
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out, fourcc, fps, (w, h))
    while True:
        ret, f = cap.read()
        if not ret:
            break
        writer.write(f)
    cap.release()
    writer.release()
    return out


def convert_video_to_gif(input_path, output_path=None, max_frames=60, scale=0.5, fps=10):
    """
    Convert video to GIF (plays in all browsers). Use when AVI/MP4 don't display.
    max_frames: limit frames to keep GIF small. scale: resize factor (0.5 = half).
    """
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
                w = int(f.shape[1] * scale)
                h = int(f.shape[0] * scale)
                f = cv2.resize(f, (w, h))
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        n += 1
    cap.release()
    if frames:
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000//fps, loop=0)
    return out


def run_detection_video(compiled, video_path, output_path=None, thresh=0.225, label="", device="CPU"):
    """
    Run detection on video. Works with any .mp4 (or cv2-supported format).
    Overlays device name and inference latency (ms) on each frame.
    Use .avi extension for notebook display (MJPEG); .mp4 for H.264.
    Returns (output_path, frame_count, fps). Writes output video if output_path given.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path:
        # AVI + MJPEG = best notebook display; MP4 = avc1/mp4v
        if str(output_path).lower().endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, result, lat_ms = run_detection_ov(compiled, frame, thresh=thresh, label=label, return_latency=True)
        txt = f"{device} {lat_ms:.0f} ms"
        cv2.putText(result, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if writer:
            writer.write(result)
        frame_count += 1
    cap.release()
    if writer:
        writer.release()
    return output_path, frame_count, fps


def draw_boxes(image, boxes, color=(173, 216, 230), thickness=2, scores=None, label=""):
    """Draw bounding boxes on image. If scores provided, draw label with confidence %."""
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if scores is not None and i < len(scores):
            text = f"{label} {int(scores[i] * 100)}%" if label else f"{int(scores[i] * 100)}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img
