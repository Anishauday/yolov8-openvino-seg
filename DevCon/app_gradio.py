"""
Gradio app: SSD-MobileNetV2 and YOLOX detection with image selection, device, and precision.
Run from folder 1: python app_gradio.py models_test
"""
import os
import glob
import cv2
import numpy as np
import gradio as gr

# Use script directory as base
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_compiled_cache = {}
_compiled_yolox_cache = {}
_model_dir = None


def get_media_images(media_dir="media"):
    """List image filenames in media folder."""
    media_path = os.path.join(_BASE_DIR, media_dir)
    if not os.path.exists(media_path):
        os.makedirs(media_path, exist_ok=True)
        return []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(media_path, ext)))
    return sorted([os.path.basename(p) for p in paths])


def get_compiled(device, precision):
    """Load and cache compiled model."""
    global _model_dir
    key = (device, precision)
    if key in _compiled_cache:
        return _compiled_cache[key]
    from utils import load_compiled_model
    if _model_dir is None:
        raise ValueError("Model not downloaded. Run the notebook download cell first.")
    compiled, xml_path = load_compiled_model(_model_dir, precision, device)
    _compiled_cache[key] = (compiled, xml_path)
    return compiled, xml_path


def run_detection(image_name, device, precision):
    """Run detection and return (image, latency_str)."""
    if not image_name:
        return None, "Select an image"
    image_path = os.path.join(_BASE_DIR, "media", image_name)
    if not os.path.exists(image_path):
        return None, f"Image not found: {image_path}"
    image = cv2.imread(image_path)
    if image is None:
        return None, "Failed to load image"
    try:
        compiled, xml_path = get_compiled(device, precision)
        input_shape = compiled.input(0).shape
        from utils import run_detection_ssdlite, draw_boxes, benchmark_latency_ms
        boxes, labels, input_tensor = run_detection_ssdlite(
            compiled, image, input_shape, target_classes=(1, 3), thresh=0.3
        )
        result_img = draw_boxes(image, boxes)
        latency_ms = benchmark_latency_ms(compiled, input_tensor, num_iter=50)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        return result_rgb, f"{latency_ms:.2f} ms | {len(boxes)} detections"
    except Exception as e:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), f"Error: {e}"


def create_demo(model_dir="models_test"):
    """Create Gradio demo for SSD-MobileNetV2. Call with model_dir after models are downloaded."""
    global _model_dir
    _model_dir = os.path.join(_BASE_DIR, model_dir) if not os.path.isabs(model_dir) else model_dir
    images = get_media_images()
    from utils import get_available_devices
    devices = get_available_devices()
    default_device = "CPU" if "CPU" in devices else (devices[0] if devices else "CPU")
    with gr.Blocks(title="SSD-MobileNetV2 Detection") as demo:
        gr.Markdown("# SSD-MobileNetV2 Detection (OpenVINO)")
        with gr.Row():
            image_dropdown = gr.Dropdown(
                choices=images,
                value=images[0] if images else None,
                label="Image",
                allow_custom_value=False,
            )
            device_dropdown = gr.Dropdown(
                choices=devices,
                value=default_device,
                label="Device",
            )
            precision_dropdown = gr.Dropdown(
                choices=["FP32", "FP16"],
                value="FP32",
                label="Precision",
            )
        run_btn = gr.Button("Run Detection", variant="primary")
        with gr.Row():
            output_image = gr.Image(label="Detection result")
            output_info = gr.Textbox(label="Latency & detections", lines=2)
        run_btn.click(
            fn=run_detection,
            inputs=[image_dropdown, device_dropdown, precision_dropdown],
            outputs=[output_image, output_info],
        )
    return demo


def get_compiled_yolox(model_name, device, precision):
    """Load and cache YOLOX compiled model."""
    global _model_dir
    key = (model_name, device, precision)
    if key in _compiled_yolox_cache:
        return _compiled_yolox_cache[key]
    from utils import load_yolox_model
    if _model_dir is None:
        raise ValueError("Model dir not set. Run create_demo_yolox(model_dir) first.")
    compiled, xml_path = load_yolox_model(_model_dir, model_name, device, precision)
    _compiled_yolox_cache[key] = (compiled, xml_path)
    return compiled, xml_path


def run_detection_yolox(image_name, model_name, device, precision):
    """Run YOLOX detection and return (image, latency_str)."""
    if not image_name:
        return None, "Select an image"
    image_path = os.path.join(_BASE_DIR, "media", image_name)
    if not os.path.exists(image_path):
        return None, f"Image not found: {image_path}"
    image = cv2.imread(image_path)
    if image is None:
        return None, "Failed to load image"
    try:
        compiled, _ = get_compiled_yolox(model_name, device, precision)
        from utils import run_detection_yolox as _run_yolox, draw_boxes, benchmark_latency_ms
        target_classes = (0, 2)  # COCO80: person, car
        boxes, labels, input_tensor = _run_yolox(compiled, image, target_classes=target_classes, thresh=0.3)
        result_img = draw_boxes(image, boxes)
        latency_ms = benchmark_latency_ms(compiled, input_tensor, num_iter=50)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        return result_rgb, f"{latency_ms:.2f} ms | {len(boxes)} detections"
    except Exception as e:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), f"Error: {e}"


def create_demo_yolox(model_dir="models_test"):
    """Create Gradio demo for YOLOX. Call with model_dir after models are downloaded."""
    global _model_dir
    _model_dir = os.path.join(_BASE_DIR, model_dir) if not os.path.isabs(model_dir) else model_dir
    images = get_media_images()
    from utils import get_available_devices
    devices = get_available_devices()
    default_device = "CPU" if "CPU" in devices else (devices[0] if devices else "CPU")
    with gr.Blocks(title="YOLOX Detection") as demo:
        gr.Markdown("# YOLOX Detection (OpenVINO) — person & car")
        with gr.Row():
            image_dropdown = gr.Dropdown(
                choices=images,
                value=images[0] if images else None,
                label="Image",
                allow_custom_value=False,
            )
            model_dropdown = gr.Dropdown(
                choices=["yolox_s", "yolox_l"],
                value="yolox_s",
                label="Model",
            )
            device_dropdown = gr.Dropdown(
                choices=devices,
                value=default_device,
                label="Device",
            )
            precision_dropdown = gr.Dropdown(
                choices=["FP32", "FP16"],
                value="FP32",
                label="Precision",
            )
        run_btn = gr.Button("Run Detection", variant="primary")
        with gr.Row():
            output_image = gr.Image(label="Detection result")
            output_info = gr.Textbox(label="Latency & detections", lines=2)
        run_btn.click(
            fn=run_detection_yolox,
            inputs=[image_dropdown, model_dropdown, device_dropdown, precision_dropdown],
            outputs=[output_image, output_info],
        )
    return demo


if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models_test"
    if not os.path.exists(model_dir):
        print("Models not found. Run: omz_downloader + omz_converter first.")
        print("Or run from notebook after download.")
    demo = create_demo(model_dir)
    demo.launch()
