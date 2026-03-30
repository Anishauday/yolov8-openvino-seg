"""
Gradio apps for Geti Lab 4: car detection + color classification, and optional
``benchmark_app`` throughput/latency UI.

Uses ``bm_utils.py`` only. Run:
  python app_gradio.py
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import openvino as ov

import bm_utils as u

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODELS = os.path.join(_BASE, "models")
_DEFAULT_MEDIA = os.path.join(_BASE, "media")

_cache: dict = {}


def _get_compiled(
    precision: str,
    det_device: str,
    cls_device: str,
    models_root: str,
):
    key = (precision, det_device, cls_device, models_root)
    if key in _cache:
        return _cache[key]
    core = ov.Core()
    det_xml, cls_xml = u.model_xml_paths(models_root, precision)
    if not det_xml.is_file() or not cls_xml.is_file():
        raise FileNotFoundError(f"Need IRs at {det_xml} and {cls_xml}")
    det = core.read_model(str(det_xml))
    cls_ = core.read_model(str(cls_xml))
    compiled_det = core.compile_model(det, det_device)
    compiled_cls = core.compile_model(cls_, cls_device)
    det_label = u.load_labels(det_xml.parent)
    det_lbl = det_label[0] if det_label else "CAR"
    cls_labels = u.load_labels(cls_xml.parent)
    _cache[key] = (core, compiled_det, compiled_cls, det_lbl, cls_labels)
    return _cache[key]


def run_inference(
    image: Optional[np.ndarray],
    precision: str,
    det_device: str,
    cls_device: str,
    confidence_threshold: float,
    models_root: str,
) -> Tuple[Optional[np.ndarray], str]:
    """Detection + classification on the image; returns RGB result and a short status line."""
    if image is None:
        return None, "Upload or select an image."
    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    try:
        _, compiled_det, compiled_cls, det_lbl, cls_labels = _get_compiled(
            precision, det_device, cls_device, models_root
        )
        tensor, oh, ow = u.prepare_detection_tensor(img, compiled_det)
        detection_output = compiled_det([tensor])
        ih, iw = u.detection_input_hw(compiled_det)
        boxes, _ = u.postprocess_detection(
            detection_output, oh, ow, iw, ih, thresh=confidence_threshold
        )
        color_names, _ = u.classify_crops_with_labels(
            compiled_cls, img, boxes, cls_labels, detection_label=det_lbl
        )
        vis = u.draw_labeled_boxes(img, boxes, color_names)
        detection_ms, classification_ms, num_boxes = u.run_pipeline_latency_ms(
            compiled_det, compiled_cls, img, thresh=confidence_threshold
        )
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        msg = (
            f"Pipeline: Detection {detection_ms:.1f} ms + Classification {classification_ms:.1f} ms | "
            f"boxes: {num_boxes} | confidence threshold: {confidence_threshold:.3f}"
        )
        return rgb, msg
    except Exception as e:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"Error: {e}"


def create_demo(models_root: Optional[str] = None, media_dir: Optional[str] = None):
    models_root = models_root or _DEFAULT_MODELS
    media_dir = media_dir or _DEFAULT_MEDIA
    samples = u.list_media_images(media_dir)
    core = ov.Core()
    devices = list(core.available_devices)
    default_dev = "CPU" if "CPU" in devices else devices[0]

    with gr.Blocks(title="Lab 4 — Car detection + color (OpenVINO)") as demo:
        gr.Markdown("# Car detection + color classification\n")
        with gr.Row():
            precision = gr.Dropdown(
                ["FP32", "FP16", "INT8"],
                value="INT8",
                label="Precision",
            )
            det_dev = gr.Dropdown(devices, value=default_dev, label="Detection device")
            cls_dev = gr.Dropdown(devices, value="CPU", label="Classification device")
        confidence_threshold = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=u.DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
            step=0.025,
            label="Detection confidence threshold (min score to keep a box)",
        )
        img_in = gr.Image(type="numpy", label="Image")
        run_btn = gr.Button("Run Inference", variant="primary")
        out_img = gr.Image(label="Result")
        out_txt = gr.Textbox(label="Info", lines=3)
        run_btn.click(
            lambda im, p, d, c, t: run_inference(im, p, d, c, t, models_root),
            [img_in, precision, det_dev, cls_dev, confidence_threshold],
            [out_img, out_txt],
        )
        if samples:
            gr.Examples(
                label="Sample images",
                examples=[[os.path.join(media_dir, s)] for s in samples[:3]],
                inputs=[img_in],
            )

    return demo


def build_gr_blocks(available_devices, goal: str):
    """Gradio UI: precision + device, run ``benchmark_app``, plot (Geti detection IR)."""
    model_options = ["car_detection"]
    precision_options = ["FP32", "FP16", "INT8"]
    if goal == "Throughput":
        device_options = list(available_devices) + ["AUTO throughput"]
    else:
        device_options = list(available_devices) + ["AUTO"]

    with gr.Blocks(fill_width=True) as demo:
        gr.Markdown(f"# Benchmark Model {goal} by Device (detection IR)")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                model_name = gr.Dropdown(
                    model_options, label="Choose a model", value=model_options[0]
                )
                precision = gr.Dropdown(
                    precision_options, label="Choose a precision", value=precision_options[0]
                )
                device = gr.Dropdown(device_options, label="Choose a device", value=device_options[0])
                hint = gr.Radio(
                    choices=[goal.lower()],
                    label="Goal",
                    value=goal.lower(),
                    visible=False,
                )
                run = gr.Button("Run Benchmark", variant="primary")
                clear = gr.Button("Clear Plot Data")
            with gr.Column(scale=2, min_width=300):
                plot = gr.Plot()
        with gr.Row():
            cmd = gr.Textbox(label="Command used for this run:")
            run.click(u.run_benchmark, inputs=[model_name, precision, device, hint], outputs=[plot, cmd])
            clear.click(u.clear_plot_results, inputs=[model_name, hint], outputs=[plot])
    return demo


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=_DEFAULT_MODELS, help="Models root")
    ap.add_argument("--media", default=_DEFAULT_MEDIA, help="Media folder")
    args = ap.parse_args()
    create_demo(models_root=args.models, media_dir=args.media).launch()
