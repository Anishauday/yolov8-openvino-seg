"""
Gradio app: Car detection + color classification.
Run: python app_gradio.py
"""
import os
import cv2
import numpy as np
import gradio as gr

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _BASE

_model_cache = {}
_cached_key = None

# Caption + Examples above the main image: img_input must be created before gr.Examples(inputs=[img_input]).
# Order uses :nth-child on the Column's direct children (Gradio wraps each block); do not rely on inner .elem_classes.
DEMO_CSS = """
.sample-above {
  display: flex !important;
  flex-direction: column !important;
  gap: 0.75rem;
}
/* DOM order: (1) Image, (2) caption, (3) Examples column — visual: caption, samples, image */
.sample-above--with-samples > *:nth-child(1) { order: 3 !important; }
.sample-above--with-samples > *:nth-child(2) { order: 1 !important; }
.sample-above--with-samples > *:nth-child(3) { order: 2 !important; }
/* No samples: (1) Image, (2) hint — visual: hint, image */
.sample-above--no-samples > *:nth-child(1) { order: 2 !important; }
.sample-above--no-samples > *:nth-child(2) { order: 1 !important; }
"""


def _get_models(model_precision, device):
    """Load models for given precision and device. Cached by (model_precision, device)."""
    global _model_cache, _cached_key
    key = (model_precision, device)
    if key == _cached_key and key in _model_cache:
        return _model_cache[key]
    from ov_utils import load_model, get_available_model_precisions
    models_dir = os.path.join(_ROOT, "models")
    if model_precision not in get_available_model_precisions(models_dir):
        raise FileNotFoundError(f"Model {model_precision} not found in {models_dir}")
    det_dir = os.path.join(models_dir, model_precision, "Detection")
    cls_dir = os.path.join(models_dir, model_precision, "Classification")
    det_compiled, det_cfg = load_model(det_dir, device=device)
    cls_compiled, cls_cfg = load_model(cls_dir, device=device)
    det_label = det_cfg.get("model_parameters", {}).get("labels", "CAR")
    lbl = cls_cfg.get("model_parameters", {}).get("labels", "")
    cls_labels = [s.strip() for s in lbl.split()] if lbl else []
    _model_cache[key] = (det_compiled, cls_compiled, det_label, cls_labels)
    _cached_key = key
    return _model_cache[key]


def run_pipeline_ui(image, model_precision, device):
    """Run detection + classification on image."""
    if image is None:
        return None, "Provide an image (sample, upload, or webcam)"
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    from ov_utils import get_notebook_config, is_transient_npu_device_error, run_detect_and_classify

    thresh = get_notebook_config(_ROOT)["THRESH"]

    def _infer(dev: str):
        det_c, cls_c, det_lbl, cls_lbls = _get_models(model_precision, dev)
        return run_detect_and_classify(
            det_c, cls_c, img, det_label=det_lbl, cls_labels=cls_lbls, thresh=thresh
        )

    try:
        boxes, scores, color_labels, result = _infer(device)
    except Exception as e:
        # NPU can surface ZE_RESULT_ERROR_DEVICE_LOST even when a frame was produced; retry on CPU without showing the noisy error.
        if device != "CPU" and is_transient_npu_device_error(e):
            try:
                boxes, scores, color_labels, result = _infer("CPU")
            except Exception as e2:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"Error: {e2}"
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"Error: {e}"

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    n = len(boxes)
    colors_str = ", ".join(c.split(" - ")[-1] for c in color_labels) if color_labels else ""
    return result_rgb, f"{n} car(s): {colors_str}"


def create_demo():
    from ov_utils import get_available_devices, get_available_model_precisions, get_sample_images_by_prefix

    models_dir = os.path.join(_ROOT, "models")
    precisions = get_available_model_precisions(models_dir)
    default_model = precisions[0] if precisions else "INT8"
    devices = get_available_devices()
    default_device = "CPU" if "CPU" in devices else (devices[0] if devices else "CPU")

    media_dir = os.path.join(_ROOT, "media")
    sample_paths = get_sample_images_by_prefix(media_dir, prefix="sample_image_", max_count=3)
    examples = [[p] for p in sample_paths[:3]]

    def load_first():
        if sample_paths:
            img = cv2.imread(sample_paths[0])
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    with gr.Blocks(title="Car Detection + Color Classification") as demo:
        gr.Markdown("# Car Detection + Color Classification (OpenVINO)\nDetect cars, classify color. Labels: CAR - [Color]")
        with gr.Row():
            precision_dd = gr.Dropdown(
                choices=precisions,
                value=default_model,
                label="Precision",
                allow_custom_value=False,
            )
            device_dd = gr.Dropdown(
                choices=devices,
                value=default_device,
                label="Device",
                allow_custom_value=False,
            )
        # img_input must exist before gr.Examples(inputs=[img_input]). See DEMO_CSS .sample-above--* for visual order.
        _sample_col_cls = ["sample-above", "sample-above--with-samples" if examples else "sample-above--no-samples"]
        with gr.Column(elem_classes=_sample_col_cls):
            img_input = gr.Image(label="Image (or upload / webcam)", sources=["upload", "webcam"], type="numpy")
            if examples:
                gr.Markdown("**Sample images** (click a thumbnail to load)")
                with gr.Column():
                    gr.Examples(
                        examples=examples,
                        inputs=[img_input],
                        label="Samples",
                        examples_per_page=3,
                    )
            else:
                gr.Markdown(
                    "*Add up to 3 files under `media/` named `sample_image_*.jpg` (e.g. `sample_image_1.jpg`) to show quick picks.*"
                )
        run_btn = gr.Button("Run Detection + Classification", variant="primary")
        with gr.Row():
            out_img = gr.Image(label="Result")
            out_txt = gr.Textbox(label="Info", lines=2)
        demo.load(load_first, outputs=[img_input])
        run_btn.click(
            run_pipeline_ui,
            inputs=[img_input, precision_dd, device_dd],
            outputs=[out_img, out_txt],
        )
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(css=DEMO_CSS)
