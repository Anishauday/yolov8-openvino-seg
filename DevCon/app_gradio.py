"""
Gradio app: Car detection — sample, upload, or webcam.
Run: python app_gradio.py
"""
import os
import cv2
import numpy as np
import gradio as gr

_BASE = os.path.dirname(os.path.abspath(__file__))
# media, models, output are inside folder 1
_ROOT = _BASE
_compiled = None
_labels = "car"


def _get_compiled():
    global _compiled, _labels
    if _compiled is not None:
        return _compiled, _labels
    from ov_utils import load_model, load_config
    model_dir = os.path.join(_ROOT, "models", "INT8")
    _compiled, config = load_model(model_dir, device="CPU")
    _labels = config.get("model_parameters", {}).get("labels", "car")
    return _compiled, _labels


def run_detection_ui(image):
    """Run detection on image from any source."""
    if image is None:
        return None, "Provide an image (sample, upload, or webcam)"
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    try:
        compiled, label = _get_compiled()
        from ov_utils import run_detection_ov
        boxes, result = run_detection_ov(compiled, img, thresh=0.225, label=label)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result_rgb, f"{len(boxes)} {label}(s) detected"
    except Exception as e:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"Error: {e}"


def create_demo():
    from ov_utils import get_sample_images
    samples = get_sample_images(os.path.join(_ROOT, "media"))
    default_path = samples[0][1] if samples and os.path.exists(samples[0][1]) else None
    choice_list = [(n, p) for n, p in samples] if samples else []

    def load_sample(sel):
        if not sel:
            return None
        path = sel
        for n, p in samples:
            if n == sel:
                path = p
                break
            if p == sel:
                path = p
                break
        if path and os.path.exists(str(path)):
            img = cv2.imread(str(path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    with gr.Blocks(title="Car Detection") as demo:
        gr.Markdown("# Car Detection (OpenVINO)\nSelect sample, upload, or use webcam.")
        with gr.Row():
            sample_dd = gr.Dropdown(
                choices=choice_list,
                value=default_path,
                label="Sample",
                allow_custom_value=False,
            )
            img_input = gr.Image(
                label="Image (or upload / webcam)",
                sources=["upload", "webcam"],
                type="numpy",
            )
        run_btn = gr.Button("Run Detection", variant="primary")
        with gr.Row():
            out_img = gr.Image(label="Result")
            out_txt = gr.Textbox(label="Info", lines=2)

        sample_dd.change(load_sample, inputs=[sample_dd], outputs=[img_input])
        demo.load(load_sample, inputs=[sample_dd], outputs=[img_input])
        run_btn.click(run_detection_ui, inputs=[img_input], outputs=[out_img, out_txt])
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
