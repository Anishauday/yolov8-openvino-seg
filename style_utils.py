import collections
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import gradio as gr

# Global variables (original pattern)
core = ov.Core()
compiled_model_dict = None
compiled_model = None
vid_in = None
stop_processing = False
USE_WEBCAM_state = True
thread_active = False
H, W = 224, 224
output_layer = 0
no_filter = False
_processing_times = collections.deque(maxlen=200)

style_options = ["mosaic", "rain-princess", "candy", "udnie", "pointilism", "No filter"]


def load_models(model_list, device_list):
    """Load and compile OpenVINO models for all style/device combinations."""
    global core
    result = {}

    for model_name in model_list:
        if model_name == "No filter":
            continue

        for device in device_list:
            ir_path = Path(f"model/{model_name}-9.xml")
            if not ir_path.exists():
                continue

            key_name = f"{model_name}_{device}"
            model = core.read_model(model=ir_path)
            compiled = core.compile_model(model=model, device_name=device)
            result[key_name] = compiled

    return result


def preprocess_rgb(frame_bgr, H, W):
    """Preprocess BGR frame for model. Converts to RGB, resizes, transposes."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = np.array(frame_rgb).astype("float32")
    image = cv2.resize(src=image, dsize=(W, H), interpolation=cv2.INTER_AREA)
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    return image


def convert_result_to_image(frame_shape, stylized_image) -> np.ndarray:
    """Postprocess stylized image. Returns BGR for cv2.imshow."""
    h, w = frame_shape[:2]
    stylized_image = stylized_image.squeeze().transpose(1, 2, 0)
    stylized_image = cv2.resize(src=stylized_image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)
    return stylized_image


def controller(model_name, device, use_webcam):
    """Called when Start is clicked. Sets up video source and spawns processing thread."""
    global compiled_model, compiled_model_dict, USE_WEBCAM_state, vid_in
    global H, W, thread_active, no_filter, output_layer

    cam_id = 0
    video_candidates = [
        "Coco%20Walking%20in%20Berkeley.mp4",
        "Coco Walking in Berkeley.mp4",
    ]
    video_file = next((v for v in video_candidates if Path(v).exists()), video_candidates[0])
    source = cam_id if use_webcam else video_file

    if model_name is None or device is None:
        cleanup()
        return "Stopped"

    if model_name == "No filter":
        no_filter = True
        H, W = 224, 224
        output_layer = 0
    else:
        no_filter = False
        key_name = f"{model_name}_{device}"
        compiled_model = compiled_model_dict.get(key_name)
        if compiled_model is None:
            cleanup()
            return f"Model not loaded for {model_name} on {device}"

        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        _, _, H, W = list(input_layer.shape)

    if vid_in is None or USE_WEBCAM_state != use_webcam:
        vid_in = cv2.VideoCapture(source)
        if not vid_in.isOpened():
            cleanup()
            return "Error: Could not open webcam or video source."
        USE_WEBCAM_state = use_webcam

        if not thread_active:
            vid_thread = threading.Thread(target=run_style_transfer, daemon=True)
            vid_thread.start()
            thread_active = True

    return f"Running {model_name} on {device}"


def run_style_transfer():
    """Video loop in background thread. Displays in cv2 popup on top of browser."""
    global compiled_model, vid_in, stop_processing, output_layer, H, W
    global USE_WEBCAM_state, no_filter, _processing_times

    title = "Style Transfer - Press Q or click Clear to stop"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)

    while not stop_processing:
        ret, frame = vid_in.read()
        if not ret:
            break

        if USE_WEBCAM_state:
            frame = cv2.flip(frame, 1)

        scale = 720 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(
                src=frame,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        start_time = time.time()

        if no_filter:
            result_image = frame.copy()
        else:
            image = preprocess_rgb(frame, H, W)
            stylized_image = compiled_model([image])[output_layer]
            result_image = convert_result_to_image(frame.shape, stylized_image)
            _processing_times.append(time.time() - start_time)

        processing_time_det = (time.time() - start_time) * 1000
        f_width = frame.shape[1]
        fps = 1000 / processing_time_det if processing_time_det > 0 else 0.0
        text = "No filter" if no_filter else f"Inference: {processing_time_det:.1f}ms ({fps:.1f} FPS)"

        cv2.putText(
            result_image,
            text=text,
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=f_width / 1000,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        cv2.imshow(title, result_image)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cleanup()


def cleanup():
    """Stop video, release resources, close cv2 windows."""
    global vid_in, stop_processing, thread_active

    stop_processing = True
    thread_active = False

    if vid_in is not None:
        vid_in.release()
        vid_in = None

    cv2.destroyAllWindows()
    stop_processing = False


def build_gr_interface():
    global compiled_model_dict
    compiled_model_dict = load_models(style_options, core.available_devices)
    device_options = core.available_devices

    with gr.Blocks(title="Style Transfer") as demo:
        gr.Markdown("# Neural Style Transfer with OpenVINO")

        style_dd = gr.Dropdown(
            style_options,
            label="Choose a style",
            value=style_options[0],
            interactive=True,
        )
        device_dd = gr.Dropdown(
            device_options,
            label="Choose a device",
            value=device_options[0],
            interactive=True,
        )
        use_webcam = gr.Checkbox(label="Use webcam?", value=True)

        with gr.Row():
            start_btn = gr.Button("Start", variant="primary")
            clear_btn = gr.Button("Clear inputs and stop", variant="stop")

        status = gr.Textbox(label="Status", interactive=False)

        start_btn.click(
            fn=controller,
            inputs=[style_dd, device_dd, use_webcam],
            outputs=status,
        )

        def do_cleanup():
            cleanup()
            return "Stopped"

        clear_btn.click(fn=do_cleanup, inputs=None, outputs=status)

    return demo
