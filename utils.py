"""
Utility functions for object detection notebooks.
Provides download_video, avi_to_mp4, run_inference_save_mp4, run_inference_on_image, run_live_inference,
and display_video (embeds MP4 or GIF for playback in Cursor/VS Code/Jupyter).
"""

from pathlib import Path

import cv2
import requests


def display_video(video_path: str, use_mp4: bool = True, max_size_mb: float = 15, max_seconds: float = 8, max_width: int = 480, fps: int = 6):
    """
    Display a video file in a Jupyter notebook.

    Tries MP4 first (full quality, native playback). Falls back to GIF if MP4
    is too large or fails. Works in Cursor, VS Code, and Jupyter Lab/Notebook.

    :param video_path: Path to the MP4 video file
    :param use_mp4: If True, embed MP4 directly (fall back to GIF if too large)
    :param max_size_mb: Max MP4 size for embedding (larger files use GIF)
    :param max_seconds: For GIF fallback: max duration (seconds)
    :param max_width: For GIF fallback: max frame width
    :param fps: For GIF fallback: frames per second
    """
    import base64
    import tempfile

    from IPython.display import HTML, Image, display

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size_mb = path.stat().st_size / (1024 * 1024)

    if use_mp4 and file_size_mb <= max_size_mb:
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            html = f'<video controls width="100%" playsinline><source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'
            display(HTML(html))
            return
        except Exception:
            pass

    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(video_fps / fps))
    max_frames = min(int(max_seconds * fps), total_frames // frame_skip) if total_frames else int(max_seconds * fps)

    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        for _ in range(frame_skip - 1):
            cap.read()
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * scale)))
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(f"Could not read frames from: {video_path}")

    import imageio
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        imageio.mimsave(tmp.name, frames, fps=fps, loop=0)
        with open(tmp.name, "rb") as f:
            gif_data = f.read()
        Path(tmp.name).unlink(missing_ok=True)

    display(Image(data=gif_data))


def run_inference_on_image(model, image_path: str, device: str = None):
    """
    Run object detection on a single image.

    :param model: YOLO model (PyTorch or OpenVINO)
    :param image_path: Path to input image
    :param device: Optional device string (e.g. "intel:cpu", "intel:gpu")
    :return: Annotated BGR image (numpy array)
    """
    kwargs = {"verbose": False}
    if device:
        kwargs["device"] = device
    results = model(str(image_path), **kwargs)
    return results[0].plot()


def download_video(url: str, filepath: str) -> Path:
    """
    Download a video from a URL and save it to the specified path.

    :param url: URL that points to the video file to download
    :param filepath: Full path (including directory and filename) to save the video
    :return: Path to the downloaded file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    filesize = int(response.headers.get("Content-Length", 0))
    with open(filepath, "wb") as f:
        try:
            from tqdm import tqdm
            with tqdm(total=filesize or None, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(16384):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except ImportError:
            for chunk in response.iter_content(16384):
                f.write(chunk)

    print(f"Download complete: {filepath}")
    return filepath.resolve()


def avi_to_mp4(avi_path: str) -> str:
    """
    Convert an AVI video file to MP4 format using MoviePy.

    :param avi_path: Path to the input AVI file
    :return: Path to the output MP4 file
    """
    from moviepy import VideoFileClip

    avi_path = Path(avi_path)
    mp4_path = avi_path.with_suffix(".mp4")

    clip = VideoFileClip(str(avi_path))
    clip.write_videofile(
        str(mp4_path),
        codec="libx264",
        pixel_format="yuv420p",
        audio=False,
        logger=None,
    )
    clip.close()

    return str(mp4_path)


def run_inference_save_mp4(model, source: str, output_path: str, device: str = None):
    """
    Run object detection on a video and save annotated output directly as MP4.

    Uses imageio with libx264/yuv420p for HTML5/browser compatibility
    (OpenCV mp4v is not playable in notebooks/browsers). Pads frames to be
    divisible by 16 to avoid imageio's automatic resize and macro_block warnings.

    :param model: YOLO model (PyTorch or OpenVINO)
    :param source: Path to input video
    :param output_path: Path for output MP4 file
    :param device: Optional device string (e.g. "intel:cpu", "intel:gpu")
    :return: Tuple of (output_path, list of inference times in ms)
    """
    import imageio
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    inference_times = []
    kwargs = {"stream": True, "verbose": False}
    if device:
        kwargs["device"] = device

    def _pad_for_codec(frame):
        """Pad frame so dimensions are divisible by 16 (H.264 macroblock)."""
        h, w = frame.shape[:2]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        return frame

    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", pixelformat="yuv420p"
    )
    try:
        for result in model(source, **kwargs):
            annotated = result.plot()
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            writer.append_data(_pad_for_codec(rgb))
            if "inference" in result.speed:
                inference_times.append(result.speed["inference"])
    finally:
        writer.close()

    return str(output_path.resolve()), inference_times


def run_live_inference(model, device: str = None):
    """
    Run live object detection on webcam in a popup window.
    Exit: press 'q' or close the window (X button).

    :param model: YOLO model (PyTorch or OpenVINO)
    :param device: Optional device string (e.g. "intel:cpu", "intel:gpu")
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Is a camera connected?")

    kwargs = {"verbose": False}
    if device:
        kwargs["device"] = device

    window_name = "Live - Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, **kwargs)
            annotated = results[0].plot()
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
