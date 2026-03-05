#!/usr/bin/env python3
"""
LTX Video inference with OpenVINO GenAI Text2VideoPipeline.
Model: Lightricks/LTX-Video (OpenVINO IR FP16 or INT8)
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_BASE = SCRIPT_DIR / "LTX-Video"
FP16_DIR = MODEL_BASE / "FP16"
INT8_DIR = MODEL_BASE / "INT8"
OUTPUT_AVI = SCRIPT_DIR / "output_ltx.avi"
OUTPUT_MP4 = SCRIPT_DIR / "output_ltx.mp4"
DEFAULT_PROMPT = "A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall"


def _model_ready(path: Path) -> bool:
    """Check if OpenVINO model is complete."""
    return (path / "transformer" / "openvino_model.xml").exists()


def _resolve_model_dir(use_int8: bool) -> tuple[Optional[Path], Optional[str]]:
    """Return (model_path, format_name). Prefer requested format; fallback if needed."""
    fp16_ok = _model_ready(FP16_DIR)
    int8_ok = _model_ready(INT8_DIR)

    if use_int8:
        if int8_ok:
            return INT8_DIR, "INT8 (quantized)"
        if fp16_ok:
            return FP16_DIR, "FP16 (INT8 not found, using FP16)"
        return None, None  # no model
    else:
        if fp16_ok:
            return FP16_DIR, "FP16"
        if int8_ok:
            return INT8_DIR, "INT8 (FP16 not found, using INT8)"
        return None, None


def _save_mp4_with_imageio(video_data, num_frames: int, height: int, width: int,
                          frame_rate: int, output_path: Path) -> None:
    import imageio
    import numpy as np
    frames = [np.ascontiguousarray(video_data[0, f]) for f in range(num_frames)]
    imageio.mimsave(str(output_path), frames, fps=frame_rate, codec="libx264")
    print(f"  >> OK: MP4 saved (via imageio)")


def main():
    parser = argparse.ArgumentParser(
        description="LTX Video text-to-video with OpenVINO GenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py
  python run_inference.py "A cat walking on grass"
  python run_inference.py --int8 "Waves on a beach"
  python run_inference.py -d GPU --int8 "Custom prompt"
        """,
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument("-p", "--prompt-alt", dest="prompt_opt", default=None, help="Prompt via -p")
    parser.add_argument(
        "-q", "--int8",
        dest="use_int8",
        action="store_true",
        help="Use INT8 quantized model (if available)",
    )
    parser.add_argument(
        "-d", "--device",
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="Device (default: CPU)",
    )
    args = parser.parse_args()
    prompt = args.prompt_opt or args.prompt

    # Resolve model
    model_dir, format_name = _resolve_model_dir(args.use_int8)
    if model_dir is None:
        print("ERROR: No model found.")
        print("  >> FP16: Run install.sh")
        print("  >> INT8: Run install.sh --quantize")
        sys.exit(1)

    # Header
    print("=" * 80)
    print("  LTX Video - OpenVINO GenAI Inference")
    print("=" * 80)
    print("  Model:   Lightricks/LTX-Video")
    print("  Format:  " + format_name)
    print("  Device:  " + args.device)
    print("=" * 80)
    print("")

    # Pre-flight
    print("[Step 1] Checking model...")
    print(f"  >> Path:   {model_dir}")
    print(f"  >> Format: {format_name}")
    print("  >> OK: Model found")
    print("")

    # Load
    print("[Step 2] Loading pipeline...")
    import openvino_genai
    import cv2

    load_start = time.perf_counter()
    pipe = openvino_genai.Text2VideoPipeline(str(model_dir), args.device)
    load_time_ms = (time.perf_counter() - load_start) * 1000
    print(f"  >> Pipeline: OpenVINO GenAI Text2VideoPipeline")
    print(f"  >> Device:   {args.device}")
    print(f"  >> Load:     {load_time_ms:.0f} ms")
    print("  >> OK: Ready")
    print("")

    # Generate
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    frame_rate = 25

    print("[Step 3] Generating video...")
    print(f"  >> Prompt: \"{prompt[:70]}{'...' if len(prompt) > 70 else ''}\"")
    print(f"  >> 704x480 | 25 frames | 25 steps | {args.device}")
    print("")

    def callback(step, num_steps, latent):
        print(f"  >> Step {step + 1}/{num_steps}")
        return False

    gen_start = time.perf_counter()
    output = pipe.generate(
        prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=704,
        num_frames=25,
        num_inference_steps=25,
        num_videos_per_prompt=1,
        callback=callback,
        frame_rate=frame_rate,
        guidance_scale=3,
    )
    gen_time_ms = (time.perf_counter() - gen_start) * 1000
    print("")

    # Save
    print("[Step 4] Saving video...")
    video_tensor = output.video
    video_data = video_tensor.data
    num_frames, height, width = video_tensor.shape[1], video_tensor.shape[2], video_tensor.shape[3]

    fourcc_avi = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(OUTPUT_AVI), fourcc_avi, frame_rate, (width, height))
    for f in range(num_frames):
        frame_bgr = cv2.cvtColor(video_data[0, f], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()
    print(f"  >> AVI: {OUTPUT_AVI}")

    try:
        fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_MP4), fourcc_mp4, frame_rate, (width, height))
        if writer.isOpened():
            for f in range(num_frames):
                frame_bgr = cv2.cvtColor(video_data[0, f], cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()
            print(f"  >> MP4: {OUTPUT_MP4}")
        else:
            _save_mp4_with_imageio(video_data, num_frames, height, width, frame_rate, OUTPUT_MP4)
    except Exception:
        _save_mp4_with_imageio(video_data, num_frames, height, width, frame_rate, OUTPUT_MP4)

    print(f"  >> Frames: {num_frames} | {width}x{height} @ {frame_rate} fps")
    print("")

    # Metrics
    print("[Step 5] Performance")
    print("  >> Device:        " + args.device)
    print("  >> Format:        " + format_name)
    print(f"  >> Load:          {load_time_ms:.0f} ms")
    print(f"  >> Inference:     {gen_time_ms:.0f} ms ({gen_time_ms/1000:.1f} s)")
    if hasattr(output, "perf_metrics") and output.perf_metrics:
        print(f"  >> Pipeline Gen:  {output.perf_metrics.get_generate_duration():.0f} ms")
    print("")

    # Final
    print("=" * 80)
    print("  DONE")
    print("=" * 80)
    print("  Model:     Lightricks/LTX-Video (" + format_name + ")")
    print("  Device:    " + args.device)
    print(f"  Inference: {gen_time_ms/1000:.1f} s")
    print("")
    print("  Video files:")
    print("    " + str(OUTPUT_AVI.resolve()))
    print("    " + str(OUTPUT_MP4.resolve()))
    print("=" * 80)


if __name__ == "__main__":
    main()
