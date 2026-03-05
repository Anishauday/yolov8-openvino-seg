# YOLOX-s Object Detection with OpenVINO

This tutorial demonstrates how to run and optimize the YOLOX-s object detection model with OpenVINO. It covers model loading, NNCF quantization (INT8), and an interactive Gradio demo for image upload and webcam detection.

## Prerequisites

- Python 3.9+
- OpenVINO 2026.0+
- NNCF 2.19+

## Quick Start

### 1. Create and activate a virtual environment

**Linux / macOS**

```bash
cd notebooks/yolox-optimization
python -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt)**

```cmd
cd notebooks\yolox-optimization
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
cd notebooks\yolox-optimization
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

**Linux / macOS / Windows (same command)**

```bash
pip install "openvino>=2026.0.0" "nncf>=2.19.0"
pip install "torch>=2.8" "torchvision>=0.16" onnx tqdm opencv-python gradio matplotlib ipywidgets
```

### 3. Run the notebook

**Linux / macOS**

```bash
jupyter notebook yolox-object-detection.ipynb
```

**Windows (Command Prompt)**

```cmd
jupyter notebook yolox-object-detection.ipynb
```

**Windows (PowerShell)**

```powershell
jupyter notebook yolox-object-detection.ipynb
```

Or use JupyterLab:

```bash
jupyter lab yolox-object-detection.ipynb
```

### 4. Run from project root (alternative)

If you prefer to run from the project root:

**Linux / macOS**

```bash
cd /path/to/openvino_notebooks
jupyter notebook notebooks/yolox-optimization/yolox-object-detection.ipynb
```

**Windows**

```cmd
cd C:\path\to\openvino_notebooks
jupyter notebook notebooks\yolox-optimization\yolox-object-detection.ipynb
```

## Notebook Structure

1. **Setup** – Install OpenVINO, NNCF, and dependencies
2. **Model Acquisition** – Download YOLOX-s ONNX model
3. **Inference Utilities** – Preprocessing, postprocessing, NMS
4. **Load OpenVINO Model** – Convert and compile
5. **Run Inference** – Test on sample image
6. **NNCF Quantization** – INT8 Post-Training Quantization
7. **Gradio Demo** – Upload, webcam, sample images

## Platform Notes

- **Paths**: The notebook uses `pathlib.Path` for cross-platform path handling (Linux, Windows, macOS).
- **Model directory**: Created as `model/yolox_s/` relative to the notebook.
- **Data directory**: Created as `data/` for sample images.
- **COCO128 (optional)**: For NNCF calibration, the notebook looks for `../yolov8-optimization/datasets/coco128/`; if missing, it uses images from `data/`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: utils` | Run the notebook from `notebooks/yolox-optimization` so `utils/` is on the path |
| OpenVINO GPU not found | Install GPU drivers and OpenVINO GPU plugin; fallback to CPU is automatic |
| Gradio webcam fails | Ensure browser has camera permission; try HTTPS or `localhost` |
