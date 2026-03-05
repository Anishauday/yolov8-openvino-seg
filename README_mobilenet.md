# MobileNetV3-Large Image Classification with OpenVINO

Image classification using MobileNetV3-Large and OpenVINO 2026. Classifies images into 1000 ImageNet classes. Supports NNCF INT8 quantization (skipped if quantized model already exists).

## Prerequisites

- Python 3.9+
- OpenVINO 2026.0+
- NNCF 2.19+

## Quick Start

### 1. Create and activate a virtual environment

**Linux / macOS**

```bash
cd notebooks/mobilenetv3-classification
python -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt)**

```cmd
cd notebooks\mobilenetv3-classification
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
cd notebooks\mobilenetv3-classification
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install "openvino>=2026.0.0" "nncf>=2.19.0"
pip install "torch>=2.8" "torchvision>=0.16" onnx tqdm opencv-python gradio matplotlib ipywidgets
```

### 3. Run the notebook

**Linux / macOS**

```bash
jupyter notebook mobilenet-classification.ipynb
```

**Windows**

```cmd
jupyter notebook mobilenet-classification.ipynb
```

## Notebook Structure

1. **Setup** – Install OpenVINO, NNCF, and dependencies
2. **Model Acquisition** – Convert PyTorch MobileNetV3-Large to OpenVINO IR
3. **Inference Utilities** – ImageNet preprocessing, top-k, visualization
4. **Load OpenVINO Model** – Compile for CPU/GPU
5. **Run Inference** – Test on sample image with class label on image
6. **NNCF Quantization** – INT8 PTQ (only if `mobilenet_v3_large_int8.xml` does not exist)
7. **Gradio Demo** – Upload, webcam, sample image; classification label drawn on result

## Quantization Logic

- If `model/mobilenet_v3_large/mobilenet_v3_large_int8.xml` exists: loads it and skips NNCF.
- If not: runs NNCF PTQ, saves the model, then uses it.
- The Gradio demo uses the quantized model when available (faster inference).

## Platform Notes

- **Paths**: Uses `pathlib.Path` for cross-platform compatibility.
- **ImageNet labels**: Loaded from PyTorch tutorials (requires internet on first run).
