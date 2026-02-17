# YOLOv8 Instance Segmentation with OpenVINO

A standalone demo for running YOLOv8 instance segmentation using the [OpenVINO™ Toolkit](https://docs.openvino.ai/) for optimized inference. Based on [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks).

## Features

- Convert YOLOv8-seg PyTorch model to OpenVINO IR (FP32/FP16)
- Optional INT8 quantization via NNCF for smaller, faster models
- Live webcam/video inference
- Benchmark comparison (PyTorch vs OpenVINO FP32 vs OpenVINO INT8)

## Prerequisites

- **Python** 3.10–3.13 (64-bit)
- **Git** (for cloning)

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install -y python3-venv python3-dev libgl1-mesa-dev ffmpeg
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/yolov8-openvino-seg-demo.git
cd yolov8-openvino-seg-demo
```

### Windows

```powershell
.\run_yolov8_seg_demo.ps1
```

### Linux

```bash
chmod +x run_yolov8_seg_demo.sh
./run_yolov8_seg_demo.sh
```

The script creates a virtual environment, installs dependencies, and launches Jupyter Lab with the ISV demo notebook.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `yolov8_seg_isv_demo.ipynb` | Full ISV demo: conversion, benchmark, quantization, validation, live inference |
| `yolov8_seg_comparison_demo.ipynb` | Comparison view: side-by-side PyTorch, FP32, INT8; live demos |

## Options

### Rebuild environment

```bash
# Windows
.\run_yolov8_seg_demo.ps1 -Rebuild

# Linux
./run_yolov8_seg_demo.sh --rebuild
```

### Custom port

```bash
# Windows
.\run_yolov8_seg_demo.ps1 -Port 8889

# Linux
./run_yolov8_seg_demo.sh --port 8889
```

### Remote access (Linux)

For SSH or remote access:

```bash
./run_yolov8_seg_demo.sh --ip 0.0.0.0
```

Then on your local machine:

```bash
ssh -L 8888:localhost:8888 user@your-linux-host
```

Open `http://localhost:8888` in your browser.

### Headless Linux (no display)

If running on a server without a display, use `use_popup=False` in the live demo cell so frames render in the browser instead of an OpenCV window.

## License

Apache 2.0 — see [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks) for reference.
