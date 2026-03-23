# Car Detection — OpenVINO (Intel geti deployment)

Car detection with OpenVINO using an Intel geti-exported ATSS-MobileNetV2 INT8 model.

## Replicate on Linux (Remote Server)

### 1. Prerequisites

- Python 3.9+
- Linux (Ubuntu 20.04+ or similar)

### 2. Clone/Copy Project

Copy these files to your server:

```
project/
├── OpenVINO_Inference.ipynb   # Main notebook
├── ov_utils.py                # Inference utilities
├── app_gradio.py              # Gradio demo
├── requirements.txt
├── README.md
├── deployment/                # Model folder (required)
│   └── Detection/
│       ├── model/
│       │   ├── model.xml
│       │   ├── model.bin
│       │   └── config.json
│       └── model.json
├── sample_image.jpg           # Optional: sample image
└── Car_video.mp4             # Optional: sample video
```

### 3. Create Virtual Environment

```bash
cd /path/to/project

# Create venv
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate
```

### 4. Upgrade pip and Install Dependencies

```bash
python -m pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.txt
```

### 5. Install Jupyter (for notebook)

```bash
pip install jupyter notebook ipykernel
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook OpenVINO_Inference.ipynb
```

Or launch JupyterLab:

```bash
pip install jupyterlab
jupyter lab OpenVINO_Inference.ipynb
```

### 7. Run Gradio App (standalone)

```bash
python app_gradio.py
```

---

## File Summary for Deployment

| File | Purpose |
|------|---------|
| `OpenVINO_Inference.ipynb` | Main notebook — inference, video, benchmark |
| `ov_utils.py` | Load model, run detection, benchmark |
| `app_gradio.py` | Gradio app (sample/upload/webcam) |
| `requirements.txt` | Python dependencies |
| `deployment/` | Model IR (model.xml, model.bin, config.json) |
| `sample_image.jpg` | Sample image (optional) |
| `Car_video.mp4` | Sample video (optional) |

## Requirements

- openvino>=2024.3
- numpy, opencv-python, matplotlib, gradio
