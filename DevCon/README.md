# Car Detection + Color Classification — OpenVINO

Two-stage pipeline: (1) detect cars, (2) classify color per car. Labels: **CAR - [Color]** (no percentage).

## Structure

```
1/
├── OpenVINO_Detection_Classification.ipynb
├── ov_utils.py
├── app_gradio.py
├── requirements.txt
├── models/INT8/
│   ├── Detection/       # Car detection (model.xml, model.bin, config.json)
│   └── Classification/  # Color (model.xml, model.bin, config.json)
├── media/               # sample_image.jpg, sample_image_1.jpg … (Gradio shows up to 3 files named sample_image_*)
└── output/              # sample_image_out.jpg, Car_video_out.avi
```

---

## Ubuntu — Setup from Scratch

### 1. Prerequisites

- Ubuntu 20.04+ (or similar Linux)
- Python 3.9+

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

### 2. Create virtual environment

```bash
cd 1
python3 -m venv openvino_env
source openvino_env/bin/activate
```

### 3. Upgrade pip and install build tools

```bash
python -m pip install --upgrade pip
pip install wheel setuptools
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: OpenVINO, OpenCV, NumPy, Matplotlib, Gradio, Jupyter, JupyterLab, ipywidgets, Pillow.

### 5. Run notebook

```bash
jupyter lab OpenVINO_Detection_Classification.ipynb
```

Or for classic Jupyter:

```bash
jupyter notebook OpenVINO_Detection_Classification.ipynb
```

### 6. Run Gradio app (optional)

```bash
python app_gradio.py
```

Gradio shows up to **three** clickable thumbnails for files in `media/` whose names start with **`sample_image_`** (e.g. `sample_image_1.jpg`, `sample_image_2.png`). Detection threshold is **`0.15`** (set in `ov_utils.get_notebook_config`).

---

## Quick reference

| Step | Command |
|------|---------|
| Activate venv | `source openvino_env/bin/activate` |
| Install deps | `pip install -r requirements.txt` |
| Run notebook | `jupyter lab OpenVINO_Detection_Classification.ipynb` |
| Run Gradio | `python app_gradio.py` |
