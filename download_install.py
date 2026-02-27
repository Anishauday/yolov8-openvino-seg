# Run with venv active: env_ov\Scripts\Activate.ps1 (Windows)
# First time: python -m venv env_ov

import os

os.system('pip install -q "openvino>=2023.1.0"')
os.system('pip install -q opencv-python requests tqdm')
os.system('pip install -q --upgrade gradio')

os.system('pip install -q jupyterlab ipywidgets ipykernel')
os.system('python -m ipykernel install --user --name OpenVINO')

# Fetch `notebook_utils` module
import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)

from pathlib import Path
import notebook_utils as utils
import openvino as ov

# Pre-download all the ONNX models
models = ["mosaic", "rain-princess", "candy", "udnie", "pointilism"]

# Directory to download the model from ONNX model zoo
base_model_dir = "model"
base_url = "https://github.com/onnx/models/raw/69d69010b7ed6ba9438c392943d2715026792d40/archive/vision/style_transfer/fast_neural_style/model"

for model in models:

    model_path = Path(f"{model}-9.onnx")

    style_url = f"{base_url}/{model_path}"
    utils.download_file(style_url, directory=base_model_dir)

    ov_model = ov.convert_model(f"model/{model}-9.onnx")
    ov.save_model(ov_model, f"model/{model}-9.xml")

# download a video file
utils.download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4",
    directory="."
)
