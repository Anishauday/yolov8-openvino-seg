# Before running this, create a virtual environment and activate it
#
# python -m venv env_ovbm
# env_ovbm\Scripts\activate
#

import os

# PyTorch 2.4.0 on Windows was producing an error on import, saying fbgemm.dll or one of its dependencies was missing
os.system('pip install -q torch==2.3.0 torchvision==0.18.0')
os.system('pip install -q openvino-dev')
os.system('pip install -q "optimum-intel"')
os.system('pip install -q pandas matplotlib opencv-python')
os.system('pip install -q --upgrade gradio')
os.system('pip install -q jupyterlab ipykernel ipywidgets')

os.system('python -m ipykernel install --user --name OpenVINO_benchmark')

"""
Models are already included in this package. The following will re-download everything EXCEPT mobilenet int8

os.system('omz_downloader --name mobilenet-v2-pytorch --output_dir models --cache_dir cache')
os.system('omz_converter --name mobilenet-v2-pytorch --precisions FP16,FP32 --download_dir models --output_dir models')
os.system('move models/public/mobilenet-v2-pytorch models/mobilenet-v2')

os.system('omz_downloader --name face-detection-adas-0001 --output_dir models --cache_dir cache')
os.system('omz_converter --name face-detection-adas-0001 --precisions FP16,FP32 --download_dir models --output_dir models')
os.system('move models/intel/face-detection-adas-0001 models/face_detect')

os.system('omz_downloader --name unet-camvid-onnx-0001 --output_dir models --cache_dir cache')
os.system('omz_converter --name unet-camvid-onnx-0001 --precisions FP16-INT8,FP16,FP32 --download_dir models --output_dir models')
os.system('move models/intel/unet-camvid-onnx-0001 models/unet-camvid')
"""
