#!/bin/bash
# run_yolov8_seg_demo.sh
# Launch YOLOv8 Instance Segmentation ISV demo on Linux (standalone)
#
# Run from: yolov8-openvino-seg-demo/
#
# Usage: ./run_yolov8_seg_demo.sh
#        ./run_yolov8_seg_demo.sh --rebuild        # Delete venv and reinstall from scratch
#        ./run_yolov8_seg_demo.sh --port 8889      # Custom Jupyter port
#        ./run_yolov8_seg_demo.sh --ip 0.0.0.0     # Allow remote access (e.g. SSH tunnel)

set -e

REBUILD=false
PORT=8888
IP="127.0.0.1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --ip)
            IP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--rebuild] [--port PORT] [--ip IP]"
            echo "  --rebuild   Delete venv and reinstall"
            echo "  --port N    Jupyter port (default: 8888)"
            echo "  --ip IP     Bind address: 127.0.0.1 (local) or 0.0.0.0 (remote)"
            exit 1
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$REPO_ROOT/openvino_env"
VENV_PATH_NEW="$REPO_ROOT/openvino_env_new"
PYTHON="$VENV_PATH/bin/python"
PIP="$VENV_PATH/bin/pip"

cd "$REPO_ROOT"

# Use openvino_env_new if openvino_env is missing or broken
if [[ ! -x "$PYTHON" ]]; then
    if [[ -x "$VENV_PATH_NEW/bin/python" ]]; then
        VENV_PATH="$VENV_PATH_NEW"
        PYTHON="$VENV_PATH/bin/python"
        PIP="$VENV_PATH/bin/pip"
        echo -e "\nUsing openvino_env_new (openvino_env missing or broken)"
    fi
elif [[ ! -f "$VENV_PATH/pyvenv.cfg" ]]; then
    if [[ -x "$VENV_PATH_NEW/bin/python" ]]; then
        VENV_PATH="$VENV_PATH_NEW"
        PYTHON="$VENV_PATH/bin/python"
        PIP="$VENV_PATH/bin/pip"
        echo -e "\nUsing openvino_env_new (openvino_env missing or broken)"
    fi
fi

# --- Rebuild: delete venv ---
if [[ "$REBUILD" == "true" ]]; then
    echo -e "\n[Rebuild] Removing openvino_env..."
    if [[ -d "$VENV_PATH" ]]; then
        rm -rf "$VENV_PATH"
        echo "  openvino_env deleted."
    fi
    VENV_PATH="$REPO_ROOT/openvino_env"
    PYTHON="$VENV_PATH/bin/python"
    PIP="$VENV_PATH/bin/pip"
fi

# --- Create venv if missing ---
if [[ ! -x "$PYTHON" ]]; then
    echo -e "\n[Step 5] Creating virtual environment openvino_env..."
    python3 -m venv openvino_env
    echo "  Virtual environment created."
else
    echo -e "\n[Step 5] Virtual environment exists."
fi

# --- Upgrade pip, install requirements ---
echo -e "\n[Step 8] Upgrading pip, wheel, setuptools..."
"$PYTHON" -m pip install --upgrade pip wheel setuptools -q

echo "  Installing requirements.txt..."
if ! "$PIP" install -r requirements.txt --extra-index-url "https://download.pytorch.org/whl/cpu" -q 2>/dev/null; then
    echo "  Warning: Install had issues. Retrying without -q..."
    "$PIP" install -r requirements.txt --extra-index-url "https://download.pytorch.org/whl/cpu"
fi

echo -e "\n  Install complete."

# --- Verify ---
echo -e "\n[Verify] Checking imports..."
"$PYTHON" -c "import openvino; import nncf; import torch; from ultralytics import YOLO; import cv2; import matplotlib; print('OK')"
echo "  Imports OK."

# --- Launch notebook ---
echo -e "\n=== YOLOv8 Instance Segmentation ISV Demo ==="
echo "Launching Jupyter Lab at http://${IP}:${PORT}"
echo "Notebook: yolov8_seg_isv_demo.ipynb"
echo ""

"$PYTHON" -m jupyter lab --ip="$IP" --port="$PORT" --no-browser "yolov8_seg_isv_demo.ipynb"
