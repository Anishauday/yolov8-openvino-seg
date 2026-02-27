#!/bin/bash
# OpenVINO Build Deploy - Linux Installer for trainings
# Creates venv, installs deps, and launches Jupyter Lab from the trainings folder.
#
# If you get "bad interpreter" or "command not found", fix Windows line endings first:
#   sed -i 's/\r$//' install_linux.sh
#   (or: dos2unix install_linux.sh)

set -e

# Self-fix CRLF when script was saved with Windows line endings
SCRIPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
if grep -q $'\r' "$SCRIPTPATH" 2>/dev/null; then
  echo "Fixing Windows line endings (CRLF -> LF)..."
  sed -i 's/\r$//' "$SCRIPTPATH"
  exec bash "$SCRIPTPATH" "$@"
fi

REPO_URL="https://github.com/openvinotoolkit/openvino_build_deploy.git"
VENV_NAME="env_ov"
WORK_DIR="$HOME/work"

echo "=== OpenVINO Trainings Linux Installer ==="

# Detect trainings directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/../.git" ]] && [[ "$(basename "$SCRIPT_DIR")" == "trainings" ]]; then
    TRAININGS_DIR="$SCRIPT_DIR"
    echo "Using existing repo at: $TRAININGS_DIR"
else
    echo "Repo not found. Cloning to $WORK_DIR..."
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    if [[ -d "openvino_build_deploy" ]]; then
        echo "Repository already exists. Updating..."
        cd openvino_build_deploy
        git pull
    else
        git clone "$REPO_URL"
        cd openvino_build_deploy
    fi
    TRAININGS_DIR="$WORK_DIR/openvino_build_deploy/trainings"
    cd "$TRAININGS_DIR"
fi

echo "Trainings folder: $TRAININGS_DIR"
cd "$TRAININGS_DIR"

# System packages (sudo required)
echo ""
echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip
echo "Git version: $(git --version)"

# OpenCL for Intel GPU
sudo apt-get install -y ocl-icd-libopencl1 intel-opencl-icd 2>/dev/null || true
sudo usermod -aG render "$USER" 2>/dev/null || true
echo "Note: Log out and back in for render group to take effect (for GPU access)."

# Create or activate venv
echo ""
echo "=== Setting up Python virtual environment ==="
if [[ -d "$TRAININGS_DIR/$VENV_NAME" ]]; then
    echo "Virtual environment '$VENV_NAME' already exists. Activating..."
    source "$TRAININGS_DIR/$VENV_NAME/bin/activate"
else
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv "$TRAININGS_DIR/$VENV_NAME"
    source "$TRAININGS_DIR/$VENV_NAME/bin/activate"
fi

# Upgrade pip, wheel, setuptools
python -m pip install --upgrade pip wheel setuptools

# Core Python packages
echo ""
echo "=== Installing Python packages ==="
pip install -q "openvino>=2023.1.0"
pip install -q opencv-python requests tqdm gradio
pip install -q jupyterlab ipywidgets ipykernel

# Register Jupyter kernel
python -m ipykernel install --user --name OpenVINO

# Verify OpenVINO
echo ""
echo "=== Verifying OpenVINO ==="
python -c "
import openvino as ov
core = ov.Core()
devices = core.available_devices
print('OpenVINO available devices:', devices)
print('OpenVINO import OK.')
"

# Optional: object detection deps (uncomment if needed)
# pip install -q requests ultralytics nncf moviepy --extra-index-url https://download.pytorch.org/whl/cpu

echo ""
echo "=== Setup complete. Launching Jupyter Lab... ==="
cd "$TRAININGS_DIR"
exec jupyter lab --ip=0.0.0.0
