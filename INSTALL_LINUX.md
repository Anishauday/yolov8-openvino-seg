# Linux Installer for OpenVINO Trainings

This document explains how to run the Linux installer to set up the OpenVINO trainings environment and launch Jupyter Lab.

## Quick Start (recommended)

From a Linux terminal:

```bash
# 1. Create work directory and clone the repo
mkdir -p ~/work
cd ~/work
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
cd openvino_build_deploy/trainings

# 2. Run the installer (from inside the trainings folder)
chmod +x install_linux.sh
./install_linux.sh
```

The script will:
- Install system packages (git, python3-venv, python3-pip)
- Install OpenCL libs for Intel GPU (`ocl-icd-libopencl1`, `intel-opencl-icd`)
- Add your user to the `render` group (for GPU access)
- Create virtual environment `env_ov` (or activate if it exists)
- Install pip, wheel, setuptools, jupyterlab, ipywidgets, OpenVINO, OpenCV, Gradio, etc.
- Verify OpenVINO import
- Launch Jupyter Lab on `0.0.0.0` (accessible from network)

---

## How to Run the Installer

### Option 1: Run after cloning (recommended)

```bash
cd openvino_build_deploy/trainings
chmod +x install_linux.sh
./install_linux.sh
```

### Option 2: Standalone bootstrap (clone + install in one go)

If you don't have the repo yet, run:

```bash
mkdir -p ~/work
cd ~/work
curl -sSL https://raw.githubusercontent.com/openvinotoolkit/openvino_build_deploy/master/trainings/install_linux.sh -o install_linux.sh
chmod +x install_linux.sh
./install_linux.sh
```

This will clone the repo into `~/work/openvino_build_deploy` and run the setup. Jupyter Lab will launch automatically at the end.

---

## After Installation: Manual Launch

```bash
cd ~/work/openvino_build_deploy/trainings   # or your trainings path
source env_ov/bin/activate
jupyter lab --ip=0.0.0.0
```

Then open the URL shown in the browser (e.g. `http://localhost:8888` or your machine's IP).

---

## Notes

- **Line endings:** If you get "bad interpreter" or "No such file or directory" when running the script, fix CRLF with: `sed -i 's/\r$//' install_linux.sh`
- **Render group:** To use Intel GPU, log out and log back in after the install so the `render` group takes effect.
- **VenV location:** The virtual environment is created at `trainings/env_ov`.
- **Kernel:** The Jupyter kernel named `OpenVINO` is registered for use in notebooks.
