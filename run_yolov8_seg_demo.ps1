# run_yolov8_seg_demo.ps1
# Launch YOLOv8 Instance Segmentation ISV demo (standalone)
#
# Run from: yolov8-openvino-seg-demo/
#
# Usage: .\run_yolov8_seg_demo.ps1
#        .\run_yolov8_seg_demo.ps1 -Rebuild    # Delete venv and reinstall from scratch
#        .\run_yolov8_seg_demo.ps1 -Port 8889  # Custom port

param([switch]$Rebuild, [int]$Port = 8888)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$VenvPath = Join-Path $RepoRoot "openvino_env"
$VenvPathNew = Join-Path $RepoRoot "openvino_env_new"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$PipExe = Join-Path $VenvPath "Scripts\pip.exe"

Set-Location $RepoRoot

# Use openvino_env_new if openvino_env is missing or broken
$AltPythonExe = Join-Path $VenvPathNew "Scripts\python.exe"
$PyvenvCfg = Join-Path $VenvPath "pyvenv.cfg"
$UseAltVenv = $false
if (-not (Test-Path $PythonExe)) {
    $UseAltVenv = (Test-Path $AltPythonExe)
} elseif (-not (Test-Path $PyvenvCfg)) {
    $UseAltVenv = (Test-Path $AltPythonExe)
}
if ($UseAltVenv) {
    $VenvPath = $VenvPathNew
    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
    $PipExe = Join-Path $VenvPath "Scripts\pip.exe"
    Write-Host "Using openvino_env_new (openvino_env missing or broken)" -ForegroundColor Yellow
}

# --- Rebuild: delete venv ---
if ($Rebuild) {
    Write-Host "`n[Rebuild] Removing openvino_env..." -ForegroundColor Yellow
    if (Test-Path $VenvPath) {
        try {
            Remove-Item -Recurse -Force $VenvPath -ErrorAction Stop
            Write-Host "  openvino_env deleted." -ForegroundColor Green
        } catch {
            Write-Host "  ERROR: Could not delete openvino_env. Close all Python/Jupyter processes and try again." -ForegroundColor Red
            Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    }
}

# --- Create venv if missing ---
if (-not (Test-Path $PythonExe)) {
    Write-Host "`n[Step 5] Creating virtual environment openvino_env..." -ForegroundColor Yellow
    python -m venv openvino_env
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
    Write-Host "  Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "`n[Step 5] Virtual environment exists." -ForegroundColor Green
}

# --- Upgrade pip, install requirements ---
Write-Host "`n[Step 8] Upgrading pip, wheel, setuptools..." -ForegroundColor Yellow
& $PythonExe -m pip install --upgrade pip wheel setuptools -q
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "  Installing requirements.txt..." -ForegroundColor Yellow
& $PipExe install -r requirements.txt --extra-index-url "https://download.pytorch.org/whl/cpu" -q
if ($LASTEXITCODE -ne 0) { Write-Host "  Warning: Install had issues. Retrying without -q..." -ForegroundColor Yellow; & $PipExe install -r requirements.txt --extra-index-url "https://download.pytorch.org/whl/cpu" }

Write-Host "`n  Install complete." -ForegroundColor Green

# --- Verify ---
Write-Host "`n[Verify] Checking imports..." -ForegroundColor Yellow
& $PythonExe -c "import openvino; import nncf; import torch; from ultralytics import YOLO; import cv2; import matplotlib; print('OK')"
if ($LASTEXITCODE -ne 0) { throw "Import verification failed" }
Write-Host "  Imports OK." -ForegroundColor Green

# --- Launch notebook ---
Write-Host "`n=== YOLOv8 Instance Segmentation ISV Demo ===" -ForegroundColor Cyan
Write-Host "Launching Jupyter Lab at http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "Notebook: yolov8_seg_isv_demo.ipynb`n" -ForegroundColor Cyan

& $PythonExe -m jupyter lab --ip=127.0.0.1 --port=$Port --no-browser "yolov8_seg_isv_demo.ipynb"
