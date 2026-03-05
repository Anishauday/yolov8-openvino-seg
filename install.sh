#!/bin/bash
# LTX Video + OpenVINO GenAI - Linux installer
# Run: bash install.sh
#      bash install.sh --quantize    # Also export INT8 (quantized)
#
# If "bad interpreter" or "^M" error: run first:
#   sed -i 's/\r$//' install.sh run_inference.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse args
DO_QUANTIZE=false
for arg in "$@"; do
    case $arg in
        --quantize|-q) DO_QUANTIZE=true ;;
    esac
done

# Paths
MODEL_HF_ID="Lightricks/LTX-Video"
FP16_DIR="$SCRIPT_DIR/LTX-Video/FP16"
INT8_DIR="$SCRIPT_DIR/LTX-Video/INT8"

echo "================================================================================"
echo "  LTX Video + OpenVINO GenAI - Installation"
echo "================================================================================"
echo "  Model:       $MODEL_HF_ID"
echo "  Formats:     FP16 (default) | INT8 (if --quantize)"
echo "  FP16 path:   $FP16_DIR"
echo "  INT8 path:   $INT8_DIR"
echo "  Quantize:    $DO_QUANTIZE"
echo "================================================================================"
echo ""

# Pre-flight
echo "[Step 0] Prerequisites check..."
PYTHON_VER=$(python3 --version 2>&1)
echo "  >> Python:        $PYTHON_VER"
echo "  >> Model source:  Hugging Face ($MODEL_HF_ID)"
echo "  >> Pipeline:      OpenVINO GenAI Text2VideoPipeline"
echo "  >> OK: Ready"
echo ""

# Create venv (skip if exists)
echo "[Step 1/5] Virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  >> Created: .venv/"
else
    echo "  >> OK: .venv already exists (skipping)"
fi
source .venv/bin/activate

pip install --upgrade pip -q
echo ""

# Install export deps
echo "[Step 2/5] Export dependencies..."
pip install --upgrade-strategy eager \
    "torch>=2.1" "torchvision" "transformers>=4.40" "diffusers>=0.32" \
    "optimum-intel[openvino]" "huggingface-hub" "accelerate" \
    "sentencepiece" "einops" "safetensors" 2>&1 | tail -2
echo "  >> OK: PyTorch, Optimum-Intel, Diffusers"
echo ""

# Install OpenVINO GenAI
echo "[Step 3/5] OpenVINO GenAI..."
pip install --upgrade-strategy eager openvino-genai 2>&1 | tail -1
pip install opencv-python-headless imageio imageio-ffmpeg -q
echo "  >> OK: openvino-genai, OpenCV, imageio"
echo ""

# Export FP16
echo "[Step 4/5] FP16 model..."
if [ -f "$FP16_DIR/transformer/openvino_model.xml" ]; then
    echo "  >> OK: FP16 already exists at $FP16_DIR (skipping)"
else
    echo "  >> Exporting $MODEL_HF_ID -> FP16 (15-30 min, ~16GB RAM)..."
    optimum-cli export openvino \
        --model "$MODEL_HF_ID" \
        --task text-to-video \
        --weight-format fp16 \
        "$FP16_DIR"
    echo "  >> OK: FP16 exported to $FP16_DIR"
fi
echo ""

# Export INT8 (if --quantize)
echo "[Step 5/5] INT8 model (quantized)..."
if [ "$DO_QUANTIZE" = true ]; then
    if [ -f "$INT8_DIR/transformer/openvino_model.xml" ]; then
        echo "  >> OK: INT8 already exists at $INT8_DIR (skipping)"
    else
        echo "  >> Exporting $MODEL_HF_ID -> INT8 (may take 30+ min, ~16GB RAM)..."
        optimum-cli export openvino \
            --model "$MODEL_HF_ID" \
            --task text-to-video \
            --weight-format int8 \
            "$INT8_DIR"
        echo "  >> OK: INT8 exported to $INT8_DIR"
    fi
else
    echo "  >> Skipped (run with --quantize to export INT8)"
fi
echo ""

echo "================================================================================"
echo "  Installation complete"
echo "================================================================================"
echo "  FP16:  $FP16_DIR"
echo "  INT8:  $INT8_DIR $([ -f "$INT8_DIR/transformer/openvino_model.xml" ] && echo '(ready)' || echo '(run install.sh --quantize)')"
echo ""
echo "  Run:   bash run_inference.sh [\"prompt\"]"
echo "  INT8:  bash run_inference.sh --int8 [\"prompt\"]"
echo "================================================================================"
