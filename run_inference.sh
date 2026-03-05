#!/bin/bash
# LTX Video OpenVINO GenAI - Run inference
# Run: bash run_inference.sh
#      bash run_inference.sh "Your prompt"
#      bash run_inference.sh --int8 "Your prompt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found. Run install.sh first."
    exit 1
fi

source .venv/bin/activate
python run_inference.py "$@"
