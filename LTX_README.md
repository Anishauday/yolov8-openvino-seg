# LTX Video + OpenVINO GenAI

Text-to-video with **OpenVINO GenAI** `Text2VideoPipeline` and **Lightricks/LTX-Video**.

---

## Folder Location

```
c:\Users\anish\Cursor\Openvino\genai\ltx_video_demo\
```

On Linux (after copy): `/path/to/ltx_video_demo/`

### Files in This Folder

| File | Purpose |
|------|---------|
| `install.sh` | One-time setup: venv, deps, FP16/INT8 export |
| `run_inference.sh` | Runs inference (forwards args to Python) |
| `run_inference.py` | Inference using OpenVINO GenAI Text2VideoPipeline |
| `LTX_README.md` | This file |
| `LTX-Video/` | Created by install: FP16 and/or INT8 models |

---

## Summary

| Item | Value |
|------|-------|
| Model | Lightricks/LTX-Video |
| Formats | FP16 (default), INT8 (`--quantize` at install) |
| Pipeline | OpenVINO GenAI Text2VideoPipeline |
| Default resolution | 384×384 (avoids Intel GPU issues) |
| Device | CPU (default) or GPU (`-d GPU`) |
| Skip logic | Skips venv/model steps if already present |

---

## Prerequisites

- Linux (Ubuntu 20.04+, RHEL, etc.)
- Python 3.9+
- ~16 GB RAM (export)
- ~10 GB disk

---

## Quick Start

### 1. Copy to Linux server

```bash
scp -r ltx_video_demo user@server:/path/to/
```

### 2. Fix line endings (if from Windows)

```bash
cd /path/to/ltx_video_demo
sed -i 's/\r$//' install.sh run_inference.sh
chmod +x install.sh run_inference.sh
```

### 3. Install

```bash
bash install.sh              # FP16 only
bash install.sh --quantize    # FP16 + INT8
```

Skips export if `LTX-Video/FP16` or `LTX-Video/INT8` already exists.

### 4. Run inference

```bash
bash run_inference.sh                    # Default prompt, FP16
bash run_inference.sh "A cat walking"    # Custom prompt
bash run_inference.sh --int8 "Waves"    # INT8 model + prompt
```

### 5. Output

`output_ltx.avi` and `output_ltx.mp4` in the same folder.

---

## Options

| Flag | Usage |
|------|-------|
| (none) | Default prompt, FP16, 384×384, CPU |
| `"prompt"` | Custom prompt |
| `--int8` | Use INT8 model (fallback to FP16 if INT8 missing) |
| `-d CPU` | Use CPU (default; recommended if GPU has igc_check error) |
| `-d GPU` | Use GPU |
| `--height 480` | Video height (divisible by 32; default: 384) |
| `--width 704` | Video width (divisible by 32; default: 384) |
| `--frames 25` | Number of frames, e.g. 9, 17, 25 (default: 17) |
| `--steps 25` | Inference steps (default: 25) |
| `-p "prompt"` | Prompt via `-p` |

```bash
python run_inference.py --help

# Examples
bash run_inference.sh -d CPU "A river"              # Avoid GPU errors
bash run_inference.sh --width 704 --height 480 "X"  # Higher resolution
```

---

## One-Liner Setup

```bash
cd /path/to/ltx_video_demo && sed -i 's/\r$//' *.sh && chmod +x *.sh && bash install.sh
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `igc_check` / pixelated output on Intel GPU | Use CPU: `-d CPU` or lower res: `--width 384 --height 384` |
| `bad interpreter` / `^M` | `sed -i 's/\r$//' *.sh` |
| `Permission denied` | `chmod +x *.sh` |
| Export OOM | 16+ GB RAM or `--weight-format fp32` |
| Hugging Face 401 | `huggingface-cli login` |
