# FramePack One-Frame Inference

Research adaptation of FramePack for single-frame image inference. This project adapts [FramePack](https://github.com/lllyasviel/FramePack)'s video generation models to enable efficient one-frame inference for image editing tasks.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [CLI](#cli)
  - [Common Options](#common-options)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- Simple CLI and Python API
- Optimized for speed - Model preloading, efficient VAE decoding, fast weight loading
- Flexible configuration

---

## Installation

### Requirements

- Python 3.10+
- CUDA GPU (40GB+ VRAM for BF16, 24GB+ for FP16)
- ~50GB disk space

### Setup

```bash
# 1. Create environment
conda create -n framepack python=3.10
conda activate framepack

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Clone repos
git clone https://github.com/lllyasviel/FramePack.git
git clone https://github.com/atharshlakshmi/framepack-oneframe.git
cd framepack-oneframe

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download models (using HuggingFace CLI)
pip install huggingface-hub
mkdir -p models/

# FramePack DiT (~24GB)
huggingface-cli download lllyasviel/FramePack_F1_I2V_HY_20250503 \
  FramePackI2V_HY_bf16.safetensors --local-dir models

# HunyuanVideo models
huggingface-cli download hunyuanvideo-community/HunyuanVideo \
  vae/pytorch_model.pt \
  text_encoder/model.safetensors \
  text_encoder_2/model.safetensors \
  image_encoder/model.safetensors \
  --local-dir models

# 6. Set FramePack path (choose one):
export FRAMEPACK_PATH=/path/to/FramePack  # Environment variable
# OR use .env file
cp .env.example .env  # Then edit .env
# OR place FramePack as sibling directory

# 7. Verify (remember to change: os.environ['FRAMEPACK_PATH'])
python -c "
import os, sys
os.environ['FRAMEPACK_PATH'] = '/path/to/FramePack'
sys.path.insert(0, os.environ['FRAMEPACK_PATH'])
from diffusers_helper import hunyuan
print('✅ Installation successful!')
"
```

### Directory Structure

**Complete Layout (recommended):**
```
parent_folder/                           # Your workspace
├── FramePack/                          # Clone from lllyasviel/FramePack
│   ├── diffusers_helper/               # Core utilities (imported by our code)
│   │   ├── hunyuan.py
│   │   ├── clip_vision.py
│   │   ├── models/
│   │   └── pipelines/
│   └── ...
│
└── framepack-oneframe/       # This project
    ├── src/                           # Source code
    │   ├── cli_inference.py           # Command-line interface
    │   ├── inference_engine.py        # Main inference orchestration
    │   ├── framepack_models.py        # Model loading & management
    │   ├── conditioning_pipeline.py   # Text & image conditioning
    │   ├── latent_packing.py          # Latent index management
    │   └── __init__.py                # Package exports
    ├── requirements.txt               # Python dependencies
    ├── .env.example                   # Environment configuration template
    ├── .env                           # Local environment config (user-created)
    ├── .gitignore                     # Git ignore rules
    ├── run_example.sh                 # Example shell script
    ├── README.md                      # This file
    ├── test_imports.py                # Import verification script
    ├── test_simple.py                 # Simple test script
    ├── example.png                    # Example Image
    │
    ├── models/                               # Downloaded model weights (~50GB total)
    │   ├── FramePackI2V_HY_bf16.safetensors  # DiT model (~24GB)
    │   ├── vae/pytorch_model.pt              # VAE (~941MB)
    │   ├── llava_llama3_fp16.safetensors     # LLaMA-3 (~15GB)
    │   ├── clip_l.safetensors                # CLIP-L (~235MB)
    │   └── model.safetensors                 # SiglipVision (~817MB)
    │
    └── output/                        # Generated images (created on first run)
        └── output2.png
```


---

## Usage

### CLI

TO BE UPDATED

**Or use the example script:**
Change paths and arguments in `run.sh` based on your preferences.
```bash
bash run.sh
```

**Performance (RTX A6000, 640x512):**
- First run: ~150s (90s loading + 26s sampling + 30s decode)
- Subsequent: ~60s (26s sampling + 30s decode)

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt` | *required* | Text edit description |
| `--negative_prompt` | "" | Things to avoid |
| `--infer_steps` | 25 | Sampling steps (more = quality) |
| `--guidance_scale` | 10.0 | Prompt strength (7-15) |
| `--height` / `--width` | 640 / 512 | Output size (÷64) |
| `--seed` | 42 | Random seed |
| `--dtype` | bfloat16 | fp16/bfloat16/fp32 |
| `--vae_tiling` | False | Lower VRAM mode |


## Examples

### High Quality
```bash
--infer_steps 50 --guidance_scale 12.0 --negative_prompt "blurry, low quality"
```

### Low VRAM (24GB GPU)
```bash
--vae_tiling --vae_chunk_size 32 --dtype fp16 --height 512 --width 384
```

### Custom Resolution
```bash
--height 768 --width 512
```

**Memory Requirements:**
- BF16: ~41GB (640x512)
- FP16: ~35GB (640x512)  
- FP16 + tiling: ~28GB (640x512)

---

## Troubleshooting

**FramePack not found:**
```bash
export FRAMEPACK_PATH=/path/to/FramePack
ls $FRAMEPACK_PATH/diffusers_helper  # Verify
```

**Out of memory:** Add `--vae_tiling --dtype fp16` or lower `--height 512 --width 384`

**CUDA not available:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"
```

**Slow generation:** Ensure running on GPU with `--verbose`. Use `--infer_steps 20` for speed.

---

## License

This project depends on:
- **FramePack** - Apache 2.0 License
- **HunyuanVideo** - Tencent License
---

## Acknowledgments

- **[FramePack](https://github.com/lllyasviel/FramePack)** by lllyasviel - Core diffusers_helper implementation and packed models
- **[musubi-tuner](https://github.com/kohya-ss/musubi-tuner)** by kohya-ss - Architecture reference for one-frame inference approach and optimization strategies (VAE tiling, fast weight loading)
- **[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)** by Tencent - Base models (VAE, text encoders)
---
