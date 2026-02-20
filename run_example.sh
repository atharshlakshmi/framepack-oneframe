#!/bin/bash
# Example CLI command for single-frame inference
# 
# This script automatically loads .env if it exists, or uses defaults

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment from .env file..."
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Use environment variables or defaults
PYTHON_BIN=${PYTHON_BIN:-python}

# Example paths - UPDATE THESE to match your setup
IMAGE_PATH="${IMAGE_PATH:-./input/input.png}"
OUTPUT_PATH="${OUTPUT_PATH:-./output/edited_output.png}"
DIT_MODEL="${DIT_MODEL:-/path/to/models/FramePackI2V_HY_bf16.safetensors}"
VAE_MODEL="${VAE_MODEL:-/path/to/models/pytorch_model.pt}"
TEXT_ENCODER1="${TEXT_ENCODER1:-/path/to/models/llava_llama3_fp16.safetensors}"
TEXT_ENCODER2="${TEXT_ENCODER2:-/path/to/models/clip_l.safetensors}"
IMAGE_ENCODER="${IMAGE_ENCODER:-/path/to/models/model.safetensors}"

$PYTHON_BIN "$SCRIPT_DIR/src/cli_inference.py" \
  --image_path "$IMAGE_PATH" \
  --prompt "the cat is playing with a black cat" \
  --negative_prompt "blurry, low quality" \
  --output_path "$OUTPUT_PATH" \
  --target_index 9 \
  --infer_steps 25 \
  --seed 1234 \
  --output_format png \
  --dtype bfloat16 \
  --attn_mode sdpa \
  --height 640 \
  --width 512 \
  --dit "$DIT_MODEL" \
  --vae "$VAE_MODEL" \
  --text_encoder1 "$TEXT_ENCODER1" \
  --text_encoder2 "$TEXT_ENCODER2" \
  --image_encoder "$IMAGE_ENCODER" \
  --verbose
