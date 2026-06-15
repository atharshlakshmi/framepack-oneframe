#!/bin/bash
# Qualitative Study Script for FramePack One-Frame Inference
# 
# This script iterates through images and prompts from a CSV file
# and runs inference on each pair
#
# Usage: bash run_qualitative_study.sh <images_dir> <prompts_csv> [output_dir]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env file if it exists (project root)
PROJECT_ROOT="$SCRIPT_DIR/../.."
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment from .env file..."
    set -a  # Export all variables
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "⚠️  Warning: .env file not found at $PROJECT_ROOT/.env"
fi

# ============================================================================
# CONFIGURATION
# ============================================================================

# Command-line arguments
IMAGES_DIR="${1:-./images}"
PROMPTS_CSV="${2:-prompts.csv}"
OUTPUT_DIR="${3:-./output/qualitative_study}"

# Use environment variables or defaults
PYTHON_BIN=${PYTHON_BIN:-python}

# Optimize CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set Python path to include src directory and FramePack
PROJECT_ROOT="$SCRIPT_DIR/../.."
export PYTHONPATH="$PROJECT_ROOT/src:$FRAMEPACK_PATH:$PYTHONPATH"

if [ -z "$FRAMEPACK_PATH" ]; then
    echo "❌ Error: FRAMEPACK_PATH not set. Please set it in .env or export manually."
    echo "Example: export FRAMEPACK_PATH=/path/to/FramePack"
    exit 1
fi

if [ ! -d "$FRAMEPACK_PATH" ]; then
    echo "❌ Error: FRAMEPACK_PATH directory not found: $FRAMEPACK_PATH"
    exit 1
fi

# Model paths from environment or defaults
DIT_MODEL="${DIT_MODEL:-/path/to/models/FramePackI2V_HY_bf16.safetensors}"
VAE_MODEL="${VAE_MODEL:-/path/to/models/pytorch_model.pt}"
TEXT_ENCODER1="${TEXT_ENCODER1:-/path/to/models/llava_llama3_fp16.safetensors}"
TEXT_ENCODER2="${TEXT_ENCODER2:-/path/to/models/clip_l.safetensors}"
IMAGE_ENCODER="${IMAGE_ENCODER:-/path/to/models/model.safetensors}"

# Inference parameters (customize as needed)
INFER_STEPS=${INFER_STEPS:-25}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-10.0}
TARGET_INDEX=${TARGET_INDEX:-12}
DTYPE=${DTYPE:-bfloat16}
ATTN_MODE=${ATTN_MODE:-sdpa}
DEVICE=${DEVICE:-cuda:0}
SEED_BASE=${SEED_BASE:-42}

# ============================================================================
# VALIDATION
# ============================================================================

# Check if images directory exists
if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

# Check if CSV file exists
if [ ! -f "$PROMPTS_CSV" ]; then
    echo "❌ Error: Prompts CSV not found: $PROMPTS_CSV"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create results CSV header
RESULTS_CSV="$OUTPUT_DIR/results.csv"
if [ ! -f "$RESULTS_CSV" ]; then
    echo "image_filename,prompt,output_filename,seed,infer_steps,guidance_scale,status,timestamp" > "$RESULTS_CSV"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         FramePack Qualitative Study - Batch Inference          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo "Images directory:    $IMAGES_DIR"
echo "Prompts CSV:         $PROMPTS_CSV"
echo "Output directory:    $OUTPUT_DIR"
echo "Results CSV:         $RESULTS_CSV"
echo "────────────────────────────────────────────────────────────────"
echo ""

# ============================================================================
# PROCESS CSV AND RUN INFERENCE
# ============================================================================

# Counter for progress tracking
TOTAL_LINES=$(wc -l < "$PROMPTS_CSV")
TOTAL_SAMPLES=$((TOTAL_LINES - 1))  # Subtract header line
CURRENT_INDEX=0
SUCCESSFUL=0
FAILED=0
SKIPPED=0

# Read CSV line by line (skip header)
tail -n +2 "$PROMPTS_CSV" | while IFS=',' read -r image_filename prompt seed; do
    CURRENT_INDEX=$((CURRENT_INDEX + 1))
    
    # Skip empty lines
    if [ -z "$image_filename" ]; then
        continue
    fi
    
    # Trim whitespace
    image_filename=$(echo "$image_filename" | xargs)
    prompt=$(echo "$prompt" | xargs)
    seed=$(echo "$seed" | xargs)
    
    # Use default seed if not specified
    if [ -z "$seed" ]; then
        seed=$((SEED_BASE + CURRENT_INDEX))
    fi
    
    # Find image file (support multiple extensions)
    IMAGE_PATH=$(find "$IMAGES_DIR" -maxdepth 1 -iname "$image_filename" 2>/dev/null | head -1)
    
    if [ -z "$IMAGE_PATH" ] || [ ! -f "$IMAGE_PATH" ]; then
        echo "⊘ [$CURRENT_INDEX/$TOTAL_SAMPLES] SKIPPED: Image not found - $image_filename"
        echo "$image_filename,$prompt,,,$INFER_STEPS,$GUIDANCE_SCALE,SKIPPED,$(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_CSV"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Generate output filename
    BASENAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
    EXTENSION="${IMAGE_PATH##*.}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILENAME="${BASENAME}_${TIMESTAMP}_seed${seed}.png"
    OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILENAME"
    
    # Progress indicator
    echo "→ [$CURRENT_INDEX/$TOTAL_SAMPLES] Processing: $image_filename"
    echo "  📷 Input:  $IMAGE_PATH"
    echo "  ✍️  Prompt: $prompt"
    echo "  🔹 Seed: $seed | Steps: $INFER_STEPS | Guidance: $GUIDANCE_SCALE"
    echo ""
    
    # Run inference
    START_TIME=$(date +%s%N)
    
    if $PYTHON_BIN "$SCRIPT_DIR/../../src/cli_inference.py" \
        --image_path "$IMAGE_PATH" \
        --output_path "$OUTPUT_PATH" \
        --prompt "$prompt" \
        --target_index "$TARGET_INDEX" \
        --infer_steps "$INFER_STEPS" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --seed "$seed" \
        --output_format png \
        --dtype "$DTYPE" \
        --attn_mode "$ATTN_MODE" \
        --dit "$DIT_MODEL" \
        --vae "$VAE_MODEL" \
        --text_encoder1 "$TEXT_ENCODER1" \
        --text_encoder2 "$TEXT_ENCODER2" \
        --image_encoder "$IMAGE_ENCODER" \
        --device "$DEVICE" \
        --verbose 2>&1 | tail -20; then
        
        END_TIME=$(date +%s%N)
        DURATION=$(( (END_TIME - START_TIME) / 1000000000 ))
        
        echo "✅ SUCCESS - Time: ${DURATION}s - Output: $OUTPUT_FILENAME"
        echo "$image_filename,$prompt,$OUTPUT_FILENAME,$seed,$INFER_STEPS,$GUIDANCE_SCALE,SUCCESS,$(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_CSV"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "❌ FAILED - Inference error for: $image_filename"
        echo "$image_filename,$prompt,,,$INFER_STEPS,$GUIDANCE_SCALE,FAILED,$(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_CSV"
        FAILED=$((FAILED + 1))
    fi
    
    echo "────────────────────────────────────────────────────────────"
    echo ""
    
done

# Print summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    STUDY COMPLETED                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo "Total samples:       $TOTAL_SAMPLES"
echo "✅ Successful:       $SUCCESSFUL"
echo "❌ Failed:           $FAILED"
echo "⊘ Skipped:            $SKIPPED"
echo ""
echo "Results saved to:    $RESULTS_CSV"
echo "Outputs saved to:    $OUTPUT_DIR"
echo ""
