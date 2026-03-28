#!/bin/bash
#
# Ablation Study Runner
#
# All functionality consolidated in helpers.py
# Main orchestration in run_ablation_study.py
#
# Usage:
#   ./run_ablation_study.sh                # Full run
#   ./run_ablation_study.sh --skip-download  # Skip download
#   ./run_ablation_study.sh --skip-sampling  # Skip download and sampling
#

set -e  # Exit on any error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Load .env file from repository root if it exists
ENV_FILE="../../.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE..."
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "[WARNING] .env file not found at $ENV_FILE"
    echo "[INFO] Attempting to activate torch environment..."
    eval "$(conda shell.bash hook)"
    conda activate torch
fi

echo "=========================================="
echo "FramePack Ablation Study"
echo "=========================================="
echo "Target Frame Index Optimization"
echo ""
echo "Dataset: InstructPix2Pix (20 paired image-instruction samples)"
echo "Conditions: idx_9, idx_12, idx_15, idx_20"
echo "Quality Metrics: CLIP Score, SSIM, LPIPS"
echo "Performance: total_inference_time, peak_vram_gb"
echo "=========================================="
echo ""

# Parse arguments (legacy - kept for compatibility)
SKIP_DOWNLOAD=false
SKIP_SAMPLING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download|--skip-sampling)
            echo "[INFO] Argument ignored (using HF InstructPix2Pix dataset)"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0"
            exit 1
            ;;
    esac
done

echo ""

# Check if in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "[WARNING] Not in a conda environment"
    echo "[INFO] Recommended: conda activate torch (or your env)"
fi

# Check for Python (use active python/pip from environment)
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python not found. Please activate conda environment or install Python 3.8+"
    exit 1
fi

echo "Python version:"
python --version
echo ""

# Install dependencies (use active pip from environment)
echo "=========================================="
echo "Installing dependencies..."
echo "=========================================="
pip install --upgrade pip 2>/dev/null || true
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Build command
CMD="python run_ablation_study.py"

# Run ablation study
echo "=========================================="
echo "Running ablation study..."
echo "=========================================="
echo "Command: $CMD"
echo ""

$CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ ABLATION STUDY COMPLETE"
    echo "=========================================="
    echo ""
    echo "Results:"
    echo "  Metrics: ablation_study/metrics/results.csv"
    echo "  Generated: ablation_study/{idx_9,idx_12,idx_15,idx_20}/"
    echo "  Dataset: ablation_study/images/ (20 InstructPix2Pix samples)"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ ABLATION STUDY FAILED"
    echo "=========================================="
    echo "Exit code: $exit_code"
    echo ""
fi

exit $exit_code
