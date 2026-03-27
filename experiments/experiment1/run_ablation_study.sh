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

echo "=========================================="
echo "FramePack Ablation Study"
echo "=========================================="
echo "Multi-scale Controls in One-Frame Inference"
echo ""
echo "Dataset: COCO val2017 (20 sampled images)"
echo "Conditions: FULL, ABL-NO2X, ABL-NO4X, ABL-NONE"
echo "Quality Metrics: LPIPS, SSIM, CLIP Score"
echo "Speed Metrics: diffusion_time, peak_vram_gb, total_inference_time"
echo "=========================================="
echo ""

# Parse arguments
SKIP_DOWNLOAD=false
SKIP_SAMPLING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            echo "[INFO] Skipping COCO download"
            shift
            ;;
        --skip-sampling)
            SKIP_SAMPLING=true
            echo "[INFO] Skipping image sampling"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [--skip-download] [--skip-sampling]"
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

if [ "$SKIP_DOWNLOAD" = true ]; then
    CMD="$CMD --skip-download"
fi

if [ "$SKIP_SAMPLING" = true ]; then
    CMD="$CMD --skip-sampling"
fi

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
    echo "  Metrics: ablation_study/metrics.csv"
    echo "  Outputs: ablation_study/outputs/"
    echo "  Images: ablation_study/images/"
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
