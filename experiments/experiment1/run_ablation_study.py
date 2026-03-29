#!/usr/bin/env python3
"""
Main orchestration script for ablation study

Runs all steps in sequence:
1. Download COCO val2017 dataset (if needed)
2. Sample 20 images with fixed seed
3. Run inference for all 4 conditions
4. Compute metrics (LPIPS, SSIM, CLIP Score)
5. Generate report

Usage:
    python run_ablation_study.py [--skip-download] [--skip-sampling] [--force]
    
    --skip-download : Skip COCO download (use existing data)
    --skip-sampling : Skip image sampling (use existing samples)
    --force         : Overwrite existing outputs
"""

import argparse
import sys
import logging
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from helpers import (
    load_instructpix2pix,
    run_all_inference,
    compute_summary_metrics,
    load_prompts_csv,
)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try parent directory
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

logger = logging.getLogger(__name__)


def find_model_paths() -> Dict[str, str]:
    """Try to find FramePack models in common locations."""
    
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return {}
    
    # Expected model files
    model_map = {
        'dit': 'FramePack_F1_I2V_HY_20250503.safetensors',
        'vae': 'pytorch_model.pt',
        'text_encoder1': 'llava_llama3_fp16.safetensors',
        'text_encoder2': 'clip_l.safetensors',
        'image_encoder': 'model.safetensors',
    }
    
    model_paths = {}
    for key, filename in model_map.items():
        model_path = models_dir / filename
        if model_path.exists():
            model_paths[key] = str(model_path)
            logger.info(f"✓ Found {key}: {filename}")
        else:
            logger.warning(f"✗ Missing {key}: {filename}")
    
    return model_paths if len(model_paths) == len(model_map) else {}


def run_step(description: str, func, *args, **kwargs) -> bool:
    """Execute a step and return success status."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Step: {description}")
    logger.info(f"{'='*60}")
    
    try:
        result = func(*args, **kwargs)
        if result:
            logger.info(f"✓ {description} complete")
            return True
        else:
            logger.error(f"✗ {description} failed")
            return False
    except Exception as e:
        logger.error(f"✗ {description} failed with exception: {e}")
        return False




def check_prerequisites() -> bool:
    """Check that required files exist."""
    logger.info("Checking prerequisites...")
    
    required_files = [
        'helpers.py',
    ]
    
    for fname in required_files:
        fpath = Path(fname)
        if not fpath.exists():
            logger.error(f"✗ Missing: {fname}")
            return False
        logger.info(f"✓ {fname}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run
  python run_ablation_study.py
        """
    )
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("FramePack Ablation Study")
    logger.info("=" * 60)
    logger.info("Target Frame Index Optimization")
    logger.info("")
    logger.info("Study Protocol:")
    logger.info("  Dataset: InstructPix2Pix (20 paired image-instruction samples)")
    logger.info("  Conditions: idx_9, idx_12, idx_15, idx_20")
    logger.info("  Quality Metrics: CLIP Score, SSIM, LPIPS")
    logger.info("  Performance: total_inference_time, peak_vram_gb")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed")
        return 1
    
    logger.info("✓ All files present\n")
    
    # Step 1: Load InstructPix2Pix Dataset
    if not run_step("Load InstructPix2Pix dataset", load_instructpix2pix):
        return 1
    
    # Step 2: Run inference (requires models)
    logger.info("\nDetecting FramePack models...")
    model_paths = find_model_paths()
    
    if not model_paths:
        logger.error("✗ Models not found. All 5 required models must be present:")
        logger.error("  - FramePack_F1_I2V_HY_20250503.safetensors (dit)")
        logger.error("  - pytorch_model.pt (vae)")
        logger.error("  - llava_llama3_fp16.safetensors (text_encoder1)")
        logger.error("  - clip_l.safetensors (text_encoder2)")
        logger.error("  - model.safetensors (image_encoder)")
        logger.error(f"  Expected location: ../models/")
        return 1
    
    logger.info(f"✓ Found all {len(model_paths)}/5 required models")
    if not run_step("Run inference for all index conditions", 
                    run_all_inference,
                    "ablation_study/images",
                    "ablation_study/prompts.csv",
                    "ablation_study",
                    model_paths):
        return 1
    
    # Step 3: Compute metrics
    logger.info(f"\n{'='*60}")
    logger.info("Step: Compute ablation metrics")
    logger.info(f"{'='*60}")
    logger.info("Step: Metrics computation")
    logger.info(f"{'='*60}")
    logger.info("✓ Metrics computed and saved during inference")
    logger.info("✓ Results available in: ablation_study/metrics/results.csv")
    
    # Step 4: Generate summary metrics for poster/report
    logger.info("\nGenerating summary metrics for poster/report...")
    results_csv = Path("ablation_study") / "metrics" / "results.csv"
    if not compute_summary_metrics(str(results_csv), Path("ablation_study")):
        logger.error("Warning: Summary metrics generation failed (results still valid)")
    
    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("✓ ABLATION STUDY COMPLETE")
    logger.info("=" * 60)
    logger.info("\nOutput Structure:")
    logger.info("  ablation_study/")
    logger.info("    ├── images/                 # 20 sampled COCO images (seed=42)")
    logger.info("    ├── prompts.csv             # Image metadata and captions")
    logger.info("    ├── idx_9/                  # Generated images (target_index=9)")
    logger.info("    ├── idx_12/                 # Generated images (target_index=12)")
    logger.info("    ├── idx_15/                 # Generated images (target_index=15)")
    logger.info("    ├── idx_20/                 # Generated images (target_index=20)")
    logger.info("    ├── metrics/")
    logger.info("    │   ├── results.csv         # Per-image metrics table")
    logger.info("    │   └── summary_metrics.csv # Statistics per condition (for poster)")
    logger.info("    └── ")
    logger.info("\nResults CSV Columns (results.csv):")
    logger.info("  - condition: idx_9, idx_12, idx_15, idx_20")
    logger.info("  - img_id, prompt, output_path")
    logger.info("  - total_inference_time, peak_vram_gb")
    logger.info("  - clip_score (higher=better), ssim, lpips (vs idx_9 baseline)")
    logger.info("  - error_flag, error_message")
    logger.info("\nSummary Metrics (summary_metrics.csv - for poster):")
    logger.info("  - CLIP_Mean/Std (per condition)")
    logger.info("  - SSIM_Mean/Std (per condition vs idx_9 baseline)")
    logger.info("  - LPIPS_Mean/Std (per condition vs idx_9 baseline)")
    logger.info("  - InferTime_Sec_Mean/Std (per condition)")
    logger.info("  - PeakVRAM_GB_Mean/Max (per condition)")
    logger.info("\nNext steps:")
    logger.info("  1. Open ablation_study/metrics/results.csv to review results")
    logger.info("  2. Compare CLIP scores across indices to find best prompt adherence")
    logger.info("  3. Analyze visual outputs in ablation_study/{idx_9,idx_12,idx_15,idx_20}/")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
