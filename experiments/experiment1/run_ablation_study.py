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
from pathlib import Path
from typing import List

from helpers import (
    download_coco,
    sample_images,
    run_all_inference,
    compute_all_metrics,
    load_prompts_csv,
)

logger = logging.getLogger(__name__)


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
  # Full run with all steps
  python run_ablation_study.py
  
  # Skip download if dataset already exists
  python run_ablation_study.py --skip-download
  
  # Skip both download and sampling
  python run_ablation_study.py --skip-download --skip-sampling
        """
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip COCO dataset download (use existing coco_data/)'
    )
    parser.add_argument(
        '--skip-sampling',
        action='store_true',
        help='Skip image sampling (use existing ablation_study/images/)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 60)
    logger.info("FramePack Ablation Study")
    logger.info("=" * 60)
    logger.info("Multi-Scale Controls in One-Frame Inference")
    logger.info("")
    logger.info("Study Protocol:")
    logger.info("  Dataset: COCO val2017 (20 sampled images)")
    logger.info("  Conditions: FULL, ABL-NO2X, ABL-NO4X, ABL-NONE")
    logger.info("  Quality Metrics: LPIPS, SSIM, CLIP Score")
    logger.info("  Speed Metrics: diffusion_time, peak_vram_gb, total_inference_time")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed")
        return 1
    
    logger.info("✓ All files present\n")
    
    # Step 1: Download COCO
    if not args.skip_download:
        if not run_step("Download COCO val2017 dataset", download_coco):
            return 1
    else:
        logger.info("\n[SKIPPED] COCO download (using existing coco_data/)")
    
    # Step 2: Sample images
    if not args.skip_sampling:
        if not run_step("Sample 20 images", sample_images):
            return 1
    else:
        logger.info("\n[SKIPPED] Image sampling (using existing ablation_study/)")
    
    # Step 3: Run inference
    if not run_step("Run inference for all conditions", 
                    run_all_inference,
                    "ablation_study/images",
                    "ablation_study/prompts.csv",
                    "ablation_study/outputs"):
        return 1
    
    # Step 4: Compute metrics
    logger.info(f"\n{'='*60}")
    logger.info("Step: Compute ablation metrics")
    logger.info(f"{'='*60}")
    
    outputs_base_dir = Path("ablation_study/outputs")
    conditions = ['ABL-NO2X', 'ABL-NO4X', 'ABL-NONE']
    full_dir = outputs_base_dir / 'FULL'
    
    all_results = []
    
    for condition in conditions:
        abl_dir = outputs_base_dir / condition
        metrics_results = compute_all_metrics(
            str(full_dir),
            str(abl_dir),
            condition,
        )
        all_results.extend(metrics_results)
    
    # Save results to CSV
    if all_results:
        import csv
        
        all_keys = set()
        for result in all_results:
            all_keys.update(result.keys())
        
        fieldnames = ['condition', 'img_id'] + sorted([k for k in all_keys if k not in ['condition', 'img_id']])
        
        metrics_csv = Path("ablation_study/metrics.csv")
        logger.info(f"\nSaving metrics to {metrics_csv}...")
        
        with open(metrics_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        logger.info(f"✓ Wrote {len(all_results)} rows")
        logger.info(f"{'='*60}")
        
        # Print summary statistics
        logger.info("\nMetrics Summary:")
        for condition in conditions:
            cond_results = [r for r in all_results if r['condition'] == condition]
            if cond_results:
                logger.info(f"\n{condition} (n={len(cond_results)}):")
                metric_names = [k for k in fieldnames if k not in ['condition', 'img_id']]
                for metric in metric_names:
                    values = [r[metric] for r in cond_results if metric in r]
                    if values:
                        avg = sum(values) / len(values)
                        logger.info(f"  {metric:20s}: {avg:.4f}")
    else:
        logger.error("No metrics computed")
        return 1
    
    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("✓ ABLATION STUDY COMPLETE")
    logger.info("=" * 60)
    logger.info("\nOutput Structure:")
    logger.info("  ablation_study/")
    logger.info("    ├── images/              # 20 sampled COCO images")
    logger.info("    ├── prompts.csv          # Image metadata and captions")
    logger.info("    ├── outputs/")
    logger.info("    │   ├── FULL/            # Full pipeline outputs")
    logger.info("    │   ├── ABL-NO2X/        # 2x controls disabled")
    logger.info("    │   ├── ABL-NO4X/        # 4x controls disabled")
    logger.info("    │   ├── ABL-NONE/        # Both controls disabled")
    logger.info("    │   └── inference_results.json")
    logger.info("    └── metrics.csv          # Computed metrics")
    logger.info("\nNext steps:")
    logger.info("  1. Review metrics.csv for quality degradation")
    logger.info("  2. Analyze outputs in ablation_study/outputs/")
    logger.info("  3. Compare performance/speed tradeoffs")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
