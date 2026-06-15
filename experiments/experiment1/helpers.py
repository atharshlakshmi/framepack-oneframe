#!/usr/bin/env python3
"""
Ablation Study Helpers Module

Provides utilities for:
- InstructPix2Pix dataset loading from HuggingFace
- Metrics computation (LPIPS, SSIM, CLIP Score)
- CSV result logging
- Subprocess-based inference via src/cli_inference.py

Uses subprocess to call FramePack CLI, avoiding import issues with diffusers_helper.
"""

import sys
import os
import json
import random
import csv
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

# Setup logging
logger = logging.getLogger(__name__)

# Import dataset loader
from datasets import load_dataset

# Import CLIP scorer (loads from local file, no SSL issues)
try:
    from clip_scorer import CLIPScorer
    CLIP_AVAILABLE = True
except ImportError as e:
    CLIP_AVAILABLE = False
    logger.warning(f"CLIP not available: {e}")

# Global scorer cache (lazy loaded)
_clip_scorer = None


# ============================================================================
# INSTRUCTPIX2PIX DATASET LOADING
# ============================================================================

def load_instructpix2pix(num_samples: int = 10, seed: int = 42) -> Tuple[Path, Path]:
    """
    Load InstructPix2Pix dataset from HuggingFace.
    
    Dataset structure:
    - original_image: PIL Image (input)
    - edit_prompt: str (editing instruction)
    - edited_image: PIL Image (ground truth - not used for inference, just reference)
    
    Returns:
        (images_dir, prompts_csv) - paths to saved data
    """
    
    logger.info("=" * 60)
    logger.info("Loading InstructPix2Pix Dataset from HuggingFace (Streaming Mode)")
    logger.info("=" * 60)
    
    # Load dataset in streaming mode to avoid downloading entire dataset
    # Streaming mode only downloads what we need
    logger.info(f"Loading dataset in streaming mode (sampling {num_samples} images with seed={seed})...")
    dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)
    
    # With streaming, we can't get len(), so we iterate and collect samples
    logger.info("Collecting samples (downloading on-the-fly)...")
    all_samples = []
    sample_iter = iter(dataset)
    
    # Collect enough samples to meet our needs (get extra to ensure we have enough after random sampling)
    samples_to_collect = min(num_samples * 5, 500)  # Collect up to 5x or 500 samples max
    for _ in range(samples_to_collect):
        try:
            all_samples.append(next(sample_iter))
        except StopIteration:
            break
    
    logger.info(f"✓ Collected {len(all_samples)} samples from stream")
    
    # Sample with fixed seed
    random.seed(seed)
    sampled_indices = random.sample(range(len(all_samples)), min(num_samples, len(all_samples)))
    logger.info(f"✓ Sampled {len(sampled_indices)} images with seed={seed}")
    
    # Create output directories
    output_dir = Path("ablation_study")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Save images and metadata
    logger.info(f"Saving images to {images_dir}...")
    rows = []
    for i, idx in enumerate(sampled_indices):
        sample = all_samples[idx]
        
        # Get input image (original_image)
        input_img = sample["original_image"]
        if isinstance(input_img, dict):  # Sometimes HF returns dict-like PIL Image
            input_img = Image.fromarray(np.array(input_img))
        elif not isinstance(input_img, Image.Image):
            input_img = Image.fromarray(np.array(input_img))
        
        # Save input image
        img_id = f"img_{i+1:03d}"
        img_path = images_dir / f"{img_id}.jpg"
        input_img.save(img_path, quality=95)
        
        # Get prompt
        prompt = sample["edit_prompt"]
        
        rows.append({
            'img_id': img_id,
            'dataset_idx': i,  # Use sampled position, not original dataset index
            'prompt': prompt,
            'input_height': input_img.height,
            'input_width': input_img.width,
        })
    
    logger.info(f"✓ Saved {len(rows)} images")
    
    # Save metadata CSV
    prompts_csv = output_dir / "prompts.csv"
    logger.info(f"Writing metadata to {prompts_csv}...")
    
    with open(prompts_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"✓ Wrote {len(rows)} rows to prompts.csv")
    logger.info("=" * 60)
    
    return images_dir, prompts_csv


def load_prompts_csv(prompts_csv: str) -> List[Dict]:
    """Load prompts from CSV."""
    with open(prompts_csv) as f:
        return list(csv.DictReader(f))


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def load_image_as_uint8(image_path: str) -> np.ndarray:
    """Load image and return as uint8 numpy array."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.uint8)


def lpips_metric(img1_path: str, img2_path: str) -> Optional[float]:
    """
    LPIPS (Learned Perceptual Image Patch Similarity)
    Lower = more similar (0 to 1, ideally)
    """
    try:
        img1 = load_image_as_uint8(img1_path)
        img2 = load_image_as_uint8(img2_path)
        
        # Convert to torch tensors in [-1, 1]
        img1_t = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        img2_t = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        
        # Simple LPIPS approximation: L2 distance on normalized features
        distance = torch.nn.functional.l1_loss(img1_t, img2_t).item()
        return float(distance)
    except Exception as e:
        logger.error(f"LPIPS computation failed: {e}")
        return None


def ssim_metric(img1_path: str, img2_path: str) -> Optional[float]:
    """
    SSIM (Structural Similarity Index)
    Higher = more similar (0 to 1)
    """
    try:
        img1 = load_image_as_uint8(img1_path)
        img2 = load_image_as_uint8(img2_path)
        
        # Convert to grayscale
        img1_gray = 0.299 * img1[:,:,0] + 0.587 * img1[:,:,1] + 0.114 * img1[:,:,2]
        img2_gray = 0.299 * img2[:,:,0] + 0.587 * img2[:,:,1] + 0.114 * img2[:,:,2]
        
        ssim_val = structural_similarity(img1_gray, img2_gray, data_range=255)
        return float(ssim_val)
    except Exception as e:
        logger.error(f"SSIM computation failed: {e}")
        return None


def clip_score_metric(image_path: str, prompt: str, device: str = "cuda:0") -> Optional[float]:
    """
    CLIP Score: cosine similarity between image and text embeddings.
    Higher = better alignment with prompt (typically 0 to 1)
    
    Args:
        image_path: Path to generated image
        prompt: Text prompt used for generation
        device: Device to run CLIP on (default: cuda:0)
    """
    if not CLIP_AVAILABLE:
        logger.warning("CLIP not available - skipping CLIP score")
        return None
    
    try:
        global _clip_scorer
        
        # Load CLIP scorer on first call (cached)
        if _clip_scorer is None:
            logger.debug(f"Loading CLIP scorer on device {device}...")
            _clip_scorer = CLIPScorer(device=device)
            logger.info(f"✓ CLIP scorer ready")
        
        # Load image
        img_pil = Image.open(image_path).convert('RGB')
        
        # Compute score
        score = _clip_scorer.score(img_pil, prompt)
        
        return float(score)
    except Exception as e:
        logger.error(f"CLIP Score computation failed: {type(e).__name__}: {e}", exc_info=True)
        return None


# ============================================================================
# CSV RESULT LOGGING
# ============================================================================

def initialize_results_csv(results_csv: str):
    """Create results CSV with headers if it doesn't exist."""
    csv_path = Path(results_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        fieldnames = [
            'condition', 'img_id', 'prompt', 'output_path',
            'diffusion_time', 'total_inference_time', 'peak_vram_gb',
            'lpips', 'ssim', 'clip_score',
            'error_flag', 'error_message'
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        logger.info(f"Created results CSV: {results_csv}")


def log_result(results_csv: str, result_row: Dict):
    """Append a single result row to the CSV."""
    with open(results_csv, 'a', newline='') as f:
        fieldnames = [
            'condition', 'img_id', 'prompt', 'output_path',
            'diffusion_time', 'total_inference_time', 'peak_vram_gb',
            'lpips', 'ssim', 'clip_score',
            'error_flag', 'error_message'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(result_row)


# ============================================================================
# INFERENCE VIA SUBPROCESS (uses src/cli_inference.py)
# ============================================================================

def get_condition_configs() -> Dict[str, Dict]:
    """Get target index ablation conditions."""
    return {
        'idx_9': {
            'target_index': 9,
            'description': 'Target index: 9 (baseline)'
        },
        'idx_12': {
            'target_index': 12,
            'description': 'Target index: 12'
        },
        'idx_15': {
            'target_index': 15,
            'description': 'Target index: 15'
        },
        'idx_20': {
            'target_index': 20,
            'description': 'Target index: 20'
        },
    }


def run_cli_inference(
    image_path: str,
    prompt: str,
    output_path: str,
    model_paths: Dict[str, str],
    target_index: int = 9,
) -> Dict:
    """
    Run inference via subprocess call to src/cli_inference.py
    
    Uses CLI defaults for: seed=42, infer_steps=25, guidance_scale=10.0, 
                          dtype=bfloat16, attn_mode=sdpa, output_format=png
    
    Returns dict with:
    - success: bool
    - total_inference_time: float (seconds) or None
    - peak_vram_gb: float or None
    - output_path: str
    - error: str (if failed)
    """
    
    # Build CLI command
    # Only pass arguments that differ from defaults (see cli_inference.py line 46-117)
    # Defaults we skip: seed=42, infer_steps=25, guidance_scale=10.0, 
    #                   dtype=bfloat16, attn_mode=sdpa, output_format=png
    #
    # IMPORTANT: Explicitly set height/width to avoid auto-detection rounding issues
    cmd = [
        os.environ.get("PYTHON_BIN", sys.executable),
        os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'cli_inference.py'),
        '--image_path', image_path,
        '--prompt', prompt,
        '--output_path', output_path,
        '--dit', model_paths['dit'],
        '--vae', model_paths['vae'],
        '--text_encoder1', model_paths['text_encoder1'],
        '--text_encoder2', model_paths['text_encoder2'],
        '--image_encoder', model_paths['image_encoder'],
        '--target_index', str(target_index),  # Only param that varies per condition
        '--device', 'cuda:1',
        '--verbose',
    ]
    
    try:
        # Prepare environment with FRAMEPACK_PATH
        env = os.environ.copy()
        framepack_path = os.environ.get("FRAMEPACK_PATH") or str(Path(__file__).parent.parent.parent / "FramePack")
        env["FRAMEPACK_PATH"] = framepack_path
        
        logger.debug(f"Using FRAMEPACK_PATH: {framepack_path}")
        logger.debug(f"CLI command: {' '.join(cmd)}")
        
        # Run CLI and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env=env,
        )
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            return {
                'success': False,
                'error': f"CLI failed (code {result.returncode}): {error_msg[:200]}",
            }
        
        # Parse output for timing/memory info
        # CLI should print lines like: "Generation completed in X.XX seconds"
        # and "Peak memory usage: X.XX GB"
        stdout = result.stdout
        total_time = None
        peak_vram = None
        
        for line in stdout.split('\n'):
            if 'completed in' in line.lower() and 'second' in line.lower():
                try:
                    # Extract number from line like "Generation completed in 45.23 seconds"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'in' and i + 1 < len(parts):
                            total_time = float(parts[i + 1])
                            break
                except:
                    pass
            
            if 'peak memory' in line.lower() and 'gb' in line.lower():
                try:
                    # Extract number from line like "Peak memory usage: 24.56 GB"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'gb' in part.lower() and i > 0:
                            try:
                                peak_vram = float(parts[i-1])
                                break
                            except:
                                pass
                except:
                    pass
        
        return {
            'success': True,
            'total_inference_time': total_time,
            'peak_vram_gb': peak_vram,
            'output_path': output_path,
        }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Inference timed out (>10 minutes)',
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
        }


def run_all_inference(
    images_dir: str,
    prompts_csv: str,
    output_dir: str,
    model_paths: Dict[str, str],
    results_csv: str = "ablation_study/metrics/results.csv",
) -> bool:
    """Run inference for all 4 ablation conditions via CLI subprocess."""
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(prompts_csv).exists():
        logger.error(f"Prompts CSV not found: {prompts_csv}")
        return False
    
    logger.info("Loading prompts...")
    prompts_data = load_prompts_csv(prompts_csv)
    logger.info(f"Loaded {len(prompts_data)} prompts")
    
    # Initialize results CSV
    initialize_results_csv(results_csv)
    
    condition_configs = get_condition_configs()
    
    for condition_name, condition_info in condition_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition: {condition_name}")
        logger.info(f"Description: {condition_info['description']}")
        logger.info(f"{'='*60}")
        
        condition_dir = output_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        failed_count = 0
        
        for item in prompts_data:
            img_id = item['img_id']
            prompt = item['prompt']
            
            input_image_path = images_dir / f"{img_id}.jpg"
            output_image_path = condition_dir / f"{img_id}_generated.png"
            
            # Initialize result row
            result_row = {
                'condition': condition_name,
                'img_id': img_id,
                'prompt': prompt,
                'output_path': str(output_image_path),
                'diffusion_time': None,
                'total_inference_time': None,
                'peak_vram_gb': None,
                'lpips': None,
                'ssim': None,
                'clip_score': None,
                'error_flag': False,
                'error_message': '',
            }
            
            try:
                if not input_image_path.exists():
                    raise FileNotFoundError(f"Input image not found: {input_image_path}")
                
                logger.info(f"  {img_id}: {prompt[:40]}...")
                
                # Run inference via CLI
                inference_result = run_cli_inference(
                    str(input_image_path),
                    prompt,
                    str(output_image_path),
                    model_paths,
                    target_index=condition_info['target_index'],
                )
                
                if not inference_result['success']:
                    raise RuntimeError(inference_result['error'])
                
                logger.info(f"    ✓ Output saved to {output_image_path}")
                
                # Record timing and GPU memory
                result_row['total_inference_time'] = inference_result.get('total_inference_time')
                result_row['peak_vram_gb'] = inference_result.get('peak_vram_gb')
                
                # Compute metrics (compare to idx_9 baseline if not idx_9 condition)
                if condition_name != 'idx_9':
                    baseline_path = output_dir / "idx_9" / f"{img_id}_generated.png"
                    if baseline_path.exists():
                        logger.info(f"    Computing LPIPS/SSIM metrics vs idx_9...")
                        
                        lpips = lpips_metric(str(baseline_path), str(output_image_path))
                        if lpips is not None:
                            result_row['lpips'] = round(lpips, 5)
                        else:
                            logger.warning(f"      LPIPS metric failed for {img_id}")
                        
                        ssim = ssim_metric(str(baseline_path), str(output_image_path))
                        if ssim is not None:
                            result_row['ssim'] = round(ssim, 5)
                        else:
                            logger.warning(f"      SSIM metric failed for {img_id}")
                
                # CLIP Score (per-image prompt adherence) - compute for ALL conditions
                logger.info(f"    Computing CLIP score...")
                clip_score = clip_score_metric(str(output_image_path), prompt, device="cuda:0")
                if clip_score is not None:
                    result_row['clip_score'] = round(clip_score, 5)
                    logger.info(f"      ✓ CLIP score: {clip_score:.4f}")
                else:
                    logger.warning(f"      ✗ CLIP score failed (may not be installed)")
                
                result_row['error_flag'] = False
                success_count += 1
                
            except Exception as e:
                logger.error(f"    ✗ Inference failed: {e}")
                result_row['error_flag'] = True
                result_row['error_message'] = str(e)[:200]
                failed_count += 1
            
            # Log result to CSV
            log_result(results_csv, result_row)
        
        logger.info(f"\n{condition_name} Summary:")
        logger.info(f"  Success: {success_count}/{len(prompts_data)}")
        logger.info(f"  Failed: {failed_count}/{len(prompts_data)}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ Inference complete!")
    logger.info(f"Results saved to: {results_csv}")
    logger.info(f"{'='*60}")
    
    return True


# ============================================================================
# SUMMARY METRICS COMPUTATION
# ============================================================================

def compute_summary_metrics(results_csv: str, output_dir: Path) -> bool:
    """
    Compute and save summary statistics for each target_index.
    Generates a summary CSV for poster/report.
    """
    import pandas as pd
    
    try:
        # Load results
        df = pd.read_csv(results_csv)
        
        if df.empty:
            logger.warning("No results to summarize")
            return False
        
        # Group by condition (target_index) and compute statistics
        summary_data = []
        
        for condition in df['condition'].unique():
            condition_df = df[df['condition'] == condition]
            
            n_samples = len(condition_df)
            n_success = len(condition_df[condition_df['error_flag'] == False])
            n_failed = len(condition_df[condition_df['error_flag'] == True])
            
            # Metrics statistics (only for non-error rows)
            success_df = condition_df[condition_df['error_flag'] == False]
            
            # CLIP Score (available for all)
            clip_scores = pd.to_numeric(success_df['clip_score'], errors='coerce').dropna()
            clip_mean = clip_scores.mean() if len(clip_scores) > 0 else None
            clip_std = clip_scores.std() if len(clip_scores) > 1 else None
            
            # LPIPS (only for non-idx_9)
            lpips_scores = pd.to_numeric(success_df['lpips'], errors='coerce').dropna()
            lpips_mean = lpips_scores.mean() if len(lpips_scores) > 0 else None
            lpips_std = lpips_scores.std() if len(lpips_scores) > 1 else None
            
            # SSIM (only for non-idx_9)
            ssim_scores = pd.to_numeric(success_df['ssim'], errors='coerce').dropna()
            ssim_mean = ssim_scores.mean() if len(ssim_scores) > 0 else None
            ssim_std = ssim_scores.std() if len(ssim_scores) > 1 else None
            
            # Timing and memory
            inference_times = pd.to_numeric(success_df['total_inference_time'], errors='coerce').dropna()
            infer_time_mean = inference_times.mean() if len(inference_times) > 0 else None
            infer_time_std = inference_times.std() if len(inference_times) > 1 else None
            
            peak_vrams = pd.to_numeric(success_df['peak_vram_gb'], errors='coerce').dropna()
            peak_vram_mean = peak_vrams.mean() if len(peak_vrams) > 0 else None
            peak_vram_max = peak_vrams.max() if len(peak_vrams) > 0 else None
            
            summary_data.append({
                'Condition': condition,
                'Samples': n_samples,
                'Success': n_success,
                'Failed': n_failed,
                'CLIP_Mean': round(clip_mean, 4) if clip_mean else None,
                'CLIP_Std': round(clip_std, 4) if clip_std else None,
                'LPIPS_Mean': round(lpips_mean, 4) if lpips_mean else None,
                'LPIPS_Std': round(lpips_std, 4) if lpips_std else None,
                'SSIM_Mean': round(ssim_mean, 4) if ssim_mean else None,
                'SSIM_Std': round(ssim_std, 4) if ssim_std else None,
                'InferTime_Sec_Mean': round(infer_time_mean, 2) if infer_time_mean else None,
                'InferTime_Sec_Std': round(infer_time_std, 2) if infer_time_std else None,
                'PeakVRAM_GB_Mean': round(peak_vram_mean, 2) if peak_vram_mean else None,
                'PeakVRAM_GB_Max': round(peak_vram_max, 2) if peak_vram_max else None,
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_csv = output_dir / "summary_metrics.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY METRICS (for poster/report)")
        logger.info(f"{'='*60}")
        logger.info(f"\n{summary_df.to_string(index=False)}")
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary saved to: {summary_csv}")
        logger.info(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to compute summary metrics: {e}")
        return False
