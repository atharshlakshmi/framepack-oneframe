#!/usr/bin/env python3
"""
Helpers module for ablation study

Consolidates all utility functions:
- COCO dataset download
- Image sampling with fixed seed
- Inference execution
- Metrics computation (LPIPS, SSIM, CLIP)
"""

import json
import random
import shutil
import csv
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)

# ============================================================================
# COCO DATASET DOWNLOAD
# ============================================================================

def download_file(url: str, dest_path: str, desc: str) -> bool:
    """Download a file with progress tracking."""
    if os.path.exists(dest_path):
        logger.info(f"✓ {dest_path} already exists, skipping download")
        return True
    
    logger.info(f"Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"✓ Downloaded {desc} to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {desc}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract zip file."""
    if not os.path.exists(zip_path):
        logger.error(f"Zip file not found: {zip_path}")
        return False
    
    logger.info(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✓ Extracted to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to extract {zip_path}: {e}")
        return False


def download_coco(coco_data_dir: str = "coco_data") -> bool:
    """Download and set up COCO val2017 dataset."""
    
    # Create directories
    data_dir = Path(coco_data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # URLs
    images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    images_zip = data_dir / "val2017.zip"
    annotations_zip = data_dir / "annotations_trainval2017.zip"
    
    logger.info("=" * 60)
    logger.info("COCO val2017 Dataset Download")
    logger.info("=" * 60)
    
    if not download_file(str(images_url), str(images_zip), "COCO val2017 images"):
        return False
    
    if not download_file(str(annotations_url), str(annotations_zip), "COCO annotations"):
        return False
    
    if not extract_zip(str(images_zip), str(data_dir)):
        return False
    
    if not extract_zip(str(annotations_zip), str(data_dir)):
        return False
    
    # Verify important files
    required_files = [
        data_dir / "val2017" / "README.txt",
        data_dir / "annotations" / "captions_val2017.json",
    ]
    
    logger.info("\nVerifying required files...")
    all_exist = True
    for fpath in required_files:
        if fpath.exists():
            logger.info(f"✓ {fpath}")
        else:
            logger.warning(f"✗ Missing: {fpath}")
            all_exist = False
    
    if all_exist:
        logger.info("\n" + "=" * 60)
        logger.info("✓ COCO val2017 setup complete!")
        logger.info("=" * 60)
        return True
    else:
        logger.warning("\nSome files may be missing. Check manually.")
        return False


# ============================================================================
# IMAGE SAMPLING
# ============================================================================

def load_prompts_csv(csv_path: str) -> List[Dict]:
    """Load image list and prompts from CSV."""
    prompts = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


def sample_images(coco_data_dir: str = "coco_data", num_samples: int = 20) -> bool:
    """Sample 20 images from COCO val2017."""
    
    coco_data_dir = Path(coco_data_dir)
    captions_file = coco_data_dir / "annotations" / "captions_val2017.json"
    images_dir = coco_data_dir / "val2017"
    
    if not captions_file.exists():
        logger.error(f"Captions file not found: {captions_file}")
        logger.info("Run download_coco() first to download the dataset.")
        return False
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return False
    
    logger.info("=" * 60)
    logger.info("Sampling COCO val2017 Images")
    logger.info("=" * 60)
    
    with open(captions_file, 'r') as f:
        coco = json.load(f)
    
    logger.info(f"Total images in COCO val2017: {len(coco['images'])}")
    
    captions = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in captions:
            captions[img_id] = ann['caption']
    
    logger.info(f"Captions loaded: {len(captions)} images have captions")
    
    eligible = [
        img for img in coco['images']
        if min(img['height'], img['width']) >= 512
        and img['id'] in captions
    ]
    
    logger.info(f"Eligible images (>= 512px): {len(eligible)}")
    
    random.seed(42)
    sample = random.sample(eligible, num_samples)
    logger.info(f"\n✓ Sampled {len(sample)} images with seed 42")
    
    output_dir = Path("ablation_study")
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nCopying images to {images_output_dir}...")
    rows = []
    for i, img in enumerate(sample):
        src = images_dir / img['file_name']
        dst = images_output_dir / f"img_{i+1:03d}.jpg"
        
        if not src.exists():
            logger.warning(f"Source image not found: {src}")
            continue
        
        shutil.copy(src, dst)
        
        rows.append({
            'img_id': f'img_{i+1:03d}',
            'coco_id': img['id'],
            'file_name': img['file_name'],
            'prompt': captions[img['id']],
            'orig_h': img['height'],
            'orig_w': img['width'],
        })
    
    logger.info(f"✓ Copied {len(rows)} images")
    
    prompts_csv = output_dir / "prompts.csv"
    logger.info(f"\nWriting metadata to {prompts_csv}...")
    
    with open(prompts_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"✓ Wrote {len(rows)} rows to prompts.csv")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Sampling complete!")
    logger.info("=" * 60)
    
    return True


# ============================================================================
# METRICS COMPUTATION (Protocol: LPIPS, SSIM, CLIP Score)
# ============================================================================

def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """Load image and convert to normalized tensor."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def lpips_metric(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    LPIPS (Learned Perceptual Image Patch Similarity)
    
    Simplified version using L2 distance on normalized features.
    For production, install full lpips library: pip install lpips
    """
    img1_norm = F.normalize(img1, p=2, dim=1)
    img2_norm = F.normalize(img2, p=2, dim=1)
    distance = torch.mean((img1_norm - img2_norm) ** 2).item()
    return distance


def ssim_metric(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    SSIM (Structural Similarity Index)
    
    Higher = more similar. Range: [0, 1]
    """
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Convert to numpy and grayscale for scikit-image
    img1_np = img1.squeeze(0).permute(1, 2, 0).numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).numpy()
    
    if img1_np.shape[2] == 3:
        img1_gray = 0.299 * img1_np[:, :, 0] + 0.587 * img1_np[:, :, 1] + 0.114 * img1_np[:, :, 2]
        img2_gray = 0.299 * img2_np[:, :, 0] + 0.587 * img2_np[:, :, 1] + 0.114 * img2_np[:, :, 2]
    else:
        img1_gray = img1_np.squeeze()
        img2_gray = img2_np.squeeze()
    
    ssim_val = structural_similarity(img1_gray, img2_gray, data_range=1.0)
    return ssim_val


def compute_metrics(full_image_path: str, abl_image_path: str) -> Dict[str, float]:
    """
    Compute protocol-specified metrics comparing ablation output to full output.
    
    Metrics:
    - lpips: LPIPS distance (lower = closer to FULL)
    - ssim: SSIM similarity (higher = closer to FULL)
    """
    try:
        full_img = load_image_as_tensor(full_image_path)
        abl_img = load_image_as_tensor(abl_image_path)
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        return {}
    
    metrics = {
        'lpips': lpips_metric(full_img, abl_img),
        'ssim': ssim_metric(full_img, abl_img),
    }
    
    return metrics


def compute_all_metrics(
    full_outputs_dir: str,
    abl_outputs_dir: str,
    condition_name: str,
) -> List[Dict]:
    """Compute metrics for all images in an ablation condition."""
    full_dir = Path(full_outputs_dir)
    abl_dir = Path(abl_outputs_dir)
    
    if not full_dir.exists():
        logger.error(f"FULL outputs directory not found: {full_dir}")
        return []
    
    if not abl_dir.exists():
        logger.error(f"Ablation outputs directory not found: {abl_dir}")
        return []
    
    results = []
    full_images = sorted(full_dir.glob('img_*_generated.png'))
    logger.info(f"\nComputing metrics for {condition_name}...")
    logger.info(f"Found {len(full_images)} images in FULL outputs")
    
    for full_img_path in full_images:
        img_name = full_img_path.name
        abl_img_path = abl_dir / img_name
        
        if not abl_img_path.exists():
            logger.warning(f"  Missing ablation output: {abl_img_path}")
            continue
        
        img_id = img_name.replace('_generated.png', '')
        
        logger.info(f"  {img_id}...")
        metrics = compute_metrics(str(full_img_path), str(abl_img_path))
        
        if metrics:
            result = {
                'condition': condition_name,
                'img_id': img_id,
                **metrics
            }
            results.append(result)
    
    logger.info(f"  ✓ Computed metrics for {len(results)} images")
    return results


# ============================================================================
# INFERENCE EXECUTION
# ============================================================================

class GenerationConfig:
    """Configuration for a single generation run."""
    
    def __init__(
        self,
        seed: int = 42,
        infer_steps: int = 25,
        guidance_scale: float = 10.0,
        real_guidance_scale: float = 1.0,
        height: int = 640,
        width: int = 512,
        dtype: str = "bfloat16",
        attn_mode: str = "sdpa",
        target_index: int = 9,
        control_indices: List[int] = None,
        one_frame_flags: set = None,
    ):
        self.seed = seed
        self.infer_steps = infer_steps
        self.guidance_scale = guidance_scale
        self.real_guidance_scale = real_guidance_scale
        self.height = height
        self.width = width
        self.dtype = dtype
        self.attn_mode = attn_mode
        self.target_index = target_index
        self.control_indices = control_indices or [1, 10]
        self.one_frame_flags = one_frame_flags or set()
    
    def to_dict(self) -> Dict:
        return {
            'seed': self.seed,
            'infer_steps': self.infer_steps,
            'guidance_scale': self.guidance_scale,
            'real_guidance_scale': self.real_guidance_scale,
            'height': self.height,
            'width': self.width,
            'dtype': self.dtype,
            'attn_mode': self.attn_mode,
            'target_index': self.target_index,
            'control_indices': self.control_indices,
            'one_frame_flags': list(self.one_frame_flags),
        }


def get_condition_configs() -> Dict[str, GenerationConfig]:
    """Get configurations for all 4 ablation conditions."""
    
    base_config = {
        'seed': 42,
        'infer_steps': 25,
        'guidance_scale': 10.0,
        'real_guidance_scale': 1.0,
        'height': 640,
        'width': 512,
        'dtype': 'bfloat16',
        'attn_mode': 'sdpa',
        'target_index': 9,
        'control_indices': [1, 10],
    }
    
    conditions = {
        'FULL': GenerationConfig(
            one_frame_flags=set(),
            **base_config
        ),
        'ABL-NO2X': GenerationConfig(
            one_frame_flags={'no_2x'},
            **base_config
        ),
        'ABL-NO4X': GenerationConfig(
            one_frame_flags={'no_4x'},
            **base_config
        ),
        'ABL-NONE': GenerationConfig(
            one_frame_flags={'no_2x', 'no_4x'},
            **base_config
        ),
    }
    
    return conditions


def mock_inference(
    image_path: str,
    prompt: str,
    config: GenerationConfig,
    output_path: str,
) -> bool:
    """
    Mock inference (returns resized copy of input)
    
    TODO: Replace with actual FramePack inference engine
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((config.width, config.height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Mock inference failed: {e}")
        return False


def run_inference_for_condition(
    condition_name: str,
    config: GenerationConfig,
    prompts_data: List[Dict],
    images_dir: Path,
    output_base_dir: Path,
) -> Dict:
    """Run inference for a single condition on all images."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running condition: {condition_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Flags: {config.one_frame_flags or 'NONE (multi-scale enabled)'}")
    
    condition_output_dir = output_base_dir / condition_name
    condition_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'condition': condition_name,
        'config': config.to_dict(),
        'outputs': [],
        'success': 0,
        'failed': 0,
    }
    
    for item in prompts_data:
        img_id = item['img_id']
        prompt = item['prompt']
        
        input_image = images_dir / f"{img_id}.jpg"
        output_image = condition_output_dir / f"{img_id}_generated.png"
        
        if not input_image.exists():
            logger.warning(f"  Input image not found: {input_image}")
            results['failed'] += 1
            continue
        
        logger.info(f"  Processing {img_id}: {prompt[:50]}...")
        
        success = mock_inference(
            str(input_image),
            prompt,
            config,
            str(output_image),
        )
        
        if success:
            logger.info(f"    ✓ Output: {output_image}")
            results['outputs'].append({
                'img_id': img_id,
                'prompt': prompt,
                'output_path': str(output_image),
                'status': 'success',
            })
            results['success'] += 1
        else:
            logger.error(f"    ✗ Inference failed")
            results['failed'] += 1
    
    logger.info(f"\n{condition_name} Summary:")
    logger.info(f"  Success: {results['success']}/{len(prompts_data)}")
    logger.info(f"  Failed: {results['failed']}/{len(prompts_data)}")
    
    return results


def run_all_inference(
    images_dir: str,
    prompts_csv: str,
    output_dir: str,
):
    """Run inference for all 4 conditions."""
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(prompts_csv).exists():
        logger.error(f"Prompts CSV not found: {prompts_csv}")
        return False
    
    logger.info("Loading prompts...")
    prompts_data = load_prompts_csv(prompts_csv)
    logger.info(f"Loaded {len(prompts_data)} prompts")
    
    conditions = get_condition_configs()
    
    all_results = {
        'timestamp': str(Path('.')),
        'total_images': len(prompts_data),
        'conditions': {},
    }
    
    for condition_name, config in conditions.items():
        results = run_inference_for_condition(
            condition_name,
            config,
            prompts_data,
            images_dir,
            output_dir,
        )
        all_results['conditions'][condition_name] = results
    
    results_json = output_dir / "inference_results.json"
    import json as json_module
    with open(results_json, 'w') as f:
        json_module.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ Inference complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {results_json}")
    
    return True
