# FramePack Ablation Study: Multi-Scale Controls

This directory contains the complete ablation study protocol to evaluate the contribution of multi-scale control hierarchies in FramePack One-Frame Inference.

## Study Overview

**Research Question**: How much do the 2× and 4× upsampling control latents contribute to output quality and generation time?

**Conditions Tested**:
- **FULL**: Full pipeline with 2× and 4× multi-scale controls enabled
- **ABL-NO2X**: 2× controls disabled only
- **ABL-NO4X**: 4× controls disabled only
- **ABL-NONE**: Both controls disabled

**Dataset**: COCO val2017 (20 sampled images with fixed seed)

**Metrics Computed**:
- **LPIPS**: Learned Perceptual Image Patch Similarity (perceptual distance)
- **DSSIM**: Structural Dissimilarity (structural distance)
- **Huber Distance**: Robust L1 loss (pixel-level distance)
- **Entropy**: Shannon entropy of output (information content)
- **Variance**: Local patch variance (smoothness)

All metrics compare each ablation condition output **to the FULL condition output**, not to the original COCO image.

## Quick Start

### Option 1: Bash Script (Recommended)

```bash
# Full run with all steps
./run_ablation_study.sh

# Skip download if dataset already exists
./run_ablation_study.sh --skip-download

# Skip download and sampling (uses existing data)
./run_ablation_study.sh --skip-download --skip-sampling

# Verbose output
./run_ablation_study.sh --verbose
```

### Option 2: Manual Steps

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Download COCO val2017 (~1 GB)
python download_coco.py

# 2. Sample 20 images with fixed seed
python sample_images.py

# 3. Run inference for all 4 conditions
python run_inference.py

# 4. Compute metrics
python compute_metrics.py
```

### Option 3: Python Script

```bash
python run_ablation_study.py [--skip-download] [--skip-sampling] [--force] [--verbose]
```

## File Structure

After consolidation, the directory contains:

```
experiment1/
├── run_ablation_study.sh              # Bash entry point
├── run_ablation_study.py              # Main orchestrator
├── helpers.py                         # ★ All consolidated utilities
│   ├── download_coco()
│   ├── sample_images()
│   ├── run_all_inference()
│   ├── compute_all_metrics()
│   └── metric functions (LPIPS, SSIM)
│
├── config_template.py                 # Configuration template
├── requirements.txt                   # Dependencies
├── README.md                          # This file
├── SETUP_GUIDE.md                     # Quick start
├── .gitignore
│
├── coco_data/                         # Downloaded COCO dataset
├── ablation_study/                    # Results
│   ├── images/                        # 20 sampled images
│   ├── prompts.csv                    # Metadata + captions
│   ├── outputs/
│   │   ├── FULL/
│   │   ├── ABL-NO2X/
│   │   ├── ABL-NO4X/
│   │   └── ABL-NONE/
│   └── metrics.csv                    # RESULTS
```

## Fixed Inference Configuration

All 4 conditions use identical inference parameters to isolate only the effect of multi-scale controls:

```
  seed:                42
  infer_steps:         25
  guidance_scale:      10.0
  real_guidance_scale: 1.0
  height × width:      640 × 512
  dtype:               bfloat16
  attn_mode:           sdpa
  target_index:        9
  control_indices:     [1, 10]
  MagCache:            enabled
  VAE tiling:          disabled
```

## Reproducibility Notes

⚠️ **CRITICAL**: The image sampling uses `seed=42` for reproducibility. **Never re-run `sample_images.py`** once you've generated the initial sample. The prompt list must remain identical across all conditions.

If you need to restart from scratch:

```bash
# Completely reset the study
rm -rf ablation_study/
rm -rf coco_data/

# Then run from the beginning
./run_ablation_study.sh
```

## Understanding the Results

### metrics.csv Structure

```
condition,img_id,lpips,dssim,huber,entropy_abl,variance_abl
FULL,img_001,0.0,0.0,0.0,5.234,0.0125
ABL-NO2X,img_001,0.043,0.018,0.032,5.156,0.0118
ABL-NO4X,img_001,0.021,0.009,0.015,5.189,0.0121
ABL-NONE,img_001,0.067,0.031,0.051,5.098,0.0115
...
```

### Interpreting Metrics

Lower values are better for all metrics (except entropy, which is descriptive):

- **LPIPS < 0.05**: Imperceptible difference from FULL
- **LPIPS 0.05-0.10**: Slight perceptual difference
- **LPIPS > 0.10**: Noticeable perceptual difference

- **DSSIM < 0.01**: Very similar structure
- **DSSIM 0.01-0.05**: Moderate structural change
- **DSSIM > 0.05**: Significant structural change

- **Huber < 0.05**: Minimal pixel-level changes
- **Huber 0.05-0.15**: Moderate pixel-level changes
- **Huber > 0.15**: Major pixel-level changes

### Aggregate Statistics

After computing metrics, you can analyze:

```python
import pandas as pd

df = pd.read_csv('ablation_study/metrics.csv')

# Average metrics by condition
print(df.groupby('condition')[['lpips', 'dssim', 'huber']].mean())

# Percentage change from FULL (baseline)
full_metrics = df[df['condition'] == 'FULL'].groupby('img_id')[['lpips', 'dssim', 'huber']].mean()
for cond in ['ABL-NO2X', 'ABL-NO4X', 'ABL-NONE']:
    abl_metrics = df[df['condition'] == cond].groupby('img_id')[['lpips', 'dssim', 'huber']].mean()
    pct_increase = (abl_metrics - full_metrics) / full_metrics * 100
    print(f"\n{cond} vs FULL:")
    print(pct_increase.describe())
```

## Implementation Details

### How Multi-Scale Controls Work

FramePack uses hierarchical conditioning with latent controls at multiple upsampling levels:

1. **2× upsampling latents**: Control at intermediate resolution
2. **4× upsampling latents**: Control at higher resolution
3. When disabled: Information is only from the base latent level

Setting `one_frame_flags = set()` (FULL) enables multi-scale.
Setting `one_frame_flags = {'no_2x', 'no_4x'}` (ABL-NONE) uses only base latents.

### Metrics Implementation

- **LPIPS**: Approximated using normalized image gradient L2 distance (full LPIPS requires pretrained network)
- **DSSIM**: Structural Similarity computed with Gaussian kernel
- **Huber**: Robust loss function (less sensitive to outliers than L2)
- **Entropy**: Shannon entropy in pixel value distribution
- **Variance**: Average patch-wise variance for smoothness evaluation

For production use, consider:
- Installing `lpips` library for accurate LPIPS: `pip install lpips`
- Using `pytorch-ssim` for optimized SSIM: `pip install pytorch-ssim`

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors during inference:
1. Reduce image resolution in `run_inference.py` (height, width)
2. Reduce batch size (if implemented)
3. Enable VAE tiling: set `vae_tile_size = 256`

### Missing Dataset

```
ERROR: Captions file not found: coco_data/annotations/captions_val2017.json
```

Solution: Run `python download_coco.py` first

### Prompts CSV Not Found

```
ERROR: Prompts CSV not found: ablation_study/prompts.csv
```

Solution: Run `python sample_images.py` after downloading

### Slow Inference

- Check GPU availability: `nvidia-smi`
- Consider using `--skip-sampling` to reuse existing data
- Monitor memory usage during generation

## Next Steps

1. **Analyze Results**: Review `metrics.csv` for quality degradation patterns
2. **Visual Inspection**: Compare generated images in `outputs/` subdirectories
3. **Speed Profiling**: Add timing code to `run_inference.py` to measure generation speed
4. **Extended Study**: Run on full COCO validation set (5,000 images) for statistical significance
5. **Other Ablations**: Study other pipeline components (MagCache, attention modes, guidance scales)

## References

- [COCO Dataset](https://cocodataset.org/)
- [Ablation Study Protocol](../working_md/experiment1.md)
- [FramePack Documentation](../../README.md)

## Contact

For questions about this ablation study, refer to the protocol document: `../working_md/experiment1.md`

---

*Last Updated: 2026-03-27*
*Study: Multi-Scale Controls in FramePack One-Frame Inference*
