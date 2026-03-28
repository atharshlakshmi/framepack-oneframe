# FramePack Ablation Study: Target Frame Index Optimization

This directory contains an ablation study to optimize the target frame index for FramePack One-Frame Inference using paired image-instruction data.

## Study Overview

**Dataset**: InstructPix2Pix (20 samples from HuggingFace timbrooks/instructpix2pix-clip-filtered)

**Research Question**: Which target frame index produces the best output quality and prompt adherence?

**Conditions Tested**:
- **idx_9**: Target index = 9 (baseline)
- **idx_12**: Target index = 12
- **idx_15**: Target index = 15
- **idx_20**: Target index = 20

**Metrics Computed**:
- **CLIP Score**: Prompt-image alignment (higher = better prompt adherence)
- **SSIM**: Structural Similarity Index (compared to idx_9 baseline)
- **LPIPS**: Perceptual difference from idx_9 baseline
- **Performance**: Generation time and GPU memory usage

## Quick Start

### Run the Study

```bash
# Full run (loads InstructPix2Pix from HuggingFace)
bash run_ablation_study.sh

# Or run Python directly
python run_ablation_study.py
```

The dataset is automatically downloaded from HuggingFace on first run.

## File Structure

After running the experiment:

```
ablation_study/
├── images/                            # 20 InstructPix2Pix paired samples (original_image)
├── prompts.csv                        # Image metadata + edit instructions (edit_prompt)
├── idx_9/                             # Generated images (target_index=9)
├── idx_12/                            # Generated images (target_index=12)
├── idx_15/                            # Generated images (target_index=15)
├── idx_20/                            # Generated images (target_index=20)
└── metrics/
    └── results.csv                    # ★ Per-image metrics table
```

## Results CSV Format

`ablation_study/metrics/results.csv` contains one row per image-index pair (80 total rows):

| Column | Description | Example |
|--------|-------------|---------|
| condition | Index condition | idx_9, idx_12, idx_15, idx_20 |
| img_id | Image ID | img_001–img_020 |
| prompt | Edit instruction from dataset | "make the sky more blue" |
| output_path | Path to generated image | ablation_study/idx_9/img_001_generated.png |
| total_inference_time | End-to-end generation (seconds) | 18.5 |
| peak_vram_gb | GPU memory usage (GB) | 24.3 |
| clip_score | Prompt adherence (higher=better) | 0.82 |
| ssim | Similarity to idx_9 baseline (higher=better) | 0.91 |
| lpips | Perceptual distance from idx_9 (lower=better) | 0.045 |
| error_flag | Generation success | False |

## Fixed Inference Configuration

All 4 index conditions use identical parameters except target_index:

```
seed:             42
infer_steps:      25
guidance_scale:   10.0
height × width:   640 × 512
dtype:            bfloat16
attn_mode:        sdpa
control_indices:  [1, 10]
target_index:     [9, 12, 15, 20]  ← Only parameter that varies
```

## Running the Study

```bash
cd /mnt/hdd2/atharshlakshmi/framepack-oneframe/experiments/experiment1
bash run_ablation_study.sh
```

**Expected runtime**: ~3-4 hours (4 indices × 20 images on RTX A6000)

## Analyzing Results

After running, compare metrics across indices:

```python
import pandas as pd

df = pd.read_csv('ablation_study/metrics/results.csv')

# Average metrics by index
print(df.groupby('condition')[['clip_score', 'ssim', 'lpips', 'total_inference_time']].mean())

# Which index has best CLIP score?
best_idx = df.groupby('condition')['clip_score'].mean().idxmax()
print(f"Best for prompt adherence: {best_idx}")
```

**Interpretation**:
- **CLIP Score > 0.75**: Good instruction adherence
- **SSIM > 0.90**: Very similar to baseline idx_9
- **LPIPS < 0.05**: Negligible perceptual difference

## Implementation

The study uses:

1. **helpers.py**: InstructPix2Pix dataset loading from HuggingFace, metrics computation, CSV logging
2. **run_ablation_study.py**: Main orchestrator (loops through 4 indices)
3. **run_cli_inference()**: Subprocess wrapper that passes `--target_index` parameter

**Dataset Pipeline**:
```
InstructPix2Pix (HuggingFace: timbrooks/instructpix2pix-clip-filtered)
  → Load dataset with datasets library
  → Sample 20 pairs with seed=42 (original_image + edit_prompt)
  → Save to ablation_study/images/ and prompts.csv
  → Run inference for each original_image × 4 indices with edit_prompt
  → Compute metrics vs idx_9 baseline
```

**Dataset Fields**:
- `original_image`: PIL Image (input to FramePack)
- `edit_prompt`: str (editing instruction passed to FramePack)
- `edited_image`: PIL Image (ground truth - not used for ablation, just reference)

**Metrics**:
- CLIP Score: Image-text similarity (CLIP embeddings)
- SSIM: Structural similarity (scikit-image)
- LPIPS: Perceptual distance (L1 on normalized representations)

## Troubleshooting

**HuggingFace authentication**: If dataset doesn't download:
```bash
huggingface-cli login  # Then enter your token
```

**OOM errors**: Check `nvidia-smi`, reduce `--infer_steps` in helpers.py

**Slow generation**: Monitor GPU with `watch -n 1 nvidia-smi`

**Import errors**: Ensure `datasets` package is installed:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Compare CLIP Scores**: Which index has best instruction adherence?
2. **Visual Inspection**: Compare generated images across all 4 indices
3. **Extend Study**: Test with more indices or different guidance scales
4. **Scaling**: Run on full InstructPix2Pix train split for more samples

---

*Last Updated: 2026-03-29*
*Study: Target Frame Index Optimization using InstructPix2Pix Dataset*
*Data Source: HuggingFace timbrooks/instructpix2pix-clip-filtered*
