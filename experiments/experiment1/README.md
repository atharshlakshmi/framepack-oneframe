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
- **CLIP Score**: Prompt-image alignment (higher = better prompt adherence) ★ Saved immediately after each image
- **SSIM**: Structural Similarity Index (compared to idx_9 baseline)
- **LPIPS**: Perceptual difference from idx_9 baseline
- **Performance**: Generation time and GPU memory usage

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install CLIP (One-Time)

**Option A: If you already have CLIP installed**

Skip this step — CLIP is already available.

**Option B: If CLIP is not installed**

Install from git (may be slow on first run due to compilation):
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 3. Setup CLIP Weights (One-Time)

Download CLIP weights to bypass SSL certificate issues:

```bash
# This downloads ViT-B-32.pt using wget (bypasses SSL)
python fix_clip_ssl.py
```

This creates `~/.cache/clip/ViT-B-32.pt` (354 MB). After this, CLIP loads from local cache with no network access.

### 4. Run the Study

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
    ├── results.csv                    # ★ Per-image metrics (80 rows, 4 indices × 20 images)
    └── summary_metrics.csv            # ★ Summary statistics per index (for poster/report)
```

## Results CSV Format

### `results.csv` (Per-Image Metrics)

`ablation_study/metrics/results.csv` contains one row per image-index pair (80 total rows):

| Column | Description | Example |
|--------|-------------|----------|
| condition | Index condition | idx_9, idx_12, idx_15, idx_20 |
| img_id | Image ID | img_001–img_020 |
| prompt | Edit instruction from dataset | "make the sky more blue" |
| output_path | Path to generated image | ablation_study/idx_9/img_001_generated.png |
| total_inference_time | End-to-end generation (seconds) | 18.5 |
| peak_vram_gb | GPU memory usage (GB) | 24.3 |
| **clip_score** | Prompt adherence (higher=better) | 0.82 |
| **ssim** | Similarity to idx_9 baseline (higher=better) | 0.91 |
| **lpips** | Perceptual distance from idx_9 (lower=better) | 0.045 |
| error_flag | Generation success | False |

### `summary_metrics.csv` (For Poster/Report) ★ NEW

Automatically generated at end of study. Contains one row per condition with aggregate statistics:

| Column | Description | Example |
|--------|-------------|----------|
| Condition | Index condition | idx_9 |
| Samples | Total images | 20 |
| Success | Successful generations | 20 |
| Failed | Failed generations | 0 |
| **CLIP_Mean** | Average prompt adherence | 0.82 |
| **CLIP_Std** | Standard deviation | 0.03 |
| **LPIPS_Mean** | Average perceptual distance | 0.021 |
| **LPIPS_Std** | Standard deviation | 0.008 |
| **SSIM_Mean** | Average structural similarity | 0.91 |
| **SSIM_Std** | Standard deviation | 0.02 |
| InferTime_Sec_Mean | Average generation time | 18.5 |
| InferTime_Sec_Std | Standard deviation | 0.3 |
| PeakVRAM_GB_Mean | Average GPU memory | 24.1 |
| PeakVRAM_GB_Max | Maximum GPU memory | 24.8 |

**Use this table directly in posters/reports!**

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
2. **run_ablation_study.py**: Main orchestrator (loops through 4 indices, generates summary metrics)
3. **run_cli_inference()**: Subprocess wrapper that passes `--target_index` parameter
4. **clip_scorer.py**: Loads CLIP from local cache, computes prompt-image alignment
5. **fix_clip_ssl.py**: Downloads CLIP weights via wget (one-time setup)

### CLIP Scoring Details

**Problem**: CLIP's default downloader fails on networks with proxy/firewall (SSL certificate issues).

**Solution**: 
- `fix_clip_ssl.py` downloads weights via `wget --no-check-certificate` (bypasses Python's SSL stack)
- `clip_scorer.py` loads from `~/.cache/clip/ViT-B-32.pt` (local file path)
- CLIP accepts file paths directly — download code is never executed

**Result**: CLIP scores computed **immediately after each image is generated** (live, not post-hoc), stored in results.csv.

**Dataset Pipeline**:
```
InstructPix2Pix (HuggingFace: timbrooks/instructpix2pix-clip-filtered)
  → Load dataset with datasets library
  → Sample 20 pairs with seed=42 (original_image + edit_prompt)
  → Save to ablation_study/images/ and prompts.csv
  → Run inference for each original_image × 4 indices with edit_prompt
  → Compute metrics immediately:
    - CLIP score (vs prompt)
    - SSIM/LPIPS (vs idx_9 baseline)
  → Log to results.csv row-by-row
  → Generate summary_metrics.csv at end
```

**Dataset Fields**:
- `original_image`: PIL Image (input to FramePack)
- `edit_prompt`: str (editing instruction passed to FramePack)
- `edited_image`: PIL Image (ground truth - not used for ablation, just reference)

**Metrics**:
- CLIP Score: Image-text similarity (CLIP ViT-B/32 embeddings, normalized cosine similarity)
- SSIM: Structural similarity (scikit-image, computed vs idx_9 baseline)
- LPIPS: Perceptual distance (L1 on normalized representations, computed vs idx_9 baseline)

## Troubleshooting

### CLIP SSL Error

**Error**: `SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]` when importing CLIP

**Fix**: Run setup once:
```bash
python fix_clip_ssl.py
```

This downloads weights via `wget --no-check-certificate` → `~/.cache/clip/ViT-B-32.pt`.

After this, `clip_scorer.py` loads from local cache (no SSL issues, no network access needed).

### HuggingFace Authentication

If dataset doesn't download:
```bash
huggingface-cli login  # Then enter your token
```

### OOM Errors

Check `nvidia-smi`, reduce `--infer_steps` in helpers.py

### Slow Generation

Monitor GPU with `watch -n 1 nvidia-smi`

### Import Errors

Ensure all packages installed:
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
