# Quick Setup Guide

## Structure Consolidated ✓

**Before**: 10+ separate Python scripts
**After**: 2 main files + 1 helper module

File organization:
```
run_ablation_study.sh        ← Bash wrapper (entry point)
run_ablation_study.py        ← Main orchestrator (imports helpers)
helpers.py                   ← All utilities (consolidated)
  ├── download_coco()        (was: download_coco.py)
  ├── sample_images()        (was: sample_images.py)
  ├── run_all_inference()    (was: run_inference.py)
  ├── compute_all_metrics()  (was: compute_metrics.py)
  └── metric functions       (was: metrics.py)
```

## 30-Second Start

```bash
cd experiments/experiment1
./run_ablation_study.sh
```

That's it! The script will:
1. Install dependencies
2. Download COCO val2017 (~1 GB)
3. Sample 20 images
4. Run inference for 4 conditions
5. Compute metrics
6. Generate `ablation_study/metrics.csv`

## What Gets Created

After the study completes, you'll have:

```
ablation_study/
├── images/              20 sampled COCO images
├── prompts.csv          Image metadata + captions
├── outputs/             Generated images
│   ├── FULL/            Full pipeline (reference)
│   ├── ABL-NO2X/        2x controls disabled
│   ├── ABL-NO4X/        4x controls disabled
│   └── ABL-NONE/        Both controls disabled
└── metrics.csv          ✓ MAIN RESULT
```

## Key Files Explained

| File | Purpose |
|------|---------|
| `run_ablation_study.sh` | Bash runner (easy, all-in-one) |
| `run_ablation_study.py` | Python orchestrator (flexible) |
| `download_coco.py` | Download ~1 GB dataset |
| `sample_images.py` | Sample 20 images (seed=42, fixed) |
| `run_inference.py` | Generate outputs for all conditions |
| `compute_metrics.py` | Calculate LPIPS, DSSIM, Huber, entropy, variance |
| `metrics.py` | Metrics computation module |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |

## If Restarting

```bash
# Clean slate (removes large files)
rm -rf ablation_study/ coco_data/

# Run again
./run_ablation_study.sh
```

## If Resuming (data already exists)

```bash
# Skip download and sampling, go straight to inference
./run_ablation_study.sh --skip-download --skip-sampling
```

## Expected Runtime

- **COCO download**: ~10-20 minutes (one-time, network dependent)
- **Image sampling**: ~1 minute
- **Inference (4 conditions × 20 images)**: ~30-60 minutes (GPU dependent)
- **Metrics**: ~5-10 minutes

**Total: ~1-2 hours** (mostly waiting for inference)

## Output Format

`metrics.csv` preview:

```
condition,img_id,lpips,dssim,huber,entropy_abl,variance_abl
ABL-NO2X,img_001,0.043,0.018,0.032,5.156,0.0118
ABL-NO4X,img_001,0.021,0.009,0.015,5.189,0.0121
ABL-NONE,img_001,0.067,0.031,0.051,5.098,0.0115
...
```

Lower values = closer to FULL condition output

## Troubleshooting

**Q: Permission Denied on `run_ablation_study.sh`**
```bash
chmod +x run_ablation_study.sh
```

**Q: Python not found**
```bash
# Ensure you have conda/python installed
python3 --version

# If not installed on Linux:
sudo apt-get install python3 python3-pip
```

**Q: Out of Memory during inference**
Edit `run_inference.py` and reduce:
- `height: 640` → `512`
- `width: 512` → `384`

**Q: Network timeout downloading COCO**
Run again with:
```bash
./run_ablation_study.sh --skip-download
```
(if partial download succeeded)

## Next Steps

1. Review `ablation_study/metrics.csv` for results
2. Check `ablation_study/outputs/*/` for generated images
3. Analyze which controls contribute most to quality
4. Consider running on full COCO (5,000 images) for statistical significance

---

For complete documentation, see [README.md](README.md)
