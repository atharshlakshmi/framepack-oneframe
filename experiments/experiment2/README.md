# Experiment 2: Qualitative Study - Batch Inference

This experiment runs FramePack One-Frame Inference on a batch of images with multiple prompts using the qualitative study script.

## Directory Structure

```
experiment2/
├── images/                  # Input images folder (create this)
├── prompts.csv             # CSV file with image-prompt pairs
├── output/                 # Generated outputs (auto-created)
│   └── results.csv         # Results log
├── run_qualitative_study.sh # Main batch inference script
└── README.md               # This file
```

## Setup

### 1. Prepare Images Folder

Create an `images/` directory and place your input images:

```bash
cd experiment2
mkdir -p images
# Copy your images here
cp /path/to/your/images/* images/
```

Supported formats: `.jpg`, `.png`, `.jpeg`, etc.

### 2. Prepare Prompts CSV

Edit `prompts.csv` with the image-prompt pairs you want to test:

```csv
image_filename,prompt,seed
car.jpg,make the car red,42
car.jpg,add a sunset background,43
landscape.png,add more vibrant colors,44
```

**CSV Format:**
- `image_filename`: Name of the image file in `images/` folder
- `prompt`: Text description of the desired edit
- `seed`: Random seed for reproducibility (optional, will auto-increment if empty)

### 3. Configure Environment (Optional)

The script automatically loads settings from the project `.env` file:

```bash
# View current settings
cat ../../.env

# Or set environment variables manually
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # GPU memory optimization
export DEVICE=cuda:1                                       # GPU device (0 or 1)
export INFER_STEPS=25                                      # Inference steps
export GUIDANCE_SCALE=10.0                                 # Guidance scale
```

## Running the Tests

### Basic Usage

```bash
cd experiment2
bash run_qualitative_study.sh
```

Uses defaults:
- Images: `./images/`
- Prompts: `prompts.csv`
- Output: `./output/qualitative_study/`

### Custom Directories

```bash
bash run_qualitative_study.sh <images_dir> <prompts_csv> [output_dir]
```

Example:
```bash
bash run_qualitative_study.sh ./images prompts.csv ./my_outputs
```

### Advanced Options

Customize inference parameters via environment variables:

```bash
# Change GPU device
DEVICE=cuda:0 bash run_qualitative_study.sh

# Increase inference steps for higher quality
INFER_STEPS=50 bash run_qualitative_study.sh

# Adjust guidance scale
GUIDANCE_SCALE=7.5 bash run_qualitative_study.sh

# Combine multiple options
DEVICE=cuda:1 INFER_STEPS=30 GUIDANCE_SCALE=12.0 bash run_qualitative_study.sh
```

## Output

The script generates:

1. **Individual outputs**: Saved in `output/qualitative_study/` with naming format:
   ```
   {image_name}_{timestamp}_seed{seed}.png
   ```

2. **Results CSV**: `output/qualitative_study/results.csv` 
   
   Tracks each inference with columns:
   - `image_filename`: Input image
   - `prompt`: Text prompt used
   - `output_filename`: Generated output file
   - `seed`: Random seed
   - `infer_steps`: Number of diffusion steps
   - `guidance_scale`: Classifier-free guidance value
   - `status`: SUCCESS, FAILED, or SKIPPED
   - `timestamp`: When the inference ran


## Troubleshooting

### GPU Out of Memory

If you see `CUDA out of memory` errors:

1. **Reduce inference steps**:
   ```bash
   INFER_STEPS=15 bash run_qualitative_study.sh
   ```

2. **Use GPU memory optimization** (already enabled):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

3. **Check GPU status**:
   ```bash
   nvidia-smi  # Shows memory usage per GPU
   ```

### Image Not Found

Make sure image files are in the `images/` folder and filenames in `prompts.csv` match exactly (case-sensitive).

### Import Errors

Ensure `.env` file is properly set with `FRAMEPACK_PATH`:
```bash
cat ../../.env | grep FRAMEPACK_PATH
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda:1` | GPU device to use (`cuda:0`, `cuda:1`, etc.) |
| `INFER_STEPS` | `25` | Number of diffusion steps |
| `GUIDANCE_SCALE` | `10.0` | Classifier-free guidance strength |
| `TARGET_INDEX` | `9` | Target frame index in latent window |
| `DTYPE` | `bfloat16` | Model precision (`bfloat16`, `fp16`, `fp32`) |
| `ATTN_MODE` | `sdpa` | Attention mechanism (`sdpa`, `xformers`, `flash`, `sageattn`) |
| `SEED_BASE` | `42` | Base seed for auto-incrementing |

## Performance Tips

- **Faster inference**: Reduce `INFER_STEPS` (15-20 is often sufficient)
- **Better quality**: Increase `INFER_STEPS` (40-50) or `GUIDANCE_SCALE` (12-15)
- **Lower memory**: Reduce `INFER_STEPS` or use smaller images
- **Parallel processing**: Run multiple instances on different GPUs

## Script Details

The `run_qualitative_study.sh` script:

1. Validates input images and CSV
2. Creates output directory and results CSV
3. Iterates through each image-prompt pair
4. Calls `cli_inference.py` with appropriate parameters
5. Logs results (success/failure/skip) with timestamps
6. Prints summary statistics

See the script source for advanced customization.
