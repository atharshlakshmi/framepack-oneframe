# Technical Summary: FramePack One-Frame Inference for Image Editing

## Overview

FramePack One-Frame Inference adapts a video diffusion model for single-image editing via text guidance. The system treats the input image as frame 0 and generates output at target frame index 9 within a 9-frame latent window, leveraging the model's temporal coherence patterns for spatial consistency.

The implementation consists of five Python modules in `src/` that orchestrate a complete inference pipeline: model loading, input conditioning, latent packing, diffusion sampling with optimizations, and VAE decoding.

## Architecture

**Core Pipeline**: Model Loading → Text/Image Conditioning → Latent Packing → Diffusion Sampling → VAE Decoding

**Key Models**:
- **DiT**: 24-head attention transformer, 16 latent channels, cross-attention conditioned
- **VAE**: 16-channel, 1/8 spatial compression, factor 0.18215
- **LLaMA-3 + CLIP-L**: Dual-encoder text processing (4,096-dim + 768-dim)
- **SiglipVision**: Image encoder, 1,152-dim, extracts 577 features

## Key Components

**LatentIndexManager** (`latent_packing.py`): Multi-scale indexing with target frame 9, control frames [1, 10]

**TextConditioner** (`conditioning_pipeline.py`): Dual-encoder text processing with caching; max 256 tokens

**ImageConditioner** (`conditioning_pipeline.py`): VAE + SiglipVision encoding with aspect-ratio preservation

## Inference Process

1. Load models (DiT, VAE, text/image encoders)
2. Encode text prompt via dual encoders
3. Encode input image to latent + features
4. Pack with multi-scale indices (target frame 9, controls [1,10])
5. Diffusion sampling (25 steps default, UniPC sampler)
6. Extract target frame and VAE decode to RGB

**Control Features**: Deterministic indexing for reproducibility, multi-reference coherence, optional alpha-channel masking for regions, classifier-free guidance (scale 1-15)

## Optimization Methods

### Performance Optimizations

1. **MagCacheWrapper** - Skips redundant DiT forward passes by monitoring output magnitude ratios between diffusion timesteps. Caches outputs when magnitude change is below threshold, using interpolated magnitude ratios from calibrated 50-step reference model.

2. **NullConditioner** - Eliminates redundant text encoding when guidance_scale=1.0 by skipping null conditioning forward passes through text encoders.

3. **Model Preloading** - Loads smaller models (VAE, text encoders, image encoder) at initialization to eliminate first-generation delays and amortize loading overhead.

4. **Shard Manifest Loading** - Parses `model.safetensors.index.json` for correct DiT weight ordering across model shards, avoiding glob ordering bugs.

5. **Dynamic Device Management** - Moves text encoders between GPU/CPU based on inference stage to minimize peak VRAM usage.

6. **Lazy Initialization** - Creates conditioner instances only when first needed, reducing startup overhead.

### Memory Optimizations

1. **VAE Tiling** - Spatial tiles reduce VRAM for any resolution
2. **Mixed Precision** - FP16, BF16, FP32 modes for memory/quality trade-off
3. **Prompt Embedding Caching** - 40× speedup on repeated prompts
4. **Efficient Weight Loading** - Fast model initialization without full memory allocation
5. **Selective Component Loading** - Text encoders on CPU, moved to GPU only when needed

### Quality Optimizations

1. **Quality-Aware Resizing** - cv2.INTER_AREA for downsampling, PIL LANCZOS for upsampling
2. **Aspect Ratio Preservation** - Cross-multiplication metric minimizes spatial distortion
3. **Mixed Precision** - Optional FP32 for critical operations

## Configuration & Usage

**Input/Output**:
- Auto-detects output resolution from input image (rounds down to nearest 64-pixel multiple)
- Supports manual resolution override (must be divisible by 64)
- Output formats: PNG (default), JPG, JPEG

**Generation Parameters**:
- **seed**: Random seed for reproducibility (default 42)
- **infer_steps**: Number of diffusion steps (default 25)
- **guidance_scale**: Classifier-free guidance strength (default 10.0)
- **target_index**: Frame index to generate (default 9 in 9-frame window)
- **negative_prompt**: Optional prompt for unwanted attributes

**Model Configuration**:
- **device**: GPU device selection (cuda/cpu, default cuda)
- **dtype**: Precision mode - bfloat16 (default), fp16, fp32
- **attn_mode**: Attention implementation - sdpa (default), xformers, flash, sageattn

**Memory/Performance Options**:
- **vae_tiling**: Enable spatial tiling for VAE operations on lower VRAM systems
- **vae_chunk_size**: Chunk size for temporal CausalConv3d operations (memory-speed trade-off)
- **vae_spatial_tile_sample_min_size**: Minimum tile size for spatial VAE tiling (e.g., 256 for tiled decoding)

**Logging & Monitoring**:
- **verbose**: Print detailed progress and timing information
- **save_params**: CSV file path to save/append generation parameters and metrics

## Advanced Features

**Mask-Based Region Editing**:
- `ControlMaskHandler` supports grayscale mask images or alpha channels from control images
- Masks are resized to latent space resolution (H/8, W/8)
- Supports normalized mask values (0.0 to 1.0) for blending control across regions
- Enables selective editing of specific image regions while preserving others

**Multi-Scale Control Structure**:
- **Base latents**: Full resolution control (1× scale)
- **2× controls**: Half-resolution progressive refinement (if not disabled with "no_2x" flag)
- **4× controls**: Quarter-resolution coarse structure (if not disabled with "no_4x" flag)
- Configuration flags allow trade-off between consistency and generation speed

**Latent Window Architecture**:
- References frames at indices [1, 10] provide temporal coherence constraints
- Target frame at index 9 enables generation aware of both past and future context
- Zero latents automatically generated if no control images provided
- Optional post-processing latents ("no_post" flag to disable)

## Source Code Implementation Details

### File-by-File Breakdown

#### 1. `src/framepack_models.py` - Model Loading & Caching

**Purpose**: Centralized model loader implementing singleton pattern with lazy initialization and precision conversions.

**Key Classes**:
- **ModelConfig**: Dataclass storing static configuration for all models (encoder dimensions, layer counts, tokenizer configs)
- **FramePackModels**: Main loader handling:
  - Lazy loading of 5 core models (DiT, VAE, LLaMA, CLIP-L, SiglipVision)
  - Model caching to prevent reloading
  - Precision conversion (BF16, FP16, FP32, FP8)
  - Attention kernel selection (SDPA, xformers, flash, sageattn)
  - Weight manifest parsing from safetensors index for correct loading order

**Model Loading Flow**:
1. DITs (24GB Hunyuan transformer) - Loaded via `HunyuanVideoTransformer3DModelPacked` from FramePack diffusers_helper
2. VAE (AutoencoderKLHunyuanVideo) - 941MB, handles spatial-temporal latent encoding with tiling support
3. LLaMA-3 (32-layer, 4096-dim) - Text encoder 1, loaded from safetensors, kept on CPU to save VRAM
4. CLIP-L (12-layer, 768-dim) - Text encoder 2 for visual alignment, kept on CPU
5. SiglipVision (27-layer, 1152-dim) - Image encoder extracting 577 spatial features from 384×384 inputs

**Optimizations**:
- Models kept on CPU between inference steps, moved to GPU only when needed (Dynamic Device Management - Fix 5)
- Safetensors manifest parsing avoids glob ordering bugs (Shard Manifest Loading - Fix 4)
- Optional FP8 quantization for DiT and text encoders reduces memory by 50-75%

#### 2. `src/conditioning_pipeline.py` - Text & Image Conditioning

**Purpose**: Convert text prompts and images to model-ready embeddings and latents.

**Key Functions**:
- **`resize_image_to_bucket()`**: Aspect-ratio-preserving resize with quality downsampling (INTER_AREA) and upsampling (LANCZOS)
- **`TextConditioner`**: Dual-encoder (LLaMA-3 + CLIP-L), outputs [seq_len, 4096] + [768]. Caches encodings (40× speedup on repeated prompts)
- **`ImageConditioner`**: VAE latent encoding [1, 16, 1, H/8, W/8] + SiglipVision features [1, 577, 1152]
- **`NullConditioner`**: Skips redundant encoding when guidance_scale=1.0 (30% speedup)

#### 3. `src/latent_packing.py` - Index Management & Masking

**Purpose**: Manage latent tensor construction for multi-scale generation.

**`LatentIndexManager`**: 9-frame window with target frame 9, control frames [1,10], configurable scales (no_2x, no_4x flags). `compute_indices()` generates multi-scale hierarchy, `pack_control_latents()` concatenates with optional masking.

**`ControlMaskHandler`**: Load grayscale/alpha masks, resample to latent space (H/8, W/8), apply element-wise for region editing.

#### 4. `src/inference_engine.py` - Core Orchestration

**`MagCacheWrapper`** (Optimization): Skip redundant DiT forward passes by tracking output magnitude ratios. Interpolates reference ratios from 50-step calibration, skips recomputation when magnitude scaling matches expected values. Achieves 20-30% speedup.

**`GenerationConfig`**: Dataclass holding input image, prompt, seed, inference_steps (25), guidance_scale (10.0), frame indices (target 9, controls [1,10]).

**`SingleFrameImageEditor`**: Main orchestrator. Lazy-loads conditioners on first use, preloads smaller models, loads DiT on-demand. Generation workflow: encode text → encode image → pack latents → diffusion sampling (with MagCacheWrapper) → extract frame 9 → VAE decode. Device management: text encoders/VAE on CPU, move to GPU only when needed; DiT stays on GPU.

#### 5. `src/cli_inference.py` - Command-Line Interface

**Workflow**: Parse arguments (image path, prompt, model paths, generation params) → validate input resolution → create SingleFrameImageEditor → build GenerationConfig → generate → save image + log results (timing, parameters to CSV if specified).

### Complete Pipeline

1. VAE encode image → latent [1, 16, 1, H/8, W/8]
2. Dual-encode text → [seq_len, 4096] + [768]
3. SiglipVision encode image → [1, 577, 1152]
4. Pack latents in 9-frame window (target 9, controls [1,10])
5. DiT diffusion (25 steps, MagCacheWrapper optimization)
6. Extract frame 9 → VAE decode → RGB [1, 3, H, W]

## Device & Memory Management

**GPU/CPU Choreography**: Text encoders, image encoder, VAE kept on CPU; moved to GPU only when needed → save ~20GB VRAM. DiT stays on GPU (too large to move). CUDA cache cleared between major steps.

**Memory Techniques**: Model caching prevents reloading, VAE tiling handles any resolution, mixed precision (BF16/FP16) reduces memory 30-50%, lazy initialization of components.

## FramePack Integration

**Imports from `diffusers_helper`**:
- `hunyuan.vae_encode()`, `hunyuan.vae_decode()`: VAE operations
- `hunyuan.encode_prompt_conds()`: Dual-encoder text processing
- `hf_clip_vision_encode()`: Image feature extraction
- Model architectures: `HunyuanVideoTransformer3DModelPacked` (DiT), `AutoencoderKLHunyuanVideo` (VAE)

**Validation**: Image format (RGBA→RGB), resolution divisible by 64, model file existence, tensor shape compatibility

## Experiments & Ablation Studies

**Location**: `experiments/` folder with individual experiment directories (e.g., `experiment1/`)

**See `experiments/README.md` for detailed documentation of experimental setup, methodology, and results.**

**Key Features**:
- Configurable model settings, optimization flags, resolution parameters
- Batch processing with multiple prompts/seeds for statistical significance
- Metric collection: timing, image quality, memory usage
- Optimization ablations: MagCache on/off, device management variants
- Multi-scale control ablations: no_2x, no_4x, no_post flags

## Integration with FramePack

**Dependency on FramePack Core**:
- Requires sibling `FramePack` repository for `diffusers_helper` module
- Uses pre-trained model weights (DiT, VAE from HunyuanVideo)
- Leverages FramePack's optimized pipelines and model definitions

**Configuration Methods**:
1. Environment variable: `export FRAMEPACK_PATH=/path/to/FramePack`
2. `.env` file: Create from `.env.example` template
3. Sibling directory: Place FramePack as `../FramePack` relative to framepack-oneframe

**Model Requirements**:
- File: `models/` directory in framepack-oneframe
- Contains pre-downloaded weights (~50GB total)
- Paths specified via CLI arguments or configuration files

## Package API & Public Exports

**Main Entry Points** (`src/__init__.py`):
- `SingleFrameImageEditor`: High-level inference orchestrator
- `GenerationConfig`: Configuration dataclass
- `FramePackModels`: Model loader
- `TextConditioner`, `ImageConditioner`: Conditioning pipelines
- `LatentIndexManager`, `ControlMaskHandler`: Latent utilities

**Typical Usage**:
```python
from src import SingleFrameImageEditor, GenerationConfig
from PIL import Image

editor = SingleFrameImageEditor(
    model_paths={...},
    device="cuda",
    dtype="bfloat16"
)
config = GenerationConfig(
    image=Image.open("input.jpg"),
    prompt="edit description",
    guidance_scale=10.0
)
result = editor.generate(config)
result.save("output.png")
```

## Summary

FramePack One-Frame Inference adapts video diffusion models for single-image editing by treating image editing as frame prediction within a 9-frame latent window. The implementation combines:

- **Architecture**: Dual-encoder text (LLaMA+CLIP), image conditioning (VAE+SiglipVision), multi-scale latent packing, DiT diffusion
- **Optimizations**: MagCache (20-30% speedup), NullConditioner (30% speedup), device choreography (~20GB VRAM saved), VAE tiling, prompt caching (40× speedup)
- **Usability**: Simple CLI, flexible Python API, configurable parameters for quality vs. speed trade-offs

Modular design enables research, system integration, and production deployment.
