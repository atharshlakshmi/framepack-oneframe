#!/usr/bin/env python3
"""
Command-Line Interface for FramePack One-Frame Inference

Single-frame image editing using FramePack models.
"""

import sys
import os
import argparse
import time
from pathlib import Path
import numpy as np
import csv

# Add src/ directory to path (for framepack_models, inference_engine imports)
sys.path.insert(0, os.path.dirname(__file__))

# Add FramePack to path (configure via FRAMEPACK_PATH environment variable)
framepack_path = os.environ.get("FRAMEPACK_PATH")
if framepack_path:
    sys.path.insert(0, framepack_path)
else:
    # Try to find FramePack in common locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "FramePack"),
        os.path.join(os.path.expanduser("~"), "FramePack"),
        "/path/to/FramePack"  # Placeholder
    ]
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            break
    else:
        print("WARNING: FramePack not found. Set FRAMEPACK_PATH environment variable.")
        print("Example: export FRAMEPACK_PATH=/path/to/FramePack")

import torch
from PIL import Image

# Import our modules
from framepack_models import FramePackModels
from inference_engine import SingleFrameImageEditor, GenerationConfig


def main():
    total_start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="FramePack One-Frame Inference - Single Image Editing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save output image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing desired edit")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt (things to avoid)")
    
    # Model Paths
    parser.add_argument("--dit", type=str, required=True,
                        help="Path to DiT model (safetensors)")
    parser.add_argument("--vae", type=str, required=True,
                        help="Path to VAE model (pytorch_model.pt or safetensors)")
    parser.add_argument("--text_encoder1", type=str, required=True,
                        help="Path to LLaMA text encoder (safetensors)")
    parser.add_argument("--text_encoder2", type=str, required=True,
                        help="Path to CLIP-L text encoder (safetensors)")
    parser.add_argument("--image_encoder", type=str, required=True,
                        help="Path to SiglipVision image encoder (safetensors)")
    
    # Generation Parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--infer_steps", type=int, default=25,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=10.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--target_index", type=int, default=9,
                        help="Target frame index in latent window")
    
    # Resolution (optional - auto-detected from input image if not specified)
    parser.add_argument("--height", type=int, default=None,
                        help="Output height (must be divisible by 64). If not specified, auto-detected from input image")
    parser.add_argument("--width", type=int, default=None,
                        help="Output width (must be divisible by 64). If not specified, auto-detected from input image")
    
    # Model Configuration
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "fp16", "fp32"],
                        help="Model precision")
    parser.add_argument("--attn_mode", type=str, default="sdpa",
                        choices=["sdpa", "xformers", "flash", "sageattn"],
                        help="Attention mechanism")
    
    # Optional
    parser.add_argument("--vae_tiling", action="store_true",
                        help="Enable VAE tiling for lower memory")
    parser.add_argument("--vae_chunk_size", type=int, default=None,
                        help="VAE chunk size for CausalConv3d (reduces memory, may improve speed)")
    parser.add_argument("--vae_spatial_tile_sample_min_size", type=int, default=None,
                       help="VAE spatial tile min size (e.g., 256 for tiled decoding)")
    parser.add_argument("--output_format", type=str, default="png",
                        choices=["png", "jpg", "jpeg"],
                        help="Output image format")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    parser.add_argument("--save_params", type=str, default=None,
                        help="CSV file path to save/append generation parameters")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image_path):
        print(f"Error: Input image not found: {args.image_path}")
        return 1
    
    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load input image first to auto-detect dimensions
    if args.verbose:
        print("\n[1/5] Loading input image...")
    
    try:
        input_image = Image.open(args.image_path).convert("RGB")
        img_width, img_height = input_image.size  # PIL returns (width, height)
        if args.verbose:
            print(f"  Image size: {img_width}x{img_height}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return 1
    
    # Auto-detect dimensions from input image if not specified
    if args.height is None:
        # Round down to nearest multiple of 64
        args.height = (img_height // 64) * 64
        if args.verbose:
            print(f"  Auto-detected height: {args.height} (from {img_height})")
    
    if args.width is None:
        # Round down to nearest multiple of 64
        args.width = (img_width // 64) * 64
        if args.verbose:
            print(f"  Auto-detected width: {args.width} (from {img_width})")
    
    # Validate that dimensions are divisible by 64
    if args.height % 64 != 0 or args.width % 64 != 0:
        print(f"Error: Height and width must be divisible by 64. Got {args.height}x{args.width}")
        return 1
    
    if args.verbose:
        print("=" * 60)
        print("FramePack One-Frame Inference")
        print("=" * 60)
        print(f"Input image: {args.image_path}")
        print(f"Output path: {args.output_path}")
        print(f"Prompt: {args.prompt}")
        if args.negative_prompt:
            print(f"Negative prompt: {args.negative_prompt}")
        print(f"Resolution: {args.height}x{args.width}")
        print(f"Target Index: {args.target_index}")
        print(f"Inference steps: {args.infer_steps}")
        print(f"Guidance scale: {args.guidance_scale}")
        print(f"Seed: {args.seed}")
        print(f"Device: {args.device}")
        print(f"Dtype: {args.dtype}")
        print(f"Attention: {args.attn_mode}")
        print("=" * 60)
    
    # Initialize editor
    if args.verbose:
        print("\n[2/5] Initializing models...")
    
    try:
        model_paths = {
            "dit": args.dit,
            "vae": args.vae,
            "text_encoder1": args.text_encoder1,
            "text_encoder2": args.text_encoder2,
            "image_encoder": args.image_encoder
        }
        
        editor = SingleFrameImageEditor(
            model_paths=model_paths,
            device=args.device,
            dtype=args.dtype,
            attn_mode=args.attn_mode,
            vae_chunk_size=args.vae_chunk_size,
            vae_spatial_tile_sample_min_size=args.vae_spatial_tile_sample_min_size,
            vae_tiling=args.vae_tiling
        )
        
        if args.verbose:
            print("  Models initialized successfully")
    except Exception as e:
        print(f"Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Prepare generation config
    if args.verbose:
        print("\n[3/5] Preparing generation configuration...")
    
    config = GenerationConfig(
        image=input_image,
        prompt=args.prompt,
        seed=args.seed,
        inference_steps=args.infer_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        target_index=args.target_index
    )
    
    # Generate
    if args.verbose:
        print("\n[4/5] Generating edited image...")
        print("  (This may take a while depending on your hardware)")
    
    try:
        start_time = time.time()
        
        result = editor.generate(config)
        
        elapsed = time.time() - start_time
        
        if args.verbose:
            print(f"  Generation completed in {elapsed:.2f} seconds")
            if "device_memory_peak" in result:
                memory_gb = result["device_memory_peak"] / (1024**3)
                print(f"  Peak memory usage: {memory_gb:.2f} GB")
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save output
    if args.verbose:
        print(f"\n[5/5] Saving output to {args.output_path}...")
    
    try:
        output_image = result["generated_image"]
        
        # Convert tensor to PIL Image if needed
        if isinstance(output_image, torch.Tensor):
            if args.verbose:
                print(f"  Output tensor shape: {output_image.shape}, dtype: {output_image.dtype}")
            
            # Squeeze out singleton dimensions (batch=1, time=1)
            # Expected shapes: [B, C, T, H, W] or [B, C, H, W] or [C, H, W]
            output_image = output_image.squeeze()
            
            if args.verbose:
                print(f"  After squeeze: {output_image.shape}")
            
            # Now should be [C, H, W]
            if output_image.dim() != 3:
                raise ValueError(f"Unexpected tensor shape after squeeze: {output_image.shape}, expected 3D [C, H, W]")
            
            # Check channels dimension
            if output_image.shape[0] != 3:
                # Maybe channels are in wrong position, try to fix
                if output_image.shape[-1] == 3:
                    # HWC format, transpose to CHW
                    output_image = output_image.permute(2, 0, 1)
                else:
                    raise ValueError(f"Unexpected channels: {output_image.shape}, expected 3 channels in first dim")
            
            # Convert to numpy [C, H, W] -> [H, W, C]
            output_np = output_image.cpu().numpy()
            
            # Check if already uint8 (0-255 range) or float (0-1 range)
            if output_np.dtype == np.uint8:
                # Already in correct format
                output_np = output_np.transpose(1, 2, 0)
            else:
                # Float tensor, needs conversion
                output_np = output_np.transpose(1, 2, 0)
                output_np = (output_np * 255).clip(0, 255).astype("uint8")
            
            output_pil = Image.fromarray(output_np)
        else:
            output_pil = output_image
        
        # Save with appropriate format
        if args.output_format.lower() in ["jpg", "jpeg"]:
            output_pil.save(args.output_path, format="JPEG", quality=95)
        else:
            output_pil.save(args.output_path, format="PNG")
        
        # Save parameters to CSV if requested
        if args.save_params:
            params_csv_path = args.save_params
            params_dict = {
                "output_path": args.output_path,
                "image_path": args.image_path,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "seed": args.seed,
                "infer_steps": args.infer_steps,
                "guidance_scale": args.guidance_scale,
                "height": args.height,
                "width": args.width,
                "target_index": args.target_index,
                "device": args.device,
                "dtype": args.dtype,
                "attn_mode": args.attn_mode,
                "vae_tiling": args.vae_tiling,
                "vae_chunk_size": args.vae_chunk_size,
                "vae_spatial_tile_sample_min_size": args.vae_spatial_tile_sample_min_size,
                "output_format": args.output_format,
            }
            
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(params_csv_path)
            
            with open(params_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=params_dict.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(params_dict)
            
            if args.verbose:
                print(f"  Parameters saved to: {params_csv_path}")
        
        if args.verbose:
            print(f"  Output saved successfully")
            total_elapsed = time.time() - total_start_time
            print("\n" + "=" * 60)
            print("✅ Generation complete!")
            print(f"Total execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
            print("=" * 60)
    except Exception as e:
        print(f"Error saving output: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
