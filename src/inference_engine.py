"""
Inference Engine Module

This module provides the main orchestration for single-frame image editing inference:
- SingleFrameImageEditor: End-to-end generation pipeline
- Handles model loading, conditioning, latent packing, diffusion, and decoding

"""

import logging
import time
from typing import Dict, Optional, List, Set, Any
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

# Import from FramePack diffusers_helper only
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper import hunyuan

# Import our modules
try:
    # Try relative import (when used as package)
    from .framepack_models import FramePackModels
    from .conditioning_pipeline import TextConditioner, ImageConditioner, NullConditioner
    from .latent_packing import LatentIndexManager, ControlMaskHandler
except ImportError:
    # Fall back to absolute import (when run as script)
    from framepack_models import FramePackModels
    from conditioning_pipeline import TextConditioner, ImageConditioner, NullConditioner
    from latent_packing import LatentIndexManager, ControlMaskHandler


logger = logging.getLogger(__name__)


class MagCacheWrapper:
    """
    Accelerates DiT inference by skipping redundant forward passes (Fix 6).
    
    MagCache: Magnitude-based Cache
    - Monitors transformer output magnitude between diffusion timesteps
    - Skips forward pass when output magnitude change is below threshold
    - Gives 30-50% speedup with minimal quality loss
    
    How it works:
    1. Track ratio of output magnitudes: mag[t] / mag[t-1]
    2. Accumulate error if ratio drifts from expected value
    3. Reuse cached output from previous step if error stays small
    4. Reset and compute full forward pass if error exceeds threshold
    
    Threshold parameters:
    - retention_ratio: Skip only after 20% of steps (warmup period)
    - magcache_thresh: Max accumulated error before reset (0.24)
    - K: Max consecutive skips before forced recompute (6 steps)
    """

    # Magnitude ratios for 50 denoising steps (calibrated from reference model)
    # ratio[i] = ||output[i]|| / ||output[i-1]|| - expected output scaling
    # These values tell us how much the output magnitude changes at each step
    _MAG_RATIOS_50 = np.array([
        1.0,     1.06971, 1.29073, 1.11245, 1.09596, 1.05233, 1.01415, 1.05672,
        1.00848, 1.03632, 1.02974, 1.00984, 1.03028, 1.00681, 1.06614, 1.05022,
        1.02592, 1.01776, 1.02985, 1.00726, 1.03727, 1.01502, 1.00992, 1.03371,
        0.9976,  1.02742, 1.0093,  1.01869, 1.00815, 1.01461, 1.01152, 1.03082,
        1.0061,  1.02162, 1.01999, 0.99063, 1.01186, 1.0217,  0.99947, 1.01711,
        0.9904,  1.00258, 1.00878, 0.97039, 0.97686, 0.94315, 0.97728, 0.91154,
        0.86139, 0.76592,
    ])

    def __init__(
        self,
        transformer,
        num_steps: int = 25,
        retention_ratio: float = 0.2,
        magcache_thresh: float = 0.24,
        K: int = 6,
    ):
        self.transformer = transformer
        self.num_steps = num_steps
        self.retention_ratio = retention_ratio
        self.magcache_thresh = magcache_thresh
        self.K = K
        self._reset()

    def _reset(self):
        self.cnt = 0
        self.output_cache = None
        self.accumulated_ratio = 1.0
        self.accumulated_steps = 0
        self.accumulated_err = 0.0

    def _mag_ratios(self) -> np.ndarray:
        """Nearest-neighbour interpolate _MAG_RATIOS_50 to self.num_steps."""
        src = self._MAG_RATIOS_50
        n = self.num_steps
        if n == len(src):
            return src
        if n == 1:
            return np.array([src[-1]])
        indices = np.round(np.linspace(0, len(src) - 1, n)).astype(int)
        return src[indices]

    def __call__(self, *args, **kwargs):
        # Get magnitude ratios interpolated to current num_steps
        mag_ratios = self._mag_ratios()
        # Warm-up period: don't skip in first 20% of steps (gradual denoising)
        min_steps = max(int(self.retention_ratio * self.num_steps), 1)

        skip = False
        # Decide whether to reuse cached output or compute new forward pass
        if (
            self.output_cache is not None  # Cache exists from previous step
            and self.cnt >= min_steps  # Past warm-up period
            and self.cnt < self.num_steps - 1  # Not the final step
        ):
            # Update magnitude ratio based on expected value at this step
            self.accumulated_ratio *= float(mag_ratios[self.cnt])
            # Track error: how much actual differs from expected
            self.accumulated_err += abs(1.0 - self.accumulated_ratio)
            self.accumulated_steps += 1
            
            # Skip logic: if error stays small and we haven't skipped too many steps
            if (
                self.accumulated_err <= self.magcache_thresh  # Error within budget
                and self.accumulated_steps <= self.K  # Haven't skipped more than K steps
            ):
                skip = True  # Reuse cached output from previous step (Fix 6)
            else:
                # Threshold exceeded or too many consecutive skips — reset and recompute
                self.accumulated_ratio = 1.0
                self.accumulated_steps = 0
                self.accumulated_err = 0.0

        if skip:
            # Return cached output without computing
            self.cnt += 1
            return self.output_cache

        # Full forward pass through the transformer
        result = self.transformer(*args, **kwargs)
        self.output_cache = result  # Cache for potential reuse next step

        # Reset accumulator after computing a new forward pass
        self.accumulated_ratio = 1.0
        self.accumulated_steps = 0
        self.accumulated_err = 0.0
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self._reset()  # Reset for next generation
        return result

    def __getattr__(self, name: str):
        # Proxy all attribute access to the wrapped transformer so model.device,
        # model.config etc. continue to work transparently
        return getattr(self.transformer, name)


@dataclass
class GenerationConfig:
    """Configuration for single-frame generation"""
    image: Image.Image
    prompt: str
    seed: int
    inference_steps: int = 25
    guidance_scale: float = 10.0  # Distilled CFG scale
    real_guidance_scale: float = 1.0  # Standard CFG scale (should stay at 1.0 for FramePack)
    guidance_rescale: float = 0.0
    height: int = 640
    width: int = 512
    target_index: int = 9
    control_indices: Optional[List[int]] = None
    one_frame_flags: Optional[Set[str]] = None
    flow_shift: Optional[float] = None
    
    def __post_init__(self):
        if self.control_indices is None:
            self.control_indices = [1, 10]
        if self.one_frame_flags is None:
            self.one_frame_flags = {"no_2x", "no_4x"}


class SingleFrameImageEditor:
    """
    Main inference orchestrator for single-frame image editing.
    
    Provides end-to-end pipeline from image + prompt to generated image.
    Handles model loading, conditioning, latent packing, diffusion, and decoding.
    
    Example:
        editor = SingleFrameImageEditor(model_paths, device="cuda")
        config = GenerationConfig(
            image=pil_image,
            prompt="girl in school uniform",
            seed=42
        )
        result = editor.generate(config)
    """
    
    def __init__(
        self,
        model_paths: Dict[str, str],
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_mode: str = "sdpa",
        vae_chunk_size: Optional[int] = None,
        vae_spatial_tile_sample_min_size: Optional[int] = None,
        vae_tiling: bool = False,
        fp8_scaled: bool = False,
        fp8_llm: bool = False,
    ):
        """
        Initialize single-frame image editor.
        
        Args:
            model_paths: Dictionary with model paths
            device: Target device
            dtype: Model precision
            attn_mode: Attention mechanism
            vae_chunk_size: VAE chunk size for memory efficiency
            vae_spatial_tile_sample_min_size: VAE tiling min size
            vae_tiling: Enable VAE tiling
            fp8_scaled: Use scaled FP8 for DiT
            fp8_llm: Use FP8 for LLaMA
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        
        # Initialize models
        logger.info("Initializing FramePack models...")
        self.models = FramePackModels(
            model_paths=model_paths,
            device=self.device,
            dtype=dtype,
            attn_mode=attn_mode,
            vae_chunk_size=vae_chunk_size,
            vae_spatial_tile_sample_min_size=vae_spatial_tile_sample_min_size,
            vae_tiling=vae_tiling,
            fp8_scaled=fp8_scaled,
            fp8_llm=fp8_llm,
        )
        
        # Initialize conditioning pipelines (lazy loading)
        self.text_conditioner: Optional[TextConditioner] = None
        self.image_conditioner: Optional[ImageConditioner] = None
        self.null_conditioner: Optional[NullConditioner] = None
        
        # Mask handler
        self.mask_handler = ControlMaskHandler()
        
        # Preload models at initialization for faster first generation
        self._preload_models()
        
        logger.info("SingleFrameImageEditor initialized and ready")
    
    def _preload_models(self):
        """Preload all models at initialization to avoid delays during generation"""
        # Load text encoders
        logger.info("Preloading text encoders...")
        self.models.load_text_encoders()
        
        # Load and configure VAE
        logger.info("Preloading VAE...")
        vae = self.models.load_vae()
        self._configure_vae(vae)
        
        # Load image encoder
        logger.info("Preloading image encoder...")
        self.models.load_image_encoder()
        
        # Note: DiT is large (24GB), load on-demand during first generate() call
        logger.info("Models preloaded (DiT will load on first generation)")
    
    def _configure_vae(self, vae):
        """Configure VAE optimizations for faster decoding"""
        try:
            # Try to enable spatial tiling if available (reduces memory and can speed up decoding)
            if self.vae_spatial_tile_sample_min_size is not None:
                if hasattr(vae, 'enable_tiling'):
                    vae.enable_tiling()
                    logger.info(f"Enabled VAE tiling with min size {self.vae_spatial_tile_sample_min_size}")
                elif hasattr(vae, 'enable_spatial_tiling'):
                    vae.enable_spatial_tiling(True)
                    if hasattr(vae, 'tile_sample_min_size'):
                        vae.tile_sample_min_size = self.vae_spatial_tile_sample_min_size
                        vae.tile_latent_min_size = self.vae_spatial_tile_sample_min_size // 8
                    logger.info(f"Enabled VAE spatial tiling with min size {self.vae_spatial_tile_sample_min_size}")
            elif self.vae_tiling:
                if hasattr(vae, 'enable_tiling'):
                    vae.enable_tiling()
                    logger.info("Enabled VAE tiling")
                elif hasattr(vae, 'enable_spatial_tiling'):
                    vae.enable_spatial_tiling(True)
                    logger.info("Enabled VAE spatial tiling")
            
            # Set chunk size if available and specified
            if self.vae_chunk_size is not None:
                if hasattr(vae, 'set_chunk_size_for_causal_conv_3d'):
                    vae.set_chunk_size_for_causal_conv_3d(self.vae_chunk_size)
                    logger.info(f"Set VAE chunk size to {self.vae_chunk_size}")
        except Exception as e:
            logger.warning(f"Could not configure VAE optimizations: {e}")
    
    def _ensure_conditioners(self):
        """Lazy load conditioners"""
        if self.text_conditioner is None:
            self.text_conditioner = TextConditioner(self.models)
        if self.image_conditioner is None:
            self.image_conditioner = ImageConditioner(self.models)
        if self.null_conditioner is None:
            self.null_conditioner = NullConditioner(self.models)
    
    def generate(self, config: GenerationConfig) -> Dict[str, Any]:
        """
        Generate a single edited frame from image and prompt.
        
        Args:
            config: GenerationConfig with all parameters
            
        Returns:
            Dictionary with:
                - generated_latent: [1, 16, 1, H/8, W/8]
                - generated_image: [1, 3, H, W] uint8 tensor
                - generation_time: float (seconds)
                - device_memory_peak: int (bytes)
        """
        start_time = time.time()
        
        # Ensure all conditioners are loaded
        self._ensure_conditioners()
        
        # Step 1: Parse inputs and validate
        logger.info("Preparing generation...")
        logger.info(f"Image size: {config.image.size}")
        logger.info(f"Target resolution: {config.height}x{config.width}")
        logger.info(f"Prompt: {config.prompt}")
        logger.info(f"Seed: {config.seed}")
        
        # Step 2: Prepare text conditioning
        logger.info("Encoding text prompt...")
        text_embeddings = self.text_conditioner(config.prompt)
        null_embeddings = self.null_conditioner(
            positive_llama_vec=text_embeddings["llama_vec"],
            positive_clip_pooler=text_embeddings["clip_l_pooler"],
            guidance_scale=config.real_guidance_scale,
        )
        
        # Step 3: Prepare image conditioning
        logger.info("Encoding image...")
        image_embeddings = self.image_conditioner(
            config.image,
            height=config.height,
            width=config.width
        )
        
        # Step 4: Set up latent packing
        logger.info("Setting up latent packing...")
        latent_manager = LatentIndexManager(
            latent_window_size=9,  # Standard FramePack value
            target_index=config.target_index,
            control_indices=config.control_indices,
            flags=config.one_frame_flags,
        )
        
        # Compute indices
        indices = latent_manager.compute_indices(self.device)
        
        # Pack control latents (use start_latent as control)
        control_latents = [image_embeddings["start_latent"]]
        clean_latents = latent_manager.pack_control_latents(
            control_latents,
            height=config.height,
            width=config.width,
        )
        
        # Get multi-scale controls
        clean_latents_2x = latent_manager.get_clean_latents_2x(config.height, config.width)
        clean_latents_4x = latent_manager.get_clean_latents_4x(config.height, config.width)
        
        # Step 5: Load DiT model
        logger.info("Loading DiT model...")
        dit_model = self.models.load_dit()
        dit_model = MagCacheWrapper(
            dit_model,
            num_steps=config.inference_steps,
            retention_ratio=0.2,
            magcache_thresh=0.24,
            K=6,
        )
        # Model is already on device and in eval mode from load_dit()
        logger.info("DiT model ready for inference")
        
        # Step 6: Run diffusion sampling
        logger.info(f"Running diffusion sampling ({config.inference_steps} steps)...")
        
        # Set up generator for reproducibility
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        
        # Convert dtype string to torch dtype
        torch_dtype = self.models._str_to_torch_dtype(self.dtype)
        
        # Prepare embeddings for sampling
        llama_vec = text_embeddings["llama_vec"].to(self.device, dtype=torch_dtype)
        llama_attention_mask = text_embeddings["llama_attention_mask"].to(self.device)
        clip_l_pooler = text_embeddings["clip_l_pooler"].to(self.device, dtype=torch_dtype)
        
        llama_vec_n = null_embeddings["llama_vec"].to(self.device, dtype=torch_dtype)
        llama_attention_mask_n = null_embeddings["llama_attention_mask"].to(self.device)
        clip_l_pooler_n = null_embeddings["clip_l_pooler"].to(self.device, dtype=torch_dtype)
        
        image_encoder_features = image_embeddings["image_encoder_features"].to(self.device, dtype=torch_dtype)
        
        # Move packed latents to device
        clean_latents = clean_latents.to(self.device)
        if clean_latents_2x is not None:
            clean_latents_2x = clean_latents_2x.to(self.device)
        if clean_latents_4x is not None:
            clean_latents_4x = clean_latents_4x.to(self.device)
        
        # Call sample_hunyuan
        generated_latents = sample_hunyuan(
            transformer=dit_model,
            sampler="unipc",
            width=config.width,
            height=config.height,
            frames=1,  # Single frame
            real_guidance_scale=config.real_guidance_scale,
            distilled_guidance_scale=config.guidance_scale,
            guidance_rescale=config.guidance_rescale,
            shift=config.flow_shift,
            num_inference_steps=config.inference_steps,
            generator=generator,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=self.device,
            dtype=torch_dtype,
            image_embeddings=image_encoder_features,
            latent_indices=indices["latent_indices"],
            clean_latents=clean_latents,
            clean_latent_indices=indices["clean_latent_indices"],
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=indices["clean_latent_2x_indices"],
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=indices["clean_latent_4x_indices"],
        )
        
        # Step 7: VAE decode to pixel space
        logger.info("Decoding latent to image...")
        vae = self.models.load_vae()  # From cache, already configured
        vae.eval()
        
        with torch.no_grad():
            # For one-frame inference, decode frame-by-frame for better performance
            if generated_latents.shape[2] == 1:
                # Single frame, decode directly
                decoded_image = hunyuan.vae_decode(
                    generated_latents.to(self.device),
                    vae,
                    image_mode=True
                )
            else:
                # Multiple frames, decode each separately and concatenate
                decoded_frames = []
                for i in range(generated_latents.shape[2]):
                    frame_latent = generated_latents[:, :, i:i+1, :, :]
                    decoded_frame = hunyuan.vae_decode(
                        frame_latent.to(self.device),
                        vae,
                        image_mode=True
                    )
                    decoded_frames.append(decoded_frame.cpu())
                decoded_image = torch.cat(decoded_frames, dim=2).to(self.device)
        
        # Step 8: Post-process
        # Convert from [-1, 1] to [0, 255] uint8
        decoded_image = (decoded_image * 0.5 + 0.5).clamp(0, 1)
        decoded_image = ((decoded_image * 255).round()).to(torch.uint8).cpu()
        
        # Calculate metrics
        generation_time = time.time() - start_time
        device_memory_peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        
        # Clear CUDA cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info(f"Generation completed in {generation_time:.2f}s")
        logger.info(f"Peak memory usage: {device_memory_peak / 1024**3:.2f} GB")
        
        return {
            "generated_latent": generated_latents.cpu(),
            "generated_image": decoded_image,
            "generation_time": generation_time,
            "device_memory_peak": device_memory_peak,
        }
    
    def generate_with_control_images(
        self,
        config: GenerationConfig,
        control_images: List[Image.Image],
        control_masks: Optional[List[Optional[Image.Image]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate with multiple control images (for kisekaeichi or 1f-mc methods).
        
        Args:
            config: GenerationConfig with all parameters
            control_images: List of control/reference images
            control_masks: Optional list of mask images
            
        Returns:
            Same as generate()
        """
        # Ensure conditioners are loaded
        self._ensure_conditioners()
        
        logger.info(f"Preparing generation with {len(control_images)} control images...")
        
        # Encode control images
        control_latents, alpha_masks = self.image_conditioner.encode_control_images(
            control_images,
            height=config.height,
            width=config.width
        )
        
        # Process masks
        mask_tensors = None
        if control_masks is not None or any(alpha_masks):
            # Combine explicit masks with alpha channel masks
            combined_masks = []
            for i in range(len(control_images)):
                if control_masks is not None and i < len(control_masks) and control_masks[i] is not None:
                    mask_img = control_masks[i]
                elif i < len(alpha_masks) and alpha_masks[i] is not None:
                    mask_img = alpha_masks[i]
                else:
                    mask_img = None
                
                if mask_img is not None:
                    mask_tensor = self.mask_handler.load_mask(mask_img, config.height, config.width)
                    combined_masks.append(mask_tensor)
                else:
                    combined_masks.append(None)
            
            mask_tensors = combined_masks
        
        # Set up latent packing with control images
        latent_manager = LatentIndexManager(
            latent_window_size=9,
            target_index=config.target_index,
            control_indices=config.control_indices,
            flags=config.one_frame_flags,
        )
        
        indices = latent_manager.compute_indices(self.device)
        
        clean_latents = latent_manager.pack_control_latents(
            control_latents,
            height=config.height,
            width=config.width,
            masks=mask_tensors,
        )
        
        clean_latents_2x = latent_manager.get_clean_latents_2x(config.height, config.width)
        clean_latents_4x = latent_manager.get_clean_latents_4x(config.height, config.width)
        
        # Continue with standard generation pipeline
        # (Similar to generate() but using control latents)
        
        # Load models and prepare embeddings
        text_embeddings = self.text_conditioner(config.prompt)
        null_embeddings = self.null_conditioner(
            positive_llama_vec=text_embeddings["llama_vec"],
            positive_clip_pooler=text_embeddings["clip_l_pooler"],
            guidance_scale=config.real_guidance_scale,
        )
        
        # Use first control image for image encoder features
        image_embeddings = self.image_conditioner(
            control_images[0],
            height=config.height,
            width=config.width
        )
        
        dit_model = self.models.load_dit()
        dit_model = MagCacheWrapper(
            dit_model,
            num_steps=config.inference_steps,
            retention_ratio=0.2,
            magcache_thresh=0.24,
            K=6,
        )
        
        # Run sampling
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        torch_dtype = self.models._str_to_torch_dtype(self.dtype)
        
        # Prepare embeddings
        llama_vec = text_embeddings["llama_vec"].to(self.device, dtype=torch_dtype)
        llama_attention_mask = text_embeddings["llama_attention_mask"].to(self.device)
        clip_l_pooler = text_embeddings["clip_l_pooler"].to(self.device, dtype=torch_dtype)
        
        llama_vec_n = null_embeddings["llama_vec"].to(self.device, dtype=torch_dtype)
        llama_attention_mask_n = null_embeddings["llama_attention_mask"].to(self.device)
        clip_l_pooler_n = null_embeddings["clip_l_pooler"].to(self.device, dtype=torch_dtype)
        
        image_encoder_features = image_embeddings["image_encoder_features"].to(self.device, dtype=torch_dtype)
        
        clean_latents = clean_latents.to(self.device)
        if clean_latents_2x is not None:
            clean_latents_2x = clean_latents_2x.to(self.device)
        if clean_latents_4x is not None:
            clean_latents_4x = clean_latents_4x.to(self.device)
        
        start_time = time.time()
        
        generated_latents = sample_hunyuan(
            transformer=dit_model,
            sampler="unipc",
            width=config.width,
            height=config.height,
            frames=1,
            real_guidance_scale=config.real_guidance_scale,
            distilled_guidance_scale=config.guidance_scale,
            guidance_rescale=config.guidance_rescale,
            shift=config.flow_shift,
            num_inference_steps=config.inference_steps,
            generator=generator,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=self.device,
            dtype=torch_dtype,
            image_embeddings=image_encoder_features,
            latent_indices=indices["latent_indices"],
            clean_latents=clean_latents,
            clean_latent_indices=indices["clean_latent_indices"],
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=indices["clean_latent_2x_indices"],
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=indices["clean_latent_4x_indices"],
        )
        
        # Decode
        vae = self.models.load_vae()
        vae.to(self.device)
        vae.eval()
        
        with torch.no_grad():
            decoded_image = hunyuan.vae_decode(
                generated_latents.to(self.device),
                vae,
                image_mode=True
            )
        
        decoded_image = (decoded_image * 0.5 + 0.5).clamp(0, 1)
        decoded_image = ((decoded_image * 255).round()).to(torch.uint8).cpu()
        
        generation_time = time.time() - start_time
        device_memory_peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        return {
            "generated_latent": generated_latents.cpu(),
            "generated_image": decoded_image,
            "generation_time": generation_time,
            "device_memory_peak": device_memory_peak,
        }
    
    def clear_cache(self):
        """Clear all model and conditioning caches"""
        logger.info("Clearing all caches...")
        self.models.clear_cache()
        if self.text_conditioner is not None:
            self.text_conditioner.clear_cache()
        if self.null_conditioner is not None:
            self.null_conditioner.clear_cache()
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Caches cleared")
