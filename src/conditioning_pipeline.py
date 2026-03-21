"""
Conditioning Pipeline Module

This module handles preparation of image and text inputs into model-ready embeddings/latents:
- TextConditioner: LLaMA + CLIP-L text encoding
- ImageConditioner: Image VAE encoding + SiglipVision features
- NullConditioner: Unconditional embeddings for classifier-free guidance

"""

import logging
from typing import Dict, Tuple, Optional, Union
from functools import lru_cache
import hashlib

import torch
import numpy as np
import cv2
from PIL import Image

# Import from FramePack diffusers_helper only
from diffusers_helper import hunyuan
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.utils import crop_or_pad_yield_mask

# Import model types for type hints
from transformers import LlamaTokenizerFast, LlamaModel, CLIPTokenizer, CLIPTextModel
from transformers import SiglipImageProcessor, SiglipVisionModel

try:
    from .framepack_models import FramePackModels
except ImportError:
    from framepack_models import FramePackModels


logger = logging.getLogger(__name__)


def resize_image_to_bucket(
    image: Union[Image.Image, np.ndarray],
    bucket_reso: Tuple[int, int],
) -> np.ndarray:
    """
    Resize image to exactly fill bucket_reso (width, height) via scale-then-center-crop.
    Uses cv2.INTER_AREA for downsampling (better quality) and PIL LANCZOS for upsampling.
    Ported from musubi-tuner's implementation in image_video_dataset.py.
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    bucket_width, bucket_height = bucket_reso
    
    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    # Compute scale factor to cover the target resolution
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    
    # Compute resized dimensions with proper rounding
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    # Use appropriate resampling method based on scale direction
    if scale > 1:
        # Upsampling: use LANCZOS
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        # Downsampling: use cv2.INTER_AREA for better quality
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # Center-crop to target resolution using integer division
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image


class TextConditioner:
    """
    Text conditioning pipeline using LLaMA and CLIP-L encoders.
    
    Encodes text prompts into embeddings suitable for FramePack inference.
    Includes caching to avoid re-encoding the same prompts.
    
    Example:
        conditioner = TextConditioner(models)
        embeddings = conditioner("The girl dances gracefully")
    """
    
    def __init__(
        self,
        models: FramePackModels,
        max_seq_len: int = 512,
    ):
        """
        Initialize text conditioner.
        
        Args:
            models: FramePackModels instance with loaded text encoders
            max_seq_len: Maximum sequence length for LLaMA (default 512)
        """
        self.models = models
        self.max_seq_len = max_seq_len
        self.device = models.device
        
        # Cache for prompt encodings
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # Load tokenizers and encoders
        (self.tokenizer1, self.text_encoder1), (self.tokenizer2, self.text_encoder2) = models.load_text_encoders()
    
    def __call__(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Encode text prompt into LLaMA and CLIP embeddings.
        
        Dual-encoder approach:
        - LLaMA (4096-dim): Rich semantic understanding for video generation
        - CLIP-L (768-dim pooled): Aligned visual-semantic space for guidance
        
        Args:
            prompt: Text prompt string
            
        Returns:
            Dictionary with:
                - llama_vec: [seq_len, 4096] tensor
                - llama_attention_mask: [seq_len] tensor
                - clip_l_pooler: [768] tensor
        """
        # Check cache first to avoid redundant encoding of same prompts
        if prompt in self._cache:
            llama_vec, clip_l_pooler = self._cache[prompt]
        else:
            # Save original device locations to restore later (text encoders often on CPU)
            original_device1 = self.text_encoder1.device
            original_device2 = self.text_encoder2.device
            
            # Move encoders to computation device temporarily
            self.text_encoder1.to(self.device)
            self.text_encoder2.to(self.device)
            
            # Encode with both encoders using autocast for memory efficiency
            with torch.autocast(device_type=self.device.type, dtype=self.text_encoder1.dtype), torch.no_grad():
                llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(
                    prompt=prompt,
                    text_encoder=self.text_encoder1,
                    text_encoder_2=self.text_encoder2,
                    tokenizer=self.tokenizer1,
                    tokenizer_2=self.tokenizer2,
                    max_length=256,  # Standard FramePack value (full prompt semantic richness)
                )
            
            # Move embeddings to CPU for caching (save GPU memory)
            llama_vec = llama_vec.cpu()
            clip_l_pooler = clip_l_pooler.cpu()
            
            # Cache result to avoid re-encoding if same prompt appears again
            self._cache[prompt] = (llama_vec, clip_l_pooler)
            
            # Move encoders back to original device (usually CPU to save VRAM)
            self.text_encoder1.to(original_device1)
            self.text_encoder2.to(original_device2)
        
        # Ensure sequence length matches model expectation (crop excess or pad with zeros)
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=self.max_seq_len)
        
        return {
            "llama_vec": llama_vec,
            "llama_attention_mask": llama_attention_mask,
            "clip_l_pooler": clip_l_pooler,
        }
    
    def clear_cache(self):
        """Clear the prompt encoding cache"""
        self._cache.clear()
        logger.info("TextConditioner cache cleared")


class ImageConditioner:
    """
    Image conditioning pipeline using VAE encoding and SiglipVision features.
    
    Processes images into latent space and extracts CLIP-vision features.
    Handles image resizing to match resolution buckets.
    
    Example:
        conditioner = ImageConditioner(models)
        result = conditioner(pil_image, height=640, width=512)
    """
    
    def __init__(
        self,
        models: FramePackModels,
    ):
        """
        Initialize image conditioner.
        
        Args:
            models: FramePackModels instance with VAE and image encoder
        """
        self.models = models
        self.device = models.device
        
        # Load VAE and image encoder
        self.vae = models.load_vae()
        self.feature_extractor, self.image_encoder = models.load_image_encoder()
    
    def __call__(
        self,
        image: Image.Image,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode image into VAE latent and SiglipVision features.
        
        Two-stream processing:
        - VAE encoding: Compresses image to latent space (16 channels, 1/8 resolution)
        - Vision encoding: Extracts semantic features (1152-dim) for spatial guidance
        
        Args:
            image: PIL Image
            height: Target height
            width: Target width
            
        Returns:
            Dictionary with:
                - image_encoder_features: [1, 577, 1152] tensor (spatial features)
                - start_latent: [1, 16, 1, H/8, W/8] tensor (compressed representation)
        """
        # Handle RGBA by extracting alpha channel for later mask processing
        if image.mode == "RGBA":
            alpha = image.split()[-1]
        else:
            alpha = None
        
        # Convert to RGB for processing (VAE and encoders expect 3 channels)
        image = image.convert("RGB")
        image_np = np.array(image)  # PIL to numpy, HWC format
        
        # Resize to exact resolution using scale-then-center-crop with quality optimization
        # Uses cv2.INTER_AREA for downsampling (better quality) - Fix 3
        image_np = resize_image_to_bucket(image_np, (width, height))
        
        # Prepare tensor for VAE: normalize uint8 [0,255] to float [-1, 1]
        # Formula: x / 127.5 - 1.0 maps [0,255] -> [-1,1] (VAE's expected input range)
        image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
        # Rearrange tensor: HWC -> CHW -> NCFHW (batch=1, channels=3, frames=1)
        image_tensor = image_tensor.permute(2, 0, 1)[None, :, None]  # HWC -> CHW -> NCFHW, N=1, C=3, F=1
        
        # Extract SiglipVision features
        original_image_encoder_device = self.image_encoder.device
        self.image_encoder.to(self.device)
        
        with torch.no_grad():
            image_encoder_output = hf_clip_vision_encode(image_np, self.feature_extractor, self.image_encoder)
        
        image_encoder_features = image_encoder_output.last_hidden_state.cpu()
        
        # Move image encoder back to original device
        self.image_encoder.to(original_image_encoder_device)
        
        # VAE encode to latent space
        original_vae_device = self.vae.device
        self.vae.to(self.device)
        
        with torch.no_grad():
            start_latent = hunyuan.vae_encode(image_tensor.to(self.device), self.vae).cpu()
        
        # Move VAE back to original device
        self.vae.to(original_vae_device)
        
        # Clear CUDA cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "image_encoder_features": image_encoder_features,
            "start_latent": start_latent,
        }
    
    def encode_control_images(
        self,
        control_images: list[Image.Image],
        height: int,
        width: int,
    ) -> Tuple[list[torch.Tensor], list[Optional[Image.Image]]]:
        """
        Encode multiple control images for one-frame inference.
        
        Args:
            control_images: List of PIL Images
            height: Target height
            width: Target width
            
        Returns:
            Tuple of (control_latents, control_mask_images)
                - control_latents: List of [1, 16, 1, H/8, W/8] tensors
                - control_mask_images: List of PIL Image alpha channels (or None)
        """
        control_latents = []
        control_mask_images = []
        
        original_vae_device = self.vae.device
        self.vae.to(self.device)
        
        for ctrl_image in control_images:
            # Extract alpha channel if present
            if ctrl_image.mode == "RGBA":
                alpha = ctrl_image.split()[-1]
            else:
                alpha = None
            
            ctrl_image = ctrl_image.convert("RGB")
            ctrl_image_np = np.array(ctrl_image)
            
            # Resize to target resolution
            bucket_h, bucket_w = simple_bucket_selector(width, height)
            ctrl_image_np = resize_image_to_bucket(ctrl_image_np, (bucket_w, bucket_h))
            
            # Prepare tensor for VAE
            ctrl_image_tensor = torch.from_numpy(ctrl_image_np).float() / 127.5 - 1.0
            ctrl_image_tensor = ctrl_image_tensor.permute(2, 0, 1)[None, :, None]  # NCFHW
            
            # VAE encode
            with torch.no_grad():
                ctrl_latent = hunyuan.vae_encode(ctrl_image_tensor.to(self.device), self.vae).cpu()
            
            control_latents.append(ctrl_latent)
            control_mask_images.append(alpha)
        
        # Move VAE back to original device
        self.vae.to(original_vae_device)
        
        # Clear CUDA cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        return control_latents, control_mask_images


class NullConditioner:
    """
    Returns zero embeddings when guidance_scale == 1.0 (the distilled model default).
    fm_wrapper (wrapper.py line 39) already replaces the negative prediction with zeros
    at cfg_scale == 1.0, so encoding a prompt here is pure wasted compute.
    Only encodes a real negative prompt when guidance_scale > 1.0.
    """

    def __init__(self, models: FramePackModels, max_seq_len: int = 512):
        self.models = models
        self.max_seq_len = max_seq_len
        self.device = models.device
        self._cache: Optional[Dict[str, torch.Tensor]] = None
        (self.tokenizer1, self.text_encoder1), (self.tokenizer2, self.text_encoder2) = (
            models.load_text_encoders()
        )

    def __call__(
        self,
        positive_llama_vec: torch.Tensor,
        positive_clip_pooler: torch.Tensor,
        guidance_scale: float = 1.0,
        negative_prompt: str = "",
    ) -> Dict[str, torch.Tensor]:
        if guidance_scale == 1.0:
            # fm_wrapper discards these anyway — return correctly-shaped zeros
            return {
                "llama_vec": torch.zeros_like(positive_llama_vec),
                "llama_attention_mask": torch.zeros(
                    positive_llama_vec.shape[:2], dtype=torch.bool
                ),
                "clip_l_pooler": torch.zeros_like(positive_clip_pooler),
            }

        # guidance_scale > 1.0: encode the negative prompt (cached after first call)
        if self._cache is not None:
            return self._cache

        original_device1 = self.text_encoder1.device
        original_device2 = self.text_encoder2.device
        self.text_encoder1.to(self.device)
        self.text_encoder2.to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=self.text_encoder1.dtype), torch.no_grad():
            llama_vec_n, clip_l_pooler_n = hunyuan.encode_prompt_conds(
                prompt=negative_prompt,
                text_encoder=self.text_encoder1,
                text_encoder_2=self.text_encoder2,
                tokenizer=self.tokenizer1,
                tokenizer_2=self.tokenizer2,
                max_length=256,
            )

        llama_vec_n = llama_vec_n.cpu()
        clip_l_pooler_n = clip_l_pooler_n.cpu()
        self.text_encoder1.to(original_device1)
        self.text_encoder2.to(original_device2)

        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=self.max_seq_len
        )

        self._cache = {
            "llama_vec": llama_vec_n,
            "llama_attention_mask": llama_attention_mask_n,
            "clip_l_pooler": clip_l_pooler_n,
        }
        return self._cache

    def clear_cache(self):
        self._cache = None


# Copied verbatim from the original FramePack bucket_tools.py
_BUCKET_OPTIONS_640 = [
    (416, 960), (448, 864), (480, 832), (512, 768),
    (544, 704), (576, 672), (608, 640), (640, 608),
    (672, 576), (704, 544), (768, 512), (832, 480),
    (864, 448), (960, 416),
]


def simple_bucket_selector(width: int, height: int) -> Tuple[int, int]:
    """
    Find the nearest resolution bucket by aspect ratio.
    Uses cross-multiplication (same metric as FramePack bucket_tools.py).
    Returns (bucket_height, bucket_width).
    
    Why cross-multiplication instead of area matching?
    Cross-multiplication (abs(h1*w2 - w1*h2)) preserves aspect ratio,
    not just close to the pixel area. This prevents picking a square bucket
    for a 3:1 wide image when a 960x416 aspect-matching bucket exists.
    """
    min_metric = float("inf")
    best_bucket = None
    # Iterate through predefined buckets to find best aspect ratio match
    for bucket_h, bucket_w in _BUCKET_OPTIONS_640:
        # Cross-multiplication metric: abs(h_img * w_bucket - w_img * h_bucket)
        # Minimizing this gives the closest aspect ratio (Fix 2)
        metric = abs(height * bucket_w - width * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket
