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


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target bucket resolution.
    
    Upscales/downscales to match the short side, then crops the long side.
    
    Args:
        image: PIL Image or numpy array (HWC)
        bucket_reso: Target (width, height)
        
    Returns:
        Numpy array in HWC format
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]
    
    bucket_width, bucket_height = bucket_reso
    
    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image
    
    # Resize to match short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    
    new_width = int(image_width * scale + 0.5)
    new_height = int(image_height * scale + 0.5)
    
    # Resize image
    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((new_width, new_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        if scale != 1:
            from PIL import Image as PILImage
            image = PILImage.fromarray(image).resize((new_width, new_height), PILImage.LANCZOS)
            image = np.array(image)
    
    # Center crop to bucket size
    if new_width > bucket_width:
        left = (new_width - bucket_width) // 2
        image = image[:, left:left + bucket_width]
    if new_height > bucket_height:
        top = (new_height - bucket_height) // 2
        image = image[top:top + bucket_height, :]
    
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
        
        Args:
            prompt: Text prompt string
            
        Returns:
            Dictionary with:
                - llama_vec: [seq_len, 4096] tensor
                - llama_attention_mask: [seq_len] tensor
                - clip_l_pooler: [768] tensor
        """
        # Check cache first
        if prompt in self._cache:
            llama_vec, clip_l_pooler = self._cache[prompt]
        else:
            # Move encoders to device
            original_device1 = self.text_encoder1.device
            original_device2 = self.text_encoder2.device
            
            self.text_encoder1.to(self.device)
            self.text_encoder2.to(self.device)
            
            # Encode with both encoders
            with torch.autocast(device_type=self.device.type, dtype=self.text_encoder1.dtype), torch.no_grad():
                llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(
                    prompt=prompt,
                    text_encoder=self.text_encoder1,
                    text_encoder_2=self.text_encoder2,
                    tokenizer=self.tokenizer1,
                    tokenizer_2=self.tokenizer2,
                    max_length=256,  # Standard FramePack value
                )
            
            # Move to CPU for caching
            llama_vec = llama_vec.cpu()
            clip_l_pooler = clip_l_pooler.cpu()
            
            # Cache result
            self._cache[prompt] = (llama_vec, clip_l_pooler)
            
            # Move encoders back to original device
            self.text_encoder1.to(original_device1)
            self.text_encoder2.to(original_device2)
        
        # Crop or pad to target sequence length
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
        
        Args:
            image: PIL Image
            height: Target height
            width: Target width
            
        Returns:
            Dictionary with:
                - image_encoder_features: [1, 577, 1152] tensor
                - start_latent: [1, 16, 1, H/8, W/8] tensor
        """
        # Convert to RGB and extract alpha channel if present
        if image.mode == "RGBA":
            alpha = image.split()[-1]
        else:
            alpha = None
        
        image = image.convert("RGB")
        image_np = np.array(image)  # PIL to numpy, HWC
        
        # Resize to target resolution
        image_np = resize_image_to_bucket(image_np, (width, height))
        
        # Prepare tensor for VAE: normalize to [-1, 1] and format as NCFHW
        image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
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
            ctrl_image_np = resize_image_to_bucket(ctrl_image_np, (width, height))
            
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
    Null conditioning for classifier-free guidance.
    
    Generates unconditional embeddings (empty prompt) for CFG.
    Caches the result since it's always the same.
    
    Example:
        conditioner = NullConditioner(models)
        null_embeddings = conditioner()
    """
    
    def __init__(
        self,
        models: FramePackModels,
        max_seq_len: int = 512,
    ):
        """
        Initialize null conditioner.
        
        Args:
            models: FramePackModels instance with loaded text encoders
            max_seq_len: Maximum sequence length for LLaMA (default 512)
        """
        self.models = models
        self.max_seq_len = max_seq_len
        self.device = models.device
        
        # Cache for null embeddings (computed once)
        self._cache: Optional[Dict[str, torch.Tensor]] = None
        
        # Load tokenizers and encoders
        (self.tokenizer1, self.text_encoder1), (self.tokenizer2, self.text_encoder2) = models.load_text_encoders()
    
    def __call__(self) -> Dict[str, torch.Tensor]:
        """
        Get null (unconditional) embeddings.
        
        Returns:
            Dictionary with:
                - llama_vec: [seq_len, 4096] tensor
                - llama_attention_mask: [seq_len] tensor
                - clip_l_pooler: [768] tensor
        """
        if self._cache is not None:
            return self._cache
        
        # Move encoders to device
        original_device1 = self.text_encoder1.device
        original_device2 = self.text_encoder2.device
        
        self.text_encoder1.to(self.device)
        self.text_encoder2.to(self.device)
        
        # Encode empty prompt
        empty_prompt = ""
        with torch.autocast(device_type=self.device.type, dtype=self.text_encoder1.dtype), torch.no_grad():
            llama_vec_n, clip_l_pooler_n = hunyuan.encode_prompt_conds(
                prompt=empty_prompt,
                text_encoder=self.text_encoder1,
                text_encoder_2=self.text_encoder2,
                tokenizer=self.tokenizer1,
                tokenizer_2=self.tokenizer2,
                max_length=256,
            )
        
        # Move to CPU for caching
        llama_vec_n = llama_vec_n.cpu()
        clip_l_pooler_n = clip_l_pooler_n.cpu()
        
        # Move encoders back to original device
        self.text_encoder1.to(original_device1)
        self.text_encoder2.to(original_device2)
        
        # Crop or pad to target sequence length
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=self.max_seq_len)
        
        # Cache result
        self._cache = {
            "llama_vec": llama_vec_n,
            "llama_attention_mask": llama_attention_mask_n,
            "clip_l_pooler": clip_l_pooler_n,
        }
        
        return self._cache
    
    def clear_cache(self):
        """Clear the null embeddings cache"""
        self._cache = None
        logger.info("NullConditioner cache cleared")


def simple_bucket_selector(width: int, height: int) -> Tuple[int, int]:
    """
    Simple resolution bucket selector.
    
    Snaps input resolution to nearest valid bucket.
    FramePack supports buckets with dimensions divisible by 64.
    
    Args:
        width: Desired width
        height: Desired height
        
    Returns:
        Tuple of (height, width) rounded to nearest bucket
    """
    # Common FramePack buckets (height, width)
    common_buckets = [
        (256, 256), (384, 256), (512, 256), (640, 256),
        (256, 384), (384, 384), (512, 384), (640, 384),
        (256, 512), (384, 512), (512, 512), (640, 512), (768, 512),
        (256, 640), (384, 640), (512, 640), (640, 640), (768, 640),
        (256, 768), (384, 768), (512, 768), (640, 768), (768, 768),
    ]
    
    # Find closest bucket by area
    target_area = width * height
    closest_bucket = min(common_buckets, key=lambda b: abs(b[0] * b[1] - target_area))
    
    return closest_bucket
