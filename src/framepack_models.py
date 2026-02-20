"""
FramePack Models Module

This module handles loading and caching of all FramePack models:
- DiT (Diffusion Transformer)
- VAE (Variational Autoencoder)  
- Text Encoders (LLaMA + CLIP-L)
- Image Encoder (SiglipVision)

"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple, Any
import glob

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaTokenizerFast,
    LlamaConfig,
    LlamaModel,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPConfig,
    SiglipImageProcessor,
    SiglipVisionModel,
    SiglipVisionConfig,
)

# Import from FramePack diffusers_helper only
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ModelConfig:
    """Configuration constants for FramePack models"""
    
    # Text encoder output dimensions
    llama_hidden_size: int = 4096
    clip_pooler_size: int = 768
    
    # VAE parameters
    vae_scaling_factor: float = 0.18215
    vae_latent_channels: int = 16
    vae_spatial_compression: int = 8
    vae_temporal_compression: int = 4
    
    # DiT parameters
    dit_input_channels: int = 16
    dit_attention_head_dim: int = 128
    dit_num_attention_heads: int = 24
    
    # Image encoder
    image_encoder_hidden_size: int = 1152
    image_encoder_num_features: int = 577  # 384x384 / 14x14 + 1
    
    # LLaMA config (from HunyuanVideo)
    LLAMA_CONFIG = {
        "architectures": ["LlamaModel"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.46.3",
        "use_cache": True,
        "vocab_size": 128320,
    }
    
    # CLIP config (from HunyuanVideo)
    CLIP_CONFIG = {
        "architectures": ["CLIPTextModel"],
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "dropout": 0.0,
        "eos_token_id": 2,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 77,
        "model_type": "clip_text_model",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 1,
        "projection_dim": 768,
        "torch_dtype": "float16",
        "transformers_version": "4.48.0.dev0",
        "vocab_size": 49408,
    }
    
    # SiglipVision config (from FramePack/FLUX)
    SIGLIP_FEATURE_EXTRACTOR_CONFIG = {
        "do_convert_rgb": None,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_processor_type": "SiglipImageProcessor",
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "SiglipProcessor",
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"height": 384, "width": 384},
    }
    
    SIGLIP_CONFIG = {
        "architectures": ["SiglipVisionModel"],
        "attention_dropout": 0.0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "layer_norm_eps": 1e-06,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 27,
        "patch_size": 14,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.46.2",
    }


class FramePackModels:
    """
    Singleton-pattern loader for all FramePack models.
    
    Handles lazy-loading, caching, and precision conversions for:
    - DiT (Diffusion Transformer)
    - VAE (3D Causal VAE)
    - Text Encoder 1 (LLaMA)
    - Text Encoder 2 (CLIP-L)
    - Image Encoder (SiglipVision)
    
    Example:
        models = FramePackModels(model_paths, device="cuda")
        dit = models.get("dit")
        vae = models.get("vae")
    """
    
    def __init__(
        self,
        model_paths: Dict[str, str],
        device: Union[str, torch.device] = "cuda",
        dtype: str = "bfloat16",
        attn_mode: str = "sdpa",
        vae_chunk_size: Optional[int] = None,
        vae_spatial_tile_sample_min_size: Optional[int] = None,
        vae_tiling: bool = False,
        fp8_dit: bool = False,
        fp8_scaled: bool = False,
        fp8_llm: bool = False,
        disable_numpy_memmap: bool = False,
    ):
        """
        Initialize FramePackModels loader.
        
        Args:
            model_paths: Dictionary with keys "dit", "vae", "text_encoder1", 
                        "text_encoder2", "image_encoder"
            device: Device to load models on
            dtype: Model precision ("bfloat16", "fp16", "fp32")
            attn_mode: Attention mechanism ("sdpa", "xformers", "flash", "sageattn")
            vae_chunk_size: Chunk size for CausalConv3d in VAE
            vae_spatial_tile_sample_min_size: Min tile size for spatial tiling
            vae_tiling: Enable VAE spatial tiling
            fp8_dit: Use FP8 for DiT model
            fp8_scaled: Use scaled FP8 for DiT
            fp8_llm: Use FP8 for LLaMA text encoder
            disable_numpy_memmap: Disable numpy memmap when loading safetensors
        """
        self.model_paths = model_paths
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.attn_mode = attn_mode
        self.vae_chunk_size = vae_chunk_size
        self.vae_spatial_tile_sample_min_size = vae_spatial_tile_sample_min_size
        self.vae_tiling = vae_tiling
        self.fp8_dit = fp8_dit
        self.fp8_scaled = fp8_scaled
        self.fp8_llm = fp8_llm
        self.disable_numpy_memmap = disable_numpy_memmap
        
        # Model config
        self.config = ModelConfig()
        
        # Cache for loaded models
        self._cache: Dict[str, Any] = {}
        
        # Validate required model paths
        required_keys = ["dit", "vae", "text_encoder1", "text_encoder2", "image_encoder"]
        for key in required_keys:
            if key not in model_paths:
                raise ValueError(f"Missing required model path: {key}")
    
    def _str_to_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str.lower(), torch.bfloat16)
    
    def _load_safetensors_with_splits(self, path: str) -> Dict[str, torch.Tensor]:
        """Load safetensors file, supporting split files (e.g., model-00001-of-00002.safetensors)"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Check if it's a split file pattern
        if os.path.isfile(path):
            # Single file
            return load_file(path)
        
        # Directory - look for split files
        split_pattern = os.path.join(path, "*.safetensors")
        split_files = sorted(glob.glob(split_pattern))
        
        if not split_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")
        
        # Load and merge all splits
        state_dict = {}
        for split_file in split_files:
            logger.info(f"Loading split file: {split_file}")
            split_dict = load_file(split_file)
            state_dict.update(split_dict)
        
        return state_dict
    
    def load_dit(self, force_reload: bool = False) -> HunyuanVideoTransformer3DModelPacked:
        """
        Load DiT (Diffusion Transformer) model.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            HunyuanVideoTransformer3DModelPacked instance
        """
        if "dit" in self._cache and not force_reload:
            return self._cache["dit"]
        
        logger.info(f"Loading DiT model from {self.model_paths['dit']}")
        
        dit_path = self.model_paths["dit"]
        target_dtype = self._str_to_torch_dtype(self.dtype)
        
        # Try loading from HuggingFace hub or local directory first
        if os.path.isdir(dit_path) and os.path.exists(os.path.join(dit_path, "config.json")):
            logger.info("Loading from directory with config.json")
            model = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                dit_path,
                torch_dtype=target_dtype
            )
        else:
            # Load from safetensors file
            logger.info("Loading from safetensors file")
            
            # Use init_empty_weights for faster loading (no actual tensors allocated)
            with init_empty_weights():
                # Initialize model with default FramePack config
                model = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                    'lllyasviel/FramePack_F1_I2V_HY_20250503',
                    torch_dtype=target_dtype
                )
            
            # Load weights from file
            logger.info(f"Loading DiT weights from {os.path.basename(dit_path)} (24GB)...")
            state_dict = self._load_safetensors_with_splits(dit_path)
            logger.info("Loading weights into model...")
            # Use assign=True to replace tensors directly instead of copying (much faster)
            model.load_state_dict(state_dict, strict=True, assign=True)
            logger.info("Weights loaded successfully")
        
        # Configure attention mode
        if hasattr(model, 'set_attn_mode'):
            model.set_attn_mode(self.attn_mode)
        
        # Enable high quality FP32 output for inference
        if hasattr(model, 'high_quality_fp32_output_for_inference'):
            model.high_quality_fp32_output_for_inference = True
        
        logger.info(f"Moving DiT model to {self.device} (this may take 1-2 minutes for 24GB model)...")
        model.to(self.device, dtype=target_dtype)
        logger.info("DiT model moved to device successfully")
        model.eval()
        model.requires_grad_(False)
        
        self._cache["dit"] = model
        self._clear_cuda_cache()
        
        logger.info(f"DiT model loaded successfully")
        return model
    
    def load_vae(self, force_reload: bool = False) -> AutoencoderKLHunyuanVideo:
        """
        Load VAE (Variational Autoencoder) model.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            AutoencoderKLHunyuanVideo instance
        """
        if "vae" in self._cache and not force_reload:
            return self._cache["vae"]
        
        logger.info(f"Loading VAE from {self.model_paths['vae']}")
        
        vae_path = self.model_paths["vae"]
        vae_dtype = torch.float16
        
        # Try loading from HuggingFace directory structure
        if os.path.isdir(vae_path) and os.path.exists(os.path.join(vae_path, "vae", "config.json")):
            logger.info("Loading VAE from directory with HuggingFace structure")
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                vae_path,
                subfolder="vae",
                torch_dtype=vae_dtype
            )
        elif os.path.isdir(vae_path) and os.path.exists(os.path.join(vae_path, "config.json")):
            logger.info("Loading VAE from directory")
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                vae_path,
                torch_dtype=vae_dtype
            )
        else:
            # Load from local file by first loading default model, then loading weights
            logger.info("Loading VAE from HuggingFace hub with local weights")
            vae = AutoencoderKLHunyuanVideo.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder="vae",
                torch_dtype=vae_dtype
            )
            
            # Load weights if provided as file
            if os.path.isfile(vae_path):
                logger.info(f"Loading VAE weights from {vae_path}")
                state_dict = torch.load(vae_path, map_location="cpu")
                vae.load_state_dict(state_dict, strict=True)
        
        # Enable slicing and tiling for memory efficiency
        if self.vae_tiling or self.vae_spatial_tile_sample_min_size is not None:
            vae.enable_slicing()
            vae.enable_tiling()
            logger.info("Enabled VAE slicing and tiling")
        
        vae.to(self.device, dtype=vae_dtype)
        vae.eval()
        vae.requires_grad_(False)
        
        self._cache["vae"] = vae
        self._clear_cuda_cache()
        
        logger.info(f"VAE loaded successfully")
        return vae
    
    def load_text_encoders(self, force_reload: bool = False) -> Tuple[
        Tuple[LlamaTokenizerFast, LlamaModel],
        Tuple[CLIPTokenizer, CLIPTextModel]
    ]:
        """
        Load both text encoders (LLaMA and CLIP).
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of ((llama_tokenizer, llama_model), (clip_tokenizer, clip_model))
        """
        if "text_encoder1" in self._cache and "text_encoder2" in self._cache and not force_reload:
            return (self._cache["text_encoder1"], self._cache["text_encoder2"])
        
        # Load LLaMA (Text Encoder 1)
        logger.info("Loading text encoder 1 (LLaMA)")
        tokenizer1 = LlamaTokenizerFast.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder="tokenizer"
        )
        
        te1_path = self.model_paths["text_encoder1"]
        if os.path.isdir(te1_path):
            text_encoder1 = LlamaModel.from_pretrained(
                te1_path,
                subfolder="text_encoder",
                torch_dtype=torch.float16
            )
        else:
            # Load from single file
            config = LlamaConfig(**self.config.LLAMA_CONFIG)
            with init_empty_weights():
                text_encoder1 = LlamaModel._from_config(config, torch_dtype=torch.float16)
            
            state_dict = self._load_safetensors_with_splits(te1_path)
            
            # Support weights from ComfyUI
            if "model.embed_tokens.weight" in state_dict:
                for key in list(state_dict.keys()):
                    if key.startswith("model."):
                        new_key = key.replace("model.", "")
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]
            if "tokenizer" in state_dict:
                state_dict.pop("tokenizer")
            if "lm_head.weight" in state_dict:
                state_dict.pop("lm_head.weight")
            
            text_encoder1.load_state_dict(state_dict, strict=True, assign=True)
        
        # Handle FP8 for LLaMA if requested
        if self.fp8_llm:
            org_dtype = text_encoder1.dtype
            logger.info(f"Moving and casting text encoder 1 to {self.device} and torch.float8_e4m3fn")
            text_encoder1.to(device=self.device, dtype=torch.float8_e4m3fn)
            
            # Prepare LLM for FP8
            self._prepare_llm_fp8(text_encoder1, org_dtype)
        else:
            text_encoder1.to(self.device)
        
        text_encoder1.eval()
        
        # Load CLIP (Text Encoder 2)
        logger.info("Loading text encoder 2 (CLIP)")
        tokenizer2 = CLIPTokenizer.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder="tokenizer_2"
        )
        
        te2_path = self.model_paths["text_encoder2"]
        if os.path.isdir(te2_path):
            text_encoder2 = CLIPTextModel.from_pretrained(
                te2_path,
                subfolder="text_encoder_2",
                torch_dtype=torch.float16
            )
        else:
            config = CLIPConfig(**self.config.CLIP_CONFIG)
            with init_empty_weights():
                text_encoder2 = CLIPTextModel._from_config(config, torch_dtype=torch.float16)
            
            state_dict = load_file(te2_path)
            text_encoder2.load_state_dict(state_dict, strict=True, assign=True)
        
        text_encoder2.to(self.device)
        text_encoder2.eval()
        
        self._cache["text_encoder1"] = (tokenizer1, text_encoder1)
        self._cache["text_encoder2"] = (tokenizer2, text_encoder2)
        self._clear_cuda_cache()
        
        logger.info("Text encoders loaded successfully")
        return (self._cache["text_encoder1"], self._cache["text_encoder2"])
    
    def load_image_encoder(self, force_reload: bool = False) -> Tuple[SiglipImageProcessor, SiglipVisionModel]:
        """
        Load image encoder (SiglipVision).
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (feature_extractor, image_encoder)
        """
        if "image_encoder" in self._cache and not force_reload:
            return self._cache["image_encoder"]
        
        logger.info("Loading image encoder (SiglipVision)")
        
        # Load feature extractor
        feature_extractor = SiglipImageProcessor(**self.config.SIGLIP_FEATURE_EXTRACTOR_CONFIG)
        
        # Load image encoder
        ie_path = self.model_paths["image_encoder"]
        if os.path.isdir(ie_path):
            image_encoder = SiglipVisionModel.from_pretrained(
                ie_path,
                subfolder="image_encoder",
                torch_dtype=torch.float16
            )
        else:
            config = SiglipVisionConfig(**self.config.SIGLIP_CONFIG)
            with init_empty_weights():
                image_encoder = SiglipVisionModel._from_config(config, torch_dtype=torch.float16)
            
            state_dict = load_file(ie_path)
            image_encoder.load_state_dict(state_dict, strict=True, assign=True)
        
        image_encoder.to(self.device)
        image_encoder.eval()
        
        self._cache["image_encoder"] = (feature_extractor, image_encoder)
        self._clear_cuda_cache()
        
        logger.info("Image encoder loaded successfully")
        return self._cache["image_encoder"]
    
    def get(self, name: str) -> Any:
        """
        Get a model by name, loading it if not cached.
        
        Args:
            name: Model name ("dit", "vae", "text_encoder1", "text_encoder2", "image_encoder")
            
        Returns:
            Loaded model
        """
        if name == "dit":
            return self.load_dit()
        elif name == "vae":
            return self.load_vae()
        elif name == "text_encoder1":
            te1, _ = self.load_text_encoders()
            return te1
        elif name == "text_encoder2":
            _, te2 = self.load_text_encoders()
            return te2
        elif name == "image_encoder":
            return self.load_image_encoder()
        else:
            raise ValueError(f"Unknown model name: {name}. Valid names: dit, vae, text_encoder1, text_encoder2, image_encoder")
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name is None:
            logger.info("Clearing all cached models")
            self._cache.clear()
        elif model_name in self._cache:
            logger.info(f"Clearing cached model: {model_name}")
            del self._cache[model_name]
        
        self._clear_cuda_cache()
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache to free memory"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def _prepare_llm_fp8(self, llama_model: LlamaModel, target_dtype: torch.dtype):
        """
        Prepare LLaMA model for FP8 inference.
        
        Args:
            llama_model: LLaMA model instance
            target_dtype: Target dtype for specific layers
        """
        def forward_hook(module):
            def forward(hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                return module.weight.to(input_dtype) * hidden_states.to(input_dtype)
            return forward
        
        for module in llama_model.modules():
            if module.__class__.__name__ in ["Embedding"]:
                module.to(target_dtype)
            if module.__class__.__name__ in ["LlamaRMSNorm"]:
                module.forward = forward_hook(module)
