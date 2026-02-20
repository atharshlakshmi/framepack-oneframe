"""
FramePack One-Frame Inference

Independent modules for single-frame image editing using FramePack.

Modules:
- framepack_models: Model loading and management
- conditioning_pipeline: Text and image conditioning
- latent_packing: Latent index management and packing
- inference_engine: End-to-end inference orchestration
"""

from .framepack_models import FramePackModels, ModelConfig
from .conditioning_pipeline import TextConditioner, ImageConditioner, NullConditioner, simple_bucket_selector
from .latent_packing import LatentIndexManager, ControlMaskHandler
from .inference_engine import SingleFrameImageEditor, GenerationConfig

__version__ = "0.1.0"
__all__ = [
    "FramePackModels",
    "ModelConfig",
    "TextConditioner",
    "ImageConditioner",
    "NullConditioner",
    "simple_bucket_selector",
    "LatentIndexManager",
    "ControlMaskHandler",
    "SingleFrameImageEditor",
    "GenerationConfig",
]
