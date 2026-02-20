#!/usr/bin/env python3
"""
Simple test to verify FramePack imports
"""

import sys
import os

# Add FramePack to path
framepack_path = os.environ.get("FRAMEPACK_PATH")
if framepack_path:
    sys.path.insert(0, framepack_path)
else:
    # Try relative path
    possible_path = os.path.join(os.path.dirname(__file__), "..", "FramePack")
    if os.path.exists(possible_path):
        sys.path.insert(0, possible_path)

# Add parent directory so we can import framepack-oneframe-inference as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src directory to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

print("Testing FramePack diffusers_helper imports...")

try:
    from diffusers_helper import hunyuan
    print("✓ diffusers_helper.hunyuan")
except ImportError as e:
    print(f"✗ diffusers_helper.hunyuan: {e}")

try:
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    print("✓ diffusers_helper.clip_vision")
except ImportError as e:
    print(f"✗ diffusers_helper.clip_vision: {e}")

try:
    from diffusers_helper.utils import crop_or_pad_yield_mask
    print("✓ diffusers_helper.utils")
except ImportError as e:
    print(f"✗ diffusers_helper.utils: {e}")

try:
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    print("✓ diffusers_helper.models.hunyuan_video_packed")
except ImportError as e:
    print(f"✗ diffusers_helper.models.hunyuan_video_packed: {e}")

try:
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    print("✓ diffusers_helper.pipelines.k_diffusion_hunyuan")
except ImportError as e:
    print(f"✗ diffusers_helper.pipelines.k_diffusion_hunyuan: {e}")

print("\nTesting our package as 'framepack_oneframe_inference'...")

# Rename the directory mentally - we need to import it properly
import importlib.util

# Load as package
package_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')

try:
    # Import the package
    spec = importlib.util.spec_from_file_location("framepack_oneframe_inference", init_path)
    package = importlib.util.module_from_spec(spec)
    sys.modules["framepack_oneframe_inference"] = package
    
    # Now load individual modules within the package context
    for module_name in ["framepack_models", "conditioning_pipeline", "latent_packing", "inference_engine"]:
        module_path = os.path.join(package_dir, f"{module_name}.py")
        module_spec = importlib.util.spec_from_file_location(
            f"framepack_oneframe_inference.{module_name}",
            module_path
        )
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[f"framepack_oneframe_inference.{module_name}"] = module
        module_spec.loader.exec_module(module)
        print(f"✓ {module_name}")
    
    # Now execute the __init__.py
    spec.loader.exec_module(package)
    print("✓ Package imports complete")
    
    # Test that we can access the main classes
    from framepack_oneframe_inference import (
        FramePackModels,
        SingleFrameImageEditor,
        GenerationConfig,
        TextConditioner,
        ImageConditioner,
        NullConditioner,
        LatentIndexManager,
        ControlMaskHandler
    )
    print("\n✅ All main classes accessible:")
    print("  - FramePackModels")
    print("  - SingleFrameImageEditor")
    print("  - GenerationConfig")
    print("  - TextConditioner, ImageConditioner, NullConditioner")
    print("  - LatentIndexManager, ControlMaskHandler")
    
except Exception as e:
    print(f"✗ Package import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All imports completed!")

