"""
Test Script: Verify Module Imports

Run this script to verify that all modules can be imported correctly.
This doesn't require models to be downloaded - it only tests imports.
"""

import sys
import os

# Add src directory to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Ensure the module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add FramePack to path (required for diffusers_helper imports)
framepack_path = os.environ.get("FRAMEPACK_PATH")
if framepack_path and os.path.exists(framepack_path):
    sys.path.insert(0, framepack_path)
else:
    # Try relative path
    possible_path = os.path.join(os.path.dirname(__file__), "..", "FramePack")
    if os.path.exists(possible_path):
        sys.path.insert(0, possible_path)
    else:
        print(f"Warning: FramePack not found. Set FRAMEPACK_PATH environment variable.")

def test_imports():
    """Test that all modules can be imported"""
    
    print("Testing module imports...")
    print("-" * 50)
    
    try:
        print("✓ Importing framepack_models...")
        from framepack_models import FramePackModels, ModelConfig
        print("  - FramePackModels: OK")
        print("  - ModelConfig: OK")
    except Exception as e:
        print(f"✗ Failed to import framepack_models: {e}")
        return False
    
    try:
        print("✓ Importing conditioning_pipeline...")
        from conditioning_pipeline import (
            TextConditioner, 
            ImageConditioner, 
            NullConditioner,
            simple_bucket_selector
        )
        print("  - TextConditioner: OK")
        print("  - ImageConditioner: OK")
        print("  - NullConditioner: OK")
        print("  - simple_bucket_selector: OK")
    except Exception as e:
        print(f"✗ Failed to import conditioning_pipeline: {e}")
        return False
    
    try:
        print("✓ Importing latent_packing...")
        from latent_packing import LatentIndexManager, ControlMaskHandler
        print("  - LatentIndexManager: OK")
        print("  - ControlMaskHandler: OK")
    except Exception as e:
        print(f"✗ Failed to import latent_packing: {e}")
        return False
    
    try:
        print("✓ Importing inference_engine...")
        from inference_engine import SingleFrameImageEditor, GenerationConfig
        print("  - SingleFrameImageEditor: OK")
        print("  - GenerationConfig: OK")
    except Exception as e:
        print(f"✗ Failed to import inference_engine: {e}")
        return False
    
    try:
        print("✓ Importing package __init__...")
        import framepack_oneframe_inference
        print(f"  - Package version: {framepack_oneframe_inference.__version__}")
        print(f"  - Exported modules: {len(framepack_oneframe_inference.__all__)}")
    except Exception as e:
        print(f"✗ Failed to import package: {e}")
        return False
    
    print("-" * 50)
    print("✓ All imports successful!")
    return True


def test_simple_instantiation():
    """Test basic class instantiation (no model loading)"""
    
    print("\nTesting basic class instantiation...")
    print("-" * 50)
    
    try:
        print("✓ Testing ModelConfig...")
        from framepack_models import ModelConfig
        config = ModelConfig()
        print(f"  - LLaMA hidden size: {config.llama_hidden_size}")
        print(f"  - VAE latent channels: {config.vae_latent_channels}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    try:
        print("✓ Testing LatentIndexManager...")
        from latent_packing import LatentIndexManager
        manager = LatentIndexManager(target_index=9, control_indices=[1, 10])
        print(f"  - Target index: {manager.target_index}")
        print(f"  - Control indices: {manager.control_indices}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    try:
        print("✓ Testing ControlMaskHandler...")
        from latent_packing import ControlMaskHandler
        handler = ControlMaskHandler()
        print("  - ControlMaskHandler instantiated")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    try:
        print("✓ Testing GenerationConfig...")
        from inference_engine import GenerationConfig
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        dummy_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        config = GenerationConfig(
            image=dummy_img,
            prompt="test",
            seed=42
        )
        print(f"  - Prompt: {config.prompt}")
        print(f"  - Seed: {config.seed}")
        print(f"  - Target index: {config.target_index}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("-" * 50)
    print("✓ All instantiation tests successful!")
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("FramePack One-Frame Inference - Module Test")
    print("=" * 50)
    print()
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test instantiation
    if not test_simple_instantiation():
        success = False
    
    print()
    print("=" * 50)
    if success:
        print("✓ ALL TESTS PASSED")
        print("=" * 50)
        print()
        print("Modules are working correctly!")
        print("You can now use them for inference (requires model files).")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 50)
        print()
        print("Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
