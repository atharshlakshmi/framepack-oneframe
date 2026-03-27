"""
Configuration Template for Ablation Study

Copy this file and modify to create custom study configurations.
Then pass to run_inference.py as: --config custom_config.py

NOTE: The protocol requires using FIXED parameters across all conditions.
Only modify these if you're running a variant study!
"""

# Inference Configuration (Protocol Specification - DO NOT CHANGE)
INFERENCE_CONFIG = {
    'seed': 42,                          # Fixed seed for reproducibility
    'infer_steps': 25,                   # Number of diffusion steps
    'guidance_scale': 10.0,              # Classifier-free guidance scale
    'real_guidance_scale': 1.0,          # CFG strength
    'height': 640,                       # Output height (pixels)
    'width': 512,                        # Output width (pixels)
    'dtype': 'bfloat16',                 # Floating point precision
    'attn_mode': 'sdpa',                 # Attention: 'sdpa', 'xformers', 'flash'
    'target_index': 9,                   # Target frame index
    'control_indices': [1, 10],          # Control frame indices
}

# Dataset Configuration
DATASET_CONFIG = {
    'num_samples': 20,                   # Must be <= 5000 (COCO val2017 has 5k images)
    'min_height': 512,                   # Minimum image size (shortest side)
    'min_width': 512,                    # Minimum image size (shortest side)
    'seed': 42,                          # Sampling seed (critical for reproducibility)
    'use_captions': True,                # Use COCO captions as prompts
}

# Metrics Configuration
METRICS_CONFIG = {
    'compute_lpips': True,               # Learned Perceptual Image Patch Similarity
    'compute_dssim': True,               # Structural Dissimilarity (1 - SSIM)
    'compute_huber': True,               # Robust L1 loss
    'compute_entropy': True,             # Shannon entropy
    'compute_variance': True,            # Local patch variance
    'lpips_backend': 'approx',           # 'approx' or 'vgg' (requires lpips library)
}

# Paths Configuration
PATHS_CONFIG = {
    'coco_data_dir': 'coco_data',                    # COCO dataset location
    'ablation_study_dir': 'ablation_study',         # All outputs here
    'images_dir': 'ablation_study/images',          # Sampled images
    'outputs_dir': 'ablation_study/outputs',        # Generated outputs
    'metrics_output': 'ablation_study/metrics.csv', # Results
}

# Conditions to Test
# Format: 'CONDITION_NAME': {'one_frame_flags': {...}}
ABLATION_CONDITIONS = {
    'FULL': {
        'description': 'Full pipeline with multi-scale controls',
        'one_frame_flags': set(),  # Empty = enables 2x and 4x
    },
    'ABL-NO2X': {
        'description': '2x upsampling controls disabled',
        'one_frame_flags': {'no_2x'},
    },
    'ABL-NO4X': {
        'description': '4x upsampling controls disabled',
        'one_frame_flags': {'no_4x'},
    },
    'ABL-NONE': {
        'description': 'Both multi-scale controls disabled',
        'one_frame_flags': {'no_2x', 'no_4x'},
    },
}

# ============================================================================
# VARIANT STUDY CONFIGS (Examples)
# ============================================================================

# Example 1: High Guidance Study
VARIANT_HIGH_GUIDANCE = {
    'inference': INFERENCE_CONFIG.copy(),
    'dataset': DATASET_CONFIG.copy(),
}
VARIANT_HIGH_GUIDANCE['inference']['guidance_scale'] = 15.0


# Example 2: Different Resolution Study
VARIANT_HIGH_RES = {
    'inference': INFERENCE_CONFIG.copy(),
    'dataset': DATASET_CONFIG.copy(),
}
VARIANT_HIGH_RES['inference']['height'] = 768
VARIANT_HIGH_RES['inference']['width'] = 576


# Example 3: Fewer Steps (Speed Study)
VARIANT_SPEED = {
    'inference': INFERENCE_CONFIG.copy(),
    'dataset': DATASET_CONFIG.copy(),
}
VARIANT_SPEED['inference']['infer_steps'] = 15


# ============================================================================
# Usage
# ============================================================================

if __name__ == '__main__':
    print("Configuration loaded")
    print(f"Inference steps: {INFERENCE_CONFIG['infer_steps']}")
    print(f"Sample images: {DATASET_CONFIG['num_samples']}")
    print(f"Conditions: {len(ABLATION_CONDITIONS)}")
