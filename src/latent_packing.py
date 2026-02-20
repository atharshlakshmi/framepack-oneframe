"""
Latent Packing Module

This module handles latent tensor construction and index mappings for one-frame generation:
- LatentIndexManager: Index logic and packing
- ControlMaskHandler: Mask loading and application
"""

import logging
from typing import List, Dict, Optional, Tuple, Set

import torch
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class LatentIndexManager:
    """
    Manages latent indices and packing for one-frame inference.
    
    Handles the complex index logic for packed latent sequences in FramePack's
    one-frame generation mode, including control images, multi-scale controls,
    and target frame positioning.
    
    Example:
        manager = LatentIndexManager(target_index=9, control_indices=[1, 10])
        indices = manager.compute_indices(device="cuda")
        packed_latents = manager.pack_control_latents(control_list, masks)
    """
    
    def __init__(
        self,
        latent_window_size: int = 9,
        target_index: int = 9,
        control_indices: Optional[List[int]] = None,
        flags: Optional[Set[str]] = None,
    ):
        """
        Initialize latent index manager.
        
        Args:
            latent_window_size: FramePack internal frame buffer size (default 9)
            target_index: Frame index to generate (default 9, at end of window)
            control_indices: Indices for control/reference frames (default [1, 10])
            flags: Set of flags like "no_2x", "no_4x", "no_post"
        """
        self.latent_window_size = latent_window_size
        self.target_index = target_index
        self.control_indices = control_indices if control_indices is not None else [1, 1 + latent_window_size]
        self.flags = flags if flags is not None else set()
        
        # Validate indices
        if target_index < 0:
            raise ValueError(f"target_index must be >= 0, got {target_index}")
        
        logger.info(f"LatentIndexManager initialized: window_size={latent_window_size}, "
                   f"target_index={target_index}, control_indices={self.control_indices}, "
                   f"flags={self.flags}")
    
    def compute_indices(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute all index tensors for packed latents.
        
        Args:
            device: Device to place tensors on
            
        Returns:
            Dictionary with:
                - latent_indices: [1, 1] tensor for target frame
                - clean_latent_indices: [1, N] tensor for control frames
                - clean_latent_2x_indices: [1, 2] or None
                - clean_latent_4x_indices: [1, 16] or None
        """
        # Target frame index
        latent_indices = torch.zeros((1, 1), dtype=torch.int64, device=device)
        latent_indices[:, 0] = self.target_index
        
        # Control frame indices
        num_control = len(self.control_indices)
        clean_latent_indices = torch.zeros((1, num_control), dtype=torch.int64, device=device)
        for i, idx in enumerate(self.control_indices):
            clean_latent_indices[:, i] = idx
        
        # 2x upsampling control indices
        if "no_2x" not in self.flags:
            index_start = 1 + self.latent_window_size + 1
            clean_latent_2x_indices = torch.arange(
                index_start, index_start + 2,
                dtype=torch.int64, device=device
            ).unsqueeze(0)
        else:
            clean_latent_2x_indices = None
        
        # 4x upsampling control indices
        if "no_4x" not in self.flags:
            index_start = 1 + self.latent_window_size + 1 + 2
            clean_latent_4x_indices = torch.arange(
                index_start, index_start + 16,
                dtype=torch.int64, device=device
            ).unsqueeze(0)
        else:
            clean_latent_4x_indices = None
        
        return {
            "latent_indices": latent_indices,
            "clean_latent_indices": clean_latent_indices,
            "clean_latent_2x_indices": clean_latent_2x_indices,
            "clean_latent_4x_indices": clean_latent_4x_indices,
        }
    
    def pack_control_latents(
        self,
        control_list: List[torch.Tensor],
        height: int,
        width: int,
        masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Pack control latents with optional masking.
        
        Args:
            control_list: List of control latent tensors, each [1, 16, 1, H/8, W/8]
            height: Image height (for zero latent generation if needed)
            width: Image width (for zero latent generation if needed)
            masks: Optional list of mask tensors [1, 1, 1, H/8, W/8]
            
        Returns:
            Packed control latents [1, 16, N_ctrl, H/8, W/8]
        """
        if not control_list or len(control_list) == 0:
            # No control images provided, use zero latents
            logger.info("No control images provided. Using zero latents.")
            control_list = [torch.zeros(1, 16, 1, height // 8, width // 8, dtype=torch.float32)]
        
        # Add clean latent post if needed
        if "no_post" not in self.flags:
            control_list.append(torch.zeros((1, 16, 1, height // 8, width // 8), dtype=torch.float32))
            logger.info("Added zero latents as clean latents post")
        
        # Concatenate control latents
        clean_latents = torch.cat(control_list, dim=2)  # [1, 16, N_ctrl, H/8, W/8]
        
        # Apply masks if provided
        if masks is not None:
            for i, mask in enumerate(masks):
                if mask is not None and i < clean_latents.shape[2]:
                    clean_latents[:, :, i:i+1, :, :] = clean_latents[:, :, i:i+1, :, :] * mask
                    logger.info(f"Applied mask to control latent {i}")
        
        logger.info(f"Packed control latents shape: {clean_latents.shape}")
        return clean_latents
    
    def get_clean_latents_2x(self, height: int, width: int) -> Optional[torch.Tensor]:
        """
        Generate 2x upsampling control latents (zeros) or None if disabled.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            [1, 16, 2, H/8, W/8] tensor or None
        """
        if "no_2x" in self.flags:
            return None
        
        return torch.zeros((1, 16, 2, height // 8, width // 8), dtype=torch.float32)
    
    def get_clean_latents_4x(self, height: int, width: int) -> Optional[torch.Tensor]:
        """
        Generate 4x upsampling control latents (zeros) or None if disabled.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            [1, 16, 16, H/8, W/8] tensor or None
        """
        if "no_4x" in self.flags:
            return None
        
        return torch.zeros((1, 16, 16, height // 8, width // 8), dtype=torch.float32)


class ControlMaskHandler:
    """
    Handles loading and applying masks to control latents.
    
    Supports grayscale masks and alpha channels from control images.
    Masks are resized to latent space resolution (H/8, W/8).
    
    Example:
        handler = ControlMaskHandler()
        mask_tensor = handler.load_mask("mask.png", height=640, width=512)
        masked_latent = handler.apply_mask(control_latent, mask_tensor)
    """
    
    def __init__(self):
        """Initialize control mask handler"""
        pass
    
    def load_mask(
        self,
        mask_source: Image.Image,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Load and process mask image to latent space dimensions.
        
        Args:
            mask_source: PIL Image (grayscale or alpha channel)
            height: Target image height
            width: Target image width
            
        Returns:
            Mask tensor [1, 1, 1, H/8, W/8] with values in [0, 1]
        """
        # Convert to grayscale if needed
        if mask_source.mode != "L":
            mask_source = mask_source.convert("L")
        
        # Resize to latent space dimensions
        mask_resized = mask_source.resize((width // 8, height // 8), Image.LANCZOS)
        
        # Convert to tensor
        mask_np = np.array(mask_resized).astype(np.float32) / 255.0  # 0 to 1.0
        mask_tensor = torch.from_numpy(mask_np)
        
        # Add dimensions: HW -> 111HW (BCFHW format)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        return mask_tensor
    
    def load_mask_from_path(
        self,
        mask_path: str,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Load mask from file path.
        
        Args:
            mask_path: Path to mask image file
            height: Target image height
            width: Target image width
            
        Returns:
            Mask tensor [1, 1, 1, H/8, W/8]
        """
        mask_image = Image.open(mask_path)
        return self.load_mask(mask_image, height, width)
    
    def apply_mask(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mask to latent tensor.
        
        Args:
            latent: Latent tensor [1, 16, 1, H/8, W/8]
            mask: Mask tensor [1, 1, 1, H/8, W/8]
            
        Returns:
            Masked latent tensor [1, 16, 1, H/8, W/8]
        """
        return latent * mask
    
    def load_and_apply(
        self,
        latent: torch.Tensor,
        mask_source: Image.Image,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Convenience method to load and apply mask in one step.
        
        Args:
            latent: Latent tensor [1, 16, 1, H/8, W/8]
            mask_source: PIL Image mask
            height: Target image height
            width: Target image width
            
        Returns:
            Masked latent tensor
        """
        mask = self.load_mask(mask_source, height, width)
        return self.apply_mask(latent, mask)
    
    def process_control_masks(
        self,
        control_mask_images: List[Optional[Image.Image]],
        height: int,
        width: int,
    ) -> List[Optional[torch.Tensor]]:
        """
        Process a list of mask images for multiple control latents.
        
        Args:
            control_mask_images: List of PIL Images (grayscale or alpha)
            height: Target image height
            width: Target image width
            
        Returns:
            List of mask tensors [1, 1, 1, H/8, W/8] or None
        """
        mask_tensors = []
        for mask_img in control_mask_images:
            if mask_img is not None:
                mask_tensor = self.load_mask(mask_img, height, width)
                mask_tensors.append(mask_tensor)
            else:
                mask_tensors.append(None)
        
        return mask_tensors
