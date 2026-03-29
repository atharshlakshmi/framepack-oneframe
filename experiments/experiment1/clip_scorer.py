"""
clip_scorer.py
==============
Drop-in replacement for raw CLIP loading that uses local file path instead
of triggering a network download (which can fail on networks with proxy/firewall).

This loads CLIP from a local file path, completely bypassing SSL issues.

Usage
-----
1.  Run once to download weights:
        python fix_clip_ssl.py

2.  In helpers.py, replace clip.load() calls:
        # OLD (causes SSL error):
        # model, preprocess = clip.load("ViT-B/32", device="cuda:0")

        # NEW: use CLIPScorer class
        from clip_scorer import CLIPScorer
        scorer = CLIPScorer(device="cuda:0")
        score = scorer.score(image, prompt)
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# ── Locate local weights ─────────────────────────────────────────────────────

_CACHE_DIR  = Path.home() / ".cache" / "clip"
_MODEL_FILE = _CACHE_DIR / "ViT-B-32.pt"


def _find_weights() -> str:
    """
    Look for ViT-B-32.pt in the standard cache location.
    Raises a clear error if not found, pointing to the fix script.
    """
    if _MODEL_FILE.exists():
        return str(_MODEL_FILE)

    # Also accept an env-var override so CI/cluster jobs can point to a
    # shared network path without touching the cache dir.
    env_path = os.environ.get("CLIP_MODEL_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    raise FileNotFoundError(
        f"\n"
        f"CLIP weights not found at {_MODEL_FILE}\n"
        f"\n"
        f"Fix: run this once on your GPU machine:\n"
        f"     python fix_clip_ssl.py\n"
        f"\n"
        f"Or set the environment variable:\n"
        f"     export CLIP_MODEL_PATH=/path/to/ViT-B-32.pt\n"
    )


# ── CLIPScorer ────────────────────────────────────────────────────────────────

class CLIPScorer:
    """
    Computes cosine similarity between a text prompt and an image using
    CLIP ViT-B/32.  Returns a float in [-1, 1]; higher = more aligned.

    Loads weights from the local cache — no network access required.
    This is a drop-in replacement for raw clip.load() calls.
    """

    def __init__(self, device: str = "cuda"):
        import clip  # openai/CLIP — already installed, just broken at download

        weights_path = _find_weights()
        print(f"[CLIPScorer] Loading from local file: {weights_path}")

        # Pass the file path directly — CLIP skips the download when given a path
        self.model, self.preprocess = clip.load(weights_path, device=device)
        self.model.eval()
        self.device    = device
        self.tokenizer = clip.tokenize

    @torch.no_grad()
    def score(self, image: Image.Image, prompt: str) -> float:
        """
        Args:
            image:  PIL Image (RGB). The generated output image.
            prompt: The text prompt used to produce this image.

        Returns:
            float in [-1, 1]. Typical range for good edits is [0.20, 0.35].
        """
        img_tensor  = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer([prompt], truncate=True).to(self.device)

        img_feat  = self.model.encode_image(img_tensor)
        text_feat = self.model.encode_text(text_tokens)

        img_feat  = F.normalize(img_feat,  dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        return float((img_feat * text_feat).sum())
