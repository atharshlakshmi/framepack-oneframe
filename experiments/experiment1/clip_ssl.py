#!/usr/bin/env python3
"""
clip_ssl.py
===========
Complete CLIP setup: installs CLIP and downloads weights to ~/.cache/clip/
bypassing SSL certificate errors entirely.

This is a one-time setup that:
1. Installs CLIP from GitHub (if not already installed)
2. Downloads CLIP weights via wget --no-check-certificate (bypasses SSL)
3. Verifies SHA-256 checksum

Usage
-----
    python clip_ssl.py                    # Install + download ViT-B/32 (default)
    python clip_ssl.py --model ViT-B/16   # Use ViT-B/16 instead
    python clip_ssl.py --verify-only      # Just check if already set up
    python clip_ssl.py --clip-only        # Only install CLIP, skip weights
"""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


# ── Model registry ─────────────────────────────────────────────────────────
# Source: https://github.com/openai/CLIP/blob/main/clip/clip.py
# These are the stable Azure CDN URLs — they do not rotate.
CLIP_MODELS = {
    "ViT-B/32": {
        "url":      "https://openaipublic.azureedge.net/clip/models/"
                    "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
                    "ViT-B-32.pt",
        "filename": "ViT-B-32.pt",
        "md5":      "40d365715913c9da98579312b702a82c",   # first 32 hex chars of SHA256
        "sha256":   "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        "size_mb":  354,
    },
    "ViT-B/16": {
        "url":      "https://openaipublic.azureedge.net/clip/models/"
                    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/"
                    "ViT-B-16.pt",
        "filename": "ViT-B-16.pt",
        "sha256":   "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
        "size_mb":  335,
    },
    "ViT-L/14": {
        "url":      "https://openaipublic.azureedge.net/clip/models/"
                    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/"
                    "ViT-L-14.pt",
        "filename": "ViT-L-14.pt",
        "sha256":   "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        "size_mb":  890,
    },
}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "clip"


# ── Step 1: Install CLIP ───────────────────────────────────────────────────

def install_clip():
    """Install CLIP from GitHub if not already installed."""
    print("=" * 70)
    print("STEP 1: Install CLIP")
    print("=" * 70)
    
    # Check if CLIP is already installed
    try:
        import clip
        print("✓ CLIP is already installed")
        print(f"  Location: {clip.__file__}")
        return True
    except ImportError:
        pass
    
    print("Installing CLIP from GitHub...")
    print("  (This may take 1-2 minutes on first run due to compilation)")
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--quiet",
        "git+https://github.com/openai/CLIP.git"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("✓ CLIP installed successfully")
        return True
    else:
        print("✗ CLIP installation failed")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Try: pip install --upgrade pip setuptools")
        print("  - Manual install: pip install git+https://github.com/openai/CLIP.git")
        return False


# ── Step 2: Download Weights ───────────────────────────────────────────────


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_looks_valid(path: Path, expected_sha256: str) -> bool:
    """Check if file exists, is large enough, and SHA-256 matches."""
    if not path.exists():
        return False
    if path.stat().st_size < 1_000_000:  # < 1 MB → truncated
        return False
    
    print(f"  Verifying SHA-256 of {path.name}...", end=" ", flush=True)
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        print(f"[FAIL] Expected {expected_sha256[:16]}…, got {actual[:16]}…")
        return False
    print("[OK]")
    return True


def download_with_wget(url: str, dest: Path) -> bool:
    """Download via wget with SSL disabled."""
    cmd = [
        "wget",
        "--no-check-certificate",  # Bypass self-signed certs
        "--quiet",
        "--show-progress",
        "--tries=3",
        "--timeout=120",
        "-O", str(dest),
        url,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def download_with_curl(url: str, dest: Path) -> bool:
    """Fallback to curl with SSL disabled."""
    cmd = [
        "curl",
        "-k",  # Skip SSL verification
        "--retry", "3",
        "--connect-timeout", "120",
        "-L",
        "--progress-bar",
        "-o", str(dest),
        url,
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def download_clip_weights(model_name: str = "ViT-B/32", out_path: Path | None = None) -> bool:
    """Download and verify CLIP weights."""
    print("\n" + "=" * 70)
    print(f"STEP 2: Download CLIP Weights ({model_name})")
    print("=" * 70)
    
    info = CLIP_MODELS[model_name]
    dest = out_path or (DEFAULT_CACHE_DIR / info["filename"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already valid
    if file_looks_valid(dest, info["sha256"]):
        print(f"✓ {model_name} weights already present and valid")
        print(f"  Location: {dest}")
        return True
    
    if dest.exists():
        print(f"⚠ {dest.name} exists but failed validation — re-downloading...")
        dest.unlink()
    
    print(f"Downloading {model_name} ({info['size_mb']} MB)...")
    print(f"  From: {info['url']}")
    print(f"  To:   {dest}")
    
    # Try wget first, fall back to curl
    ok = download_with_wget(info["url"], dest)
    if not ok:
        print("  wget failed, trying curl...")
        ok = download_with_curl(info["url"], dest)
    
    if not ok:
        print("✗ Download failed with both wget and curl")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Try manual download on unrestricted network")
        return False
    
    # Verify after download
    if not file_looks_valid(dest, info["sha256"]):
        print("✗ Download completed but SHA-256 verification failed")
        print("  The file may be corrupt. Deleting and retrying...")
        dest.unlink(missing_ok=True)
        return False
    
    print(f"✓ {model_name} weights downloaded and verified")
    print(f"  Location: {dest}")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Complete CLIP setup: install + download weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clip_ssl.py                    # Full setup (ViT-B/32)
  python clip_ssl.py --model ViT-B/16   # Use better ViT-B/16 model
  python clip_ssl.py --verify-only      # Just verify existing setup
        """
    )
    p.add_argument(
        "--model", default="ViT-B/32",
        choices=list(CLIP_MODELS.keys()),
        help="CLIP model to use (default: ViT-B/32)"
    )
    p.add_argument(
        "--verify-only", action="store_true",
        help="Only verify if CLIP is installed and weights exist (no download)",
    )
    p.add_argument(
        "--clip-only", action="store_true",
        help="Only install CLIP, skip weight download",
    )
    args = p.parse_args()
    
    print("\n" + "=" * 70)
    print("CLIP Complete Setup")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Cache: {DEFAULT_CACHE_DIR}")
    print()
    
    # Verify-only mode
    if args.verify_only:
        try:
            import clip
            print("✓ CLIP is installed")
        except ImportError:
            print("✗ CLIP is not installed")
            return 1
        
        info = CLIP_MODELS[args.model]
        weights_path = DEFAULT_CACHE_DIR / info["filename"]
        if file_looks_valid(weights_path, info["sha256"]):
            print(f"✓ {args.model} weights are ready")
            return 0
        else:
            print(f"✗ {args.model} weights are missing or invalid")
            return 1
    
    # Install CLIP
    if not install_clip():
        return 1
    
    # Skip weights download if --clip-only
    if args.clip_only:
        print("\n" + "=" * 70)
        print("Setup Complete: CLIP Installed")
        print("=" * 70)
        print("To download weights later, run:")
        print(f"  python clip_ssl.py --model {args.model}")
        return 0
    
    # Download weights
    if not download_clip_weights(args.model):
        return 1
    
    # Success!
    print("\n" + "=" * 70)
    print("✓ Setup Complete: CLIP Ready")
    print("=" * 70)
    print(f"CLIP is installed and weights are cached at:")
    print(f"  {DEFAULT_CACHE_DIR / CLIP_MODELS[args.model]['filename']}")
    print()
    print("You can now run the ablation study:")
    print("  bash run_ablation_study.sh")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
