#!/usr/bin/env python3
"""
fix_clip_ssl.py
===============
Run this ONCE on your GPU machine to download the CLIP ViT-B/32 weights
without going through CLIP's own SSL-blocked downloader.

After this script completes, exp1_score.py will find the weights at
~/.cache/clip/ViT-B-32.pt and load them without any network access.

Usage
-----
    python fix_clip_ssl.py

    # Or if you want the file somewhere specific:
    python fix_clip_ssl.py --out /path/to/ViT-B-32.pt
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

# CLIP's default cache directory — must match what clip.load() looks in
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "clip"


# ── Helpers ─────────────────────────────────────────────────────────────────

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
    """Return True if file exists and its SHA-256 matches."""
    if not path.exists():
        return False
    if path.stat().st_size < 1_000_000:          # < 1 MB → truncated download
        return False
    print(f"  Verifying SHA-256 of {path.name} ...", flush=True)
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        print(f"  [WARN] SHA-256 mismatch: expected {expected_sha256[:16]}… "
              f"got {actual[:16]}…")
        return False
    return True


def download_with_wget(url: str, dest: Path) -> bool:
    """
    Download url → dest using wget with SSL verification disabled.
    Returns True on success.

    Why wget --no-check-certificate rather than Python urllib?
    Because Python's urllib goes through the same OS SSL stack that is
    rejecting the self-signed cert. wget uses its own SSL layer and
    --no-check-certificate bypasses it entirely.
    """
    cmd = [
        "wget",
        "--no-check-certificate",   # bypass the self-signed cert
        "--quiet",                   # suppress progress noise
        "--show-progress",           # but keep the progress bar
        "--tries=3",                 # retry on transient failures
        "--timeout=120",             # 120s connect/read timeout
        "-O", str(dest),             # output file
        url,
    ]
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode == 0


def download_with_curl(url: str, dest: Path) -> bool:
    """Fallback to curl if wget fails."""
    cmd = [
        "curl",
        "-k",                        # --insecure: skip SSL verification
        "--retry", "3",
        "--connect-timeout", "120",
        "-L",                        # follow redirects
        "--progress-bar",
        "-o", str(dest),
        url,
    ]
    print(f"  Falling back to curl ...", flush=True)
    result = subprocess.run(cmd)
    return result.returncode == 0


def ensure_clip_model(model_name: str = "ViT-B/32",
                      out_path: Path | None = None) -> Path:
    """
    Ensure the CLIP model weights are present at out_path (or the default
    cache location). Downloads them if not present or if the file is corrupt.

    Returns the path to the validated weights file.
    """
    info = CLIP_MODELS[model_name]
    dest = out_path or (DEFAULT_CACHE_DIR / info["filename"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    if file_looks_valid(dest, info["sha256"]):
        print(f"[OK] {dest} already exists and SHA-256 is valid — nothing to do.")
        return dest

    if dest.exists():
        print(f"[INFO] {dest} exists but failed validation — re-downloading.")
        dest.unlink()

    print(f"[INFO] Downloading {model_name} ({info['size_mb']} MB) ...")
    print(f"  → {dest}")

    ok = download_with_wget(info["url"], dest)
    if not ok:
        ok = download_with_curl(info["url"], dest)

    if not ok:
        print("[ERROR] Both wget and curl failed. See troubleshooting below.")
        _print_manual_instructions(info, dest)
        sys.exit(1)

    # Validate after download
    if not file_looks_valid(dest, info["sha256"]):
        print("[ERROR] Download completed but SHA-256 verification failed.")
        print("        The file may be corrupt or the URL served wrong content.")
        dest.unlink(missing_ok=True)
        _print_manual_instructions(info, dest)
        sys.exit(1)

    print(f"[OK] Downloaded and verified: {dest}")
    return dest


def _print_manual_instructions(info: dict, dest: Path):
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("If automated download fails, copy this URL and download")
    print("the file manually on a machine with unrestricted internet,")
    print("then scp/rsync it to your GPU machine at the path below.\n")
    print(f"  URL:  {info['url']}")
    print(f"  Save to: {dest}")
    print(f"  SHA-256: {info['sha256']}")
    print("\nAfter placing the file, re-run this script to verify it.")
    print("=" * 60)


# ── Patch function for exp1_score.py ─────────────────────────────────────────

def get_local_clip_path(model_name: str = "ViT-B/32") -> str:
    """
    Call this from exp1_score.py instead of using clip.load(model_name).
    Returns the local file path as a string.

    Example usage in exp1_score.py:
        from fix_clip_ssl import get_local_clip_path
        model, preprocess = clip.load(get_local_clip_path())
    """
    info  = CLIP_MODELS[model_name]
    dest  = DEFAULT_CACHE_DIR / info["filename"]
    if not dest.exists():
        raise FileNotFoundError(
            f"CLIP model not found at {dest}. "
            f"Run:  python fix_clip_ssl.py  to download it first."
        )
    return str(dest)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Download CLIP weights bypassing SSL certificate errors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", default="ViT-B/32",
        choices=list(CLIP_MODELS.keys()),
        help="Which CLIP model to download",
    )
    p.add_argument(
        "--out", default=None,
        help="Custom output path. Defaults to ~/.cache/clip/ViT-B-32.pt",
    )
    p.add_argument(
        "--verify-only", action="store_true",
        help="Only check if the file exists and is valid — do not download",
    )
    args = p.parse_args()

    out_path = Path(args.out) if args.out else None
    info = CLIP_MODELS[args.model]

    if args.verify_only:
        dest = out_path or (DEFAULT_CACHE_DIR / info["filename"])
        valid = file_looks_valid(dest, info["sha256"])
        if valid:
            print(f"[OK] {dest} is present and valid.")
        else:
            print(f"[FAIL] {dest} is missing or invalid.")
            sys.exit(1)
        return

    dest = ensure_clip_model(args.model, out_path)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"Model saved to: {dest}")
    print()
    print("In helpers.py, your clip_score_metric should now use:")
    print()
    print("    import clip")
    print("    from clip_scorer import CLIPScorer")
    print()
    print("    scorer = CLIPScorer(device='cuda:0')")
    print("    score = scorer.score(image, prompt)")
    print()
    print("This loads from the local file — no network access at all.")
    print("=" * 60)


if __name__ == "__main__":
    main()
