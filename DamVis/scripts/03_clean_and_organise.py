"""
SCRIPT 3 — Image Cleaner and Organiser
=======================================
Run this AFTER collecting images from Google Earth and YouTube.
It automatically:
  - Removes blurry images (Laplacian variance < threshold)
  - Removes near-duplicate images (perceptual hash similarity)
  - Removes images that are mostly sky (>50% bright uniform pixels)
  - Removes images too small (<512×512)
  - Renames everything consistently: dam_XXXX.jpg
  - Generates metadata CSV

Usage:
    python 03_clean_and_organise.py --input /path/to/raw/images

Put all your raw collected images (any filenames, any subfolders)
in dataset/clean_raw/ and run this script.
Output goes to dataset/clean/ with clean filenames.
"""

import os
import cv2
import numpy as np
import shutil
import hashlib
import argparse
import csv
from pathlib import Path
from datetime import datetime

# ── Thresholds (tune if too aggressive) ──────────────────────────────────────
BLUR_THRESHOLD      = 80      # Laplacian variance — below this = too blurry
SKY_THRESHOLD       = 0.55    # >55% bright pixels = mostly sky, reject
MIN_SIZE            = 512     # Minimum width AND height in pixels
HASH_SIMILARITY     = 8       # Hamming distance — below this = near-duplicate
TARGET_SIZE         = (1024, 1024)  # Resize all kept images to this

RAW_DIR    = Path(__file__).parent.parent / "dataset" / "clean_raw"
CLEAN_DIR  = Path(__file__).parent.parent / "dataset" / "clean"
META_DIR   = Path(__file__).parent.parent / "metadata"

def compute_blur(img_gray):
    """Laplacian variance — low value = blurry."""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def compute_phash(img, hash_size=8):
    """Perceptual hash for near-duplicate detection."""
    small = cv2.resize(img, (hash_size + 1, hash_size))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else small
    diff  = gray[:, 1:] > gray[:, :-1]
    return diff.flatten()

def hamming_distance(h1, h2):
    return np.sum(h1 != h2)

def is_mostly_sky(img):
    """Reject images where >55% pixels are bright and uniform (sky-dominated)."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_mask = gray > 200
    std_map = cv2.blur(gray.astype(float), (15, 15))
    bright_frac = bright_mask.mean()
    return bright_frac > SKY_THRESHOLD

def process_image(path):
    """Load, validate, and return cleaned image or None if rejected."""
    img = cv2.imread(str(path))
    if img is None:
        return None, "unreadable"

    h, w = img.shape[:2]
    if w < MIN_SIZE or h < MIN_SIZE:
        return None, f"too_small ({w}x{h})"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = compute_blur(gray)
    if blur_score < BLUR_THRESHOLD:
        return None, f"too_blurry (score={blur_score:.1f})"

    if is_mostly_sky(img):
        return None, "mostly_sky"

    # Resize to standard size
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return img_resized, "ok"

def run(raw_dir=None):
    raw_dir   = Path(raw_dir) if raw_dir else RAW_DIR
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all image files recursively
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_files  = [p for p in raw_dir.rglob("*") if p.suffix in extensions]

    print(f"\n{'='*60}")
    print(f"  DAMVIS — Image Cleaner")
    print(f"{'='*60}")
    print(f"  Found {len(all_files)} raw images in {raw_dir}")
    print(f"  Processing...\n")

    kept        = []
    rejected    = []
    hashes_seen = []  # list of (hash_array, filename)
    counter     = 1

    stats = {"unreadable": 0, "too_small": 0, "too_blurry": 0,
             "mostly_sky": 0, "duplicate": 0, "kept": 0}

    for path in sorted(all_files):
        img, status = process_image(path)

        if status != "ok":
            key = status.split("(")[0].strip()
            stats[key] = stats.get(key, 0) + 1
            rejected.append({"original": str(path), "reason": status})
            continue

        # Duplicate check
        phash = compute_phash(img)
        is_dup = False
        for prev_hash, prev_name in hashes_seen:
            if hamming_distance(phash, prev_hash) < HASH_SIMILARITY:
                is_dup = True
                rejected.append({"original": str(path), "reason": f"duplicate_of_{prev_name}"})
                stats["duplicate"] += 1
                break

        if is_dup:
            continue

        # Save
        new_name = f"dam_{counter:04d}.jpg"
        out_path = CLEAN_DIR / new_name
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        hashes_seen.append((phash, new_name))

        # Detect source from filename
        fname_lower = path.name.lower()
        source = "google_earth" if any(x in fname_lower for x in ["ge_", "earth", "google"]) \
            else "youtube_cc"   if any(x in fname_lower for x in ["yt_", "youtube", "video"]) \
            else "public_dataset"

        kept.append({
            "filename": new_name,
            "original_path": str(path),
            "source": source,
            "width": TARGET_SIZE[0],
            "height": TARGET_SIZE[1],
            "date_added": datetime.today().strftime("%Y-%m-%d")
        })

        stats["kept"] += 1
        counter += 1

        if counter % 50 == 0:
            print(f"  Processed {counter-1} images kept so far...")

    # Write metadata CSV
    meta_path = META_DIR / "clean_images_metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename","original_path","source","width","height","date_added"])
        writer.writeheader()
        writer.writerows(kept)

    # Write rejected log
    reject_path = META_DIR / "rejected_images_log.csv"
    with open(reject_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original","reason"])
        writer.writeheader()
        writer.writerows(rejected)

    # Summary
    print(f"\n{'='*60}")
    print(f"  CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"  Input images    : {len(all_files)}")
    print(f"  ✅ Kept         : {stats['kept']}")
    print(f"  ❌ Rejected breakdown:")
    for k, v in stats.items():
        if k != "kept" and v > 0:
            print(f"       {k:<20}: {v}")
    print(f"\n  Clean images → {CLEAN_DIR}")
    print(f"  Metadata CSV → {meta_path}")
    print(f"  Reject log   → {reject_path}")
    print(f"\n  Next step: run 04_generate_degraded.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to raw image folder (default: dataset/clean_raw/)")
    args = parser.parse_args()
    run(args.input)
