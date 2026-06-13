"""
SCRIPT 4 — Degradation Pipeline (Core Script)
===============================================
Takes every clean image in dataset/clean/ and generates
12 degraded versions (4 types × 3 severities each).

For 300 clean images  → 3,600  degraded pairs
For 500 clean images  → 6,000  degraded pairs
For 1000 clean images → 12,000 degraded pairs
For 2000 clean images → 24,000 degraded pairs

Usage:
    python 04_generate_degraded.py

All outputs are saved as paired folders:
    dataset/clean/dam_0001.jpg  ←→  dataset/degraded/haze_light/dam_0001.jpg
    same filename in both = they are paired automatically
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Try to import noise for Perlin fog; fallback to Gaussian if not available
try:
    from noise import pnoise2
    HAS_PERLIN = True
except ImportError:
    HAS_PERLIN = False
    print("⚠️  'noise' package not found. Using Gaussian noise for fog.")
    print("   Install with: pip install noise")

CLEAN_DIR   = Path(__file__).parent.parent / "dataset" / "clean"
OUT_BASE    = Path(__file__).parent.parent / "dataset" / "degraded"

# ── Degradation configuration ─────────────────────────────────────────────────
HAZE_CONFIGS = {
    "haze_light":  {"beta": 0.6,  "A": 0.85},
    "haze_medium": {"beta": 1.2,  "A": 0.90},
    "haze_heavy":  {"beta": 2.0,  "A": 0.95},
}
FOG_CONFIGS = {
    "fog_light":   {"density": 0.35, "turbulence": 0.3},
    "fog_medium":  {"density": 0.60, "turbulence": 0.5},
    "fog_heavy":   {"density": 0.85, "turbulence": 0.6},
}
LOWLIGHT_CONFIGS = {
    "lowlight_dusk":  {"gamma": 2.0, "noise_sigma": 0.02},
    "lowlight_dawn":  {"gamma": 3.0, "noise_sigma": 0.04},
    "lowlight_night": {"gamma": 4.5, "noise_sigma": 0.07},
}
MIXED_CONFIGS = {
    "mixed_light":  {"beta": 0.6,  "A": 0.85, "gamma": 2.0, "noise_sigma": 0.02},
    "mixed_medium": {"beta": 1.2,  "A": 0.90, "gamma": 3.0, "noise_sigma": 0.04},
    "mixed_heavy":  {"beta": 2.0,  "A": 0.92, "gamma": 3.5, "noise_sigma": 0.05},
}

# ── Depth estimation (lightweight — no deep learning required) ─────────────────
def estimate_depth_simple(img):
    """
    Lightweight depth proxy without neural network.
    For aerial dam images: bottom of image (water surface) = near
    Top of image (distant terrain/sky) = far.
    This simulates the altitude-dependent haze in valley settings.
    """
    h, w = img.shape[:2]
    # Vertical gradient: top=far(1.0), bottom=near(0.2)
    depth = np.linspace(1.0, 0.2, h)[:, np.newaxis] * np.ones((1, w))

    # Refine with image brightness (darker areas often further)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    dark_prior = 1.0 - cv2.GaussianBlur(gray, (15, 15), 0)
    depth = 0.6 * depth + 0.4 * dark_prior

    # Normalise to [0.1, 1.0]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = 0.1 + 0.9 * depth
    return depth

# ── Haze generation ───────────────────────────────────────────────────────────
def add_haze(img_float, beta=1.2, A=0.90):
    """
    Atmospheric scattering model: I = J*t + A*(1-t)
    t(x) = exp(-beta * d(x))
    Where d(x) is the depth map.
    Dam-specific: stronger haze near water surface (bottom of image).
    """
    depth = estimate_depth_simple((img_float * 255).astype(np.uint8))

    # Valley fog effect: denser at water level
    h = img_float.shape[0]
    valley_weight = np.linspace(0.3, 1.0, h)[:, np.newaxis, np.newaxis]

    transmission = np.exp(-beta * depth)[:, :, np.newaxis]
    transmission = np.clip(transmission, 0.05, 1.0)

    A_map = A * np.ones_like(img_float)
    hazy = img_float * transmission + A_map * (1 - transmission)
    return np.clip(hazy, 0, 1)

# ── Fog generation ────────────────────────────────────────────────────────────
def generate_fog_mask(shape, density=0.6, turbulence=0.5, scale=200.0):
    """Generate spatially varying fog using Perlin or Gaussian noise."""
    h, w = shape[:2]
    if HAS_PERLIN:
        mask = np.zeros((h, w), dtype=float)
        offset_x = np.random.randint(0, 1000)
        offset_y = np.random.randint(0, 1000)
        for y in range(h):
            for x in range(0, w, 4):  # stride=4 for speed, interpolate
                val = pnoise2((x + offset_x) / scale,
                              (y + offset_y) / scale,
                              octaves=4, persistence=turbulence)
                mask[y, x:x+4] = val
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    else:
        # Gaussian blur noise as fallback
        noise = np.random.randn(h // 8, w // 8)
        mask = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # Stronger fog at bottom (valley/water surface)
    gradient = np.linspace(0.5, 1.2, h)[:, np.newaxis]
    mask = mask * gradient
    mask = np.clip(mask * density, 0, 1)
    return mask

def add_fog(img_float, density=0.6, turbulence=0.5):
    h, w = img_float.shape[:2]
    fog_mask = generate_fog_mask((h, w), density, turbulence)
    fog_mask_3ch = fog_mask[:, :, np.newaxis]

    # Fog colour: slightly warm white
    fog_colour = np.array([0.98, 0.97, 0.95])
    foggy = img_float * (1 - fog_mask_3ch) + fog_colour * fog_mask_3ch
    return np.clip(foggy, 0, 1)

# ── Low-light generation ──────────────────────────────────────────────────────
def add_low_light(img_float, gamma=2.5, noise_sigma=0.03):
    """
    Gamma darkening + Poisson-Gaussian noise.
    Simulates sensor noise in low-light UAV cameras.
    """
    dark = np.power(img_float, gamma)

    # Poisson noise (signal-dependent)
    scale = 255.0
    poisson_noise = np.random.poisson(dark * scale) / scale - dark
    poisson_noise = np.clip(poisson_noise, -0.3, 0.3)

    # Gaussian read noise
    gaussian_noise = np.random.normal(0, noise_sigma, dark.shape)

    dark_noisy = dark + 0.3 * poisson_noise + 0.7 * gaussian_noise

    # Slight colour temperature shift (orange cast for dusk/dawn)
    if gamma <= 2.5:
        dark_noisy[:, :, 2] *= 1.05  # slightly boost red channel (BGR)
        dark_noisy[:, :, 0] *= 0.97  # slightly reduce blue

    return np.clip(dark_noisy, 0, 1)

# ── Mixed (haze + low-light) ──────────────────────────────────────────────────
def add_mixed(img_float, beta=1.0, A=0.88, gamma=2.5, noise_sigma=0.03):
    hazy  = add_haze(img_float, beta, A)
    mixed = add_low_light(hazy, gamma, noise_sigma)
    return mixed

# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_all():
    clean_files = sorted(CLEAN_DIR.glob("*.jpg")) + sorted(CLEAN_DIR.glob("*.png"))

    if not clean_files:
        print(f"❌ No images found in {CLEAN_DIR}")
        print("   Run 01 and 02 scripts first to collect clean images")
        return

    print(f"\n{'='*60}")
    print(f"  DAMVIS — Degradation Pipeline")
    print(f"{'='*60}")
    print(f"  Clean images found : {len(clean_files)}")
    print(f"  Variants per image : 12")
    print(f"  Total output       : {len(clean_files) * 12:,} degraded images")
    print(f"  Estimated time     : {len(clean_files) * 12 // 3600 + 1} hours\n")

    # Create output directories
    all_configs = {**HAZE_CONFIGS, **FOG_CONFIGS, **LOWLIGHT_CONFIGS, **MIXED_CONFIGS}
    for folder in all_configs:
        (OUT_BASE / folder).mkdir(parents=True, exist_ok=True)

    counts = {k: 0 for k in all_configs}

    for img_path in tqdm(clean_files, desc="Processing images"):
        img_bgr   = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_float = img_bgr.astype(float) / 255.0
        stem      = img_path.stem

        # ── Haze ──
        for name, cfg in HAZE_CONFIGS.items():
            out = add_haze(img_float, cfg["beta"], cfg["A"])
            cv2.imwrite(str(OUT_BASE / name / f"{stem}.jpg"),
                        (out * 255).astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[name] += 1

        # ── Fog ──
        for name, cfg in FOG_CONFIGS.items():
            out = add_fog(img_float, cfg["density"], cfg["turbulence"])
            cv2.imwrite(str(OUT_BASE / name / f"{stem}.jpg"),
                        (out * 255).astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[name] += 1

        # ── Low-light ──
        for name, cfg in LOWLIGHT_CONFIGS.items():
            out = add_low_light(img_float, cfg["gamma"], cfg["noise_sigma"])
            cv2.imwrite(str(OUT_BASE / name / f"{stem}.jpg"),
                        (out * 255).astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[name] += 1

        # ── Mixed ──
        for name, cfg in MIXED_CONFIGS.items():
            out = add_mixed(img_float, cfg["beta"], cfg["A"],
                            cfg["gamma"], cfg["noise_sigma"])
            cv2.imwrite(str(OUT_BASE / name / f"{stem}.jpg"),
                        (out * 255).astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[name] += 1

    print(f"\n{'='*60}")
    print(f"  DEGRADATION COMPLETE")
    print(f"{'='*60}")
    total = sum(counts.values())
    for name, count in counts.items():
        print(f"  {name:<22}: {count} images")
    print(f"  {'TOTAL':<22}: {total:,} degraded images")
    print(f"\n  Output: {OUT_BASE}")
    print(f"\n  Next step: run 05_verify_quality.py")

if __name__ == "__main__":
    process_all()
