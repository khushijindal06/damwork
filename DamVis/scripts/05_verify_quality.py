"""
SCRIPT 5 — Quality Verification
=================================
Verifies the degradation pipeline worked correctly by:
  1. Computing PSNR/SSIM for every degraded pair
  2. Flagging anything with unrealistic PSNR values
  3. Saving visual comparison grids (clean vs all 12 variants)

Usage:
    python 05_verify_quality.py

Expected PSNR ranges (flags anything outside):
    Haze light   : 22–30 dB
    Haze heavy   : 13–20 dB
    Fog light    : 20–28 dB
    Fog heavy    : 12–18 dB
    Low-light    : 15–25 dB
    Mixed        : 12–20 dB
"""

import cv2
import numpy as np
import csv
import random
from pathlib import Path
from tqdm import tqdm

CLEAN_DIR  = Path(__file__).parent.parent / "dataset" / "clean"
DEG_DIR    = Path(__file__).parent.parent / "dataset" / "degraded"
META_DIR   = Path(__file__).parent.parent / "metadata"
QC_DIR     = Path(__file__).parent.parent / "dataset" / "quality_check"

PSNR_RANGES = {
    "haze_light":    (20, 32),
    "haze_medium":   (15, 25),
    "haze_heavy":    (11, 20),
    "fog_light":     (18, 30),
    "fog_medium":    (13, 23),
    "fog_heavy":     (10, 18),
    "lowlight_dusk": (16, 26),
    "lowlight_dawn": (12, 22),
    "lowlight_night":(8,  18),
    "mixed_light":   (14, 24),
    "mixed_medium":  (10, 20),
    "mixed_heavy":   (8,  17),
}

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(255.0 ** 2 / mse)

def compute_ssim_simple(img1, img2):
    """Simplified SSIM computation."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    i1 = img1.astype(float)
    i2 = img2.astype(float)
    mu1, mu2 = i1.mean(), i2.mean()
    sig1 = i1.std()
    sig2 = i2.std()
    sig12 = np.mean((i1 - mu1) * (i2 - mu2))
    ssim = ((2*mu1*mu2 + C1) * (2*sig12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sig1**2 + sig2**2 + C2))
    return ssim

def make_comparison_grid(clean_path, n_samples=5):
    """Create a visual grid showing clean vs all 12 degraded versions."""
    QC_DIR.mkdir(parents=True, exist_ok=True)

    clean = cv2.imread(str(clean_path))
    if clean is None:
        return
    clean_small = cv2.resize(clean, (256, 256))

    variant_dirs = sorted(DEG_DIR.iterdir())
    row_imgs = [clean_small]
    labels   = ["CLEAN"]

    for vdir in variant_dirs:
        deg_path = vdir / clean_path.name
        if deg_path.exists():
            deg = cv2.imread(str(deg_path))
            if deg is not None:
                row_imgs.append(cv2.resize(deg, (256, 256)))
                labels.append(vdir.name.replace("_", "\n"))

    if len(row_imgs) < 2:
        return

    # Arrange in 3 rows of ~5
    cols = 5
    rows_of_imgs = []
    for i in range(0, len(row_imgs), cols):
        chunk = row_imgs[i:i+cols]
        while len(chunk) < cols:
            chunk.append(np.zeros((256, 256, 3), dtype=np.uint8))
        row = np.hstack(chunk)
        rows_of_imgs.append(row)

    grid = np.vstack(rows_of_imgs)

    # Add title
    title_bar = np.ones((40, grid.shape[1], 3), dtype=np.uint8) * 30
    cv2.putText(title_bar, f"DamVis Quality Check: {clean_path.name}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    grid = np.vstack([title_bar, grid])

    out_path = QC_DIR / f"qc_{clean_path.stem}.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path

def run():
    clean_files = sorted(CLEAN_DIR.glob("*.jpg")) + sorted(CLEAN_DIR.glob("*.png"))
    if not clean_files:
        print(f"❌ No clean images found in {CLEAN_DIR}")
        return

    print(f"\n{'='*60}")
    print(f"  DAMVIS — Quality Verification")
    print(f"{'='*60}")
    print(f"  Checking {len(clean_files)} clean images × 12 variants\n")

    results    = []
    flagged    = []
    total_ok   = 0
    total_flag = 0

    for clean_path in tqdm(clean_files, desc="Verifying pairs"):
        clean_img = cv2.imread(str(clean_path))
        if clean_img is None:
            continue

        for variant_name, (lo, hi) in PSNR_RANGES.items():
            deg_path = DEG_DIR / variant_name / clean_path.name
            if not deg_path.exists():
                flagged.append({"file": clean_path.name, "variant": variant_name,
                                "issue": "file_missing", "psnr": "N/A"})
                continue

            deg_img = cv2.imread(str(deg_path))
            if deg_img is None:
                flagged.append({"file": clean_path.name, "variant": variant_name,
                                "issue": "unreadable", "psnr": "N/A"})
                continue

            psnr = compute_psnr(clean_img, deg_img)
            ssim = compute_ssim_simple(clean_img, deg_img)

            status = "ok"
            if psnr < lo or psnr > hi:
                status = f"psnr_out_of_range ({psnr:.1f} dB, expected {lo}–{hi})"
                flagged.append({"file": clean_path.name, "variant": variant_name,
                                "issue": status, "psnr": f"{psnr:.2f}"})
                total_flag += 1
            else:
                total_ok += 1

            results.append({
                "clean_file": clean_path.name,
                "variant": variant_name,
                "psnr_db": round(psnr, 2),
                "ssim": round(ssim, 4),
                "status": status
            })

    # Save full PSNR report
    META_DIR.mkdir(parents=True, exist_ok=True)
    psnr_path = META_DIR / "quality_report.csv"
    with open(psnr_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clean_file","variant","psnr_db","ssim","status"])
        writer.writeheader()
        writer.writerows(results)

    # Save flagged report
    flag_path = META_DIR / "flagged_images.csv"
    with open(flag_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file","variant","issue","psnr"])
        writer.writeheader()
        writer.writerows(flagged)

    # Make visual grids for 10 random clean images
    print("\n  Generating visual comparison grids (10 samples)...")
    sample_files = random.sample(clean_files, min(10, len(clean_files)))
    grid_paths = []
    for f in sample_files:
        gp = make_comparison_grid(f)
        if gp:
            grid_paths.append(gp)

    # Print per-variant average PSNR
    print(f"\n{'─'*50}")
    print(f"  {'Variant':<22} {'Avg PSNR':>10} {'Avg SSIM':>10}")
    print(f"{'─'*50}")
    for variant in PSNR_RANGES:
        variant_results = [r for r in results if r["variant"] == variant]
        if variant_results:
            avg_psnr = np.mean([r["psnr_db"] for r in variant_results])
            avg_ssim = np.mean([r["ssim"] for r in variant_results])
            lo, hi = PSNR_RANGES[variant]
            flag = "✅" if lo <= avg_psnr <= hi else "⚠️"
            print(f"  {flag} {variant:<20} {avg_psnr:>9.2f} {avg_ssim:>10.4f}")

    print(f"\n{'='*60}")
    print(f"  VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  ✅ Pairs passed : {total_ok:,}")
    print(f"  ⚠️  Pairs flagged: {total_flag:,}")
    if total_flag > 0:
        print(f"  → Review: {flag_path}")
    print(f"  PSNR report  : {psnr_path}")
    print(f"  Visual grids : {QC_DIR} ({len(grid_paths)} saved)")

if __name__ == "__main__":
    run()
