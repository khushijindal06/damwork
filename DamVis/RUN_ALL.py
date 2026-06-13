"""
RUN_ALL.py — DamVis Dataset Master Script
==========================================
Run this for a complete overview of the pipeline and
to execute all steps in sequence.

Usage:
    python RUN_ALL.py --step all       # run everything (after images collected)
    python RUN_ALL.py --step clean     # just clean & organise images
    python RUN_ALL.py --step degrade   # just run degradation
    python RUN_ALL.py --step verify    # just verify quality
    python RUN_ALL.py --step readme    # just generate README
    python RUN_ALL.py --step guide     # print the full pipeline guide
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"

def run_script(name):
    path = SCRIPTS / name
    print(f"\n{'='*60}")
    print(f"  RUNNING: {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(path)])
    return result.returncode == 0

def print_full_guide():
    guide = """
╔══════════════════════════════════════════════════════════╗
║         DAMVIS DATASET — COMPLETE PIPELINE GUIDE         ║
╚══════════════════════════════════════════════════════════╝

TARGET: 2,000–5,000 clean images → 24,000–60,000 degraded pairs

STEP 0 — INSTALL DEPENDENCIES (once only)
─────────────────────────────────────────
  pip install opencv-python numpy tqdm noise yt-dlp Pillow
  sudo apt install ffmpeg     # Linux
  brew install ffmpeg          # Mac
  # Windows: download from ffmpeg.org

STEP 1 — GET YOUR CLEAN IMAGES (YOU DO THIS — takes 2 weeks)
──────────────────────────────────────────────────────────────
  Run: python scripts/01_google_earth_guide.py
  → It prints exact coordinates for 25 Indian dams
  → It creates: metadata/damvis_dams.kml

  Then:
  a) Open Google Earth Pro → File → Import → damvis_dams.kml
  b) Click each dam pin
  c) Set altitude to 100–200m
  d) Screenshot dam face, crest, slope, spillway (50–80 per dam)
  e) Save screenshots to:  dataset/clean_raw/

  Also:
  a) Go to YouTube → search dam drone footage → filter Creative Commons
  b) Copy CC video URLs into scripts/02_youtube_downloader.py
  c) Run: python scripts/02_youtube_downloader.py
  → Downloads videos + extracts 1 frame/second automatically

  TARGET: 2,000–4,000 raw images in dataset/clean_raw/

STEP 2 — CLEAN AND ORGANISE (automated)
────────────────────────────────────────
  Run: python scripts/03_clean_and_organise.py

  What it does automatically:
  ✓ Removes blurry images (Laplacian variance < 80)
  ✓ Removes near-duplicates (perceptual hash)
  ✓ Removes sky-dominated images (>55% bright pixels)
  ✓ Removes images smaller than 512×512
  ✓ Resizes everything to 1024×1024
  ✓ Renames to dam_0001.jpg, dam_0002.jpg ...
  ✓ Generates metadata CSV

  Expected: 60–75% of raw images pass → ~1,500–3,000 clean images

STEP 3 — GENERATE DEGRADED IMAGES (automated, runs overnight)
──────────────────────────────────────────────────────────────
  Run: python scripts/04_generate_degraded.py

  What it does:
  ✓ For each clean image, generates 12 degraded versions:
    - haze_light, haze_medium, haze_heavy
    - fog_light,  fog_medium,  fog_heavy
    - lowlight_dusk, lowlight_dawn, lowlight_night
    - mixed_light, mixed_medium, mixed_heavy

  Output count example:
    500 clean  →  6,000 degraded pairs
    1000 clean → 12,000 degraded pairs
    2000 clean → 24,000 degraded pairs
    3000 clean → 36,000 degraded pairs ← target for 5k dataset

  Estimated time: ~4–8 hours for 3,000 images on laptop CPU

STEP 4 — VERIFY QUALITY (automated)
─────────────────────────────────────
  Run: python scripts/05_verify_quality.py

  What it does:
  ✓ Computes PSNR and SSIM for every pair
  ✓ Flags anything outside expected ranges
  ✓ Saves 10 visual comparison grids to dataset/quality_check/
  ✓ Generates quality_report.csv

  Review the comparison grids to visually confirm degradations
  look realistic.

STEP 5 — ANNOTATE (you do this, takes 3–4 weeks)
─────────────────────────────────────────────────
  Tool: CVAT (free) at https://cvat.ai

  a) Create account on cvat.ai
  b) Create project with classes:
     Segmentation: dam_body, embankment_slope, water_surface,
                   spillway, vegetation, bare_soil, access_road
     Detection:    crack_defect, seepage_stain
  c) Upload 1,000 clean images from dataset/clean/
  d) Use SAM (Segment Anything) auto-annotation → manual correction
  e) Export as:
     Segmentation → PNG masks → save to dataset/annotations/masks/
     Detection    → YOLO format → save to dataset/annotations/boxes/
  f) Annotate 200–300 images per week

  Target: 1,000–1,500 annotated images (clean images only —
          annotations auto-apply to degraded counterparts)

STEP 6 — FINALISE (automated)
──────────────────────────────
  Run: python scripts/06_generate_readme.py

  → Computes final statistics
  → Generates complete README.md for GitHub
  → Dataset is ready to upload

FINAL DATASET SIZE TARGETS:
────────────────────────────
  Minimum (small pilot):  300 clean  →  3,600 pairs + 300 annotated
  Good (publishable):    1500 clean  → 18,000 pairs + 1000 annotated
  Strong (PhD):          3000 clean  → 36,000 pairs + 1500 annotated

WHERE TO UPLOAD:
────────────────
  GitHub      : code + small samples + README
  Zenodo      : full dataset (free, gives DOI for citation)
  HuggingFace : alternative, great for ML community visibility
"""
    print(guide)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="guide",
                        choices=["guide","all","clean","degrade","verify","readme"])
    args = parser.parse_args()

    if args.step == "guide":
        print_full_guide()
    elif args.step == "clean":
        run_script("03_clean_and_organise.py")
    elif args.step == "degrade":
        run_script("04_generate_degraded.py")
    elif args.step == "verify":
        run_script("05_verify_quality.py")
    elif args.step == "readme":
        run_script("06_generate_readme.py")
    elif args.step == "all":
        steps = [
            ("03_clean_and_organise.py", "Clean & Organise"),
            ("04_generate_degraded.py",  "Generate Degraded"),
            ("05_verify_quality.py",     "Verify Quality"),
            ("06_generate_readme.py",    "Generate README"),
        ]
        for script, label in steps:
            ok = run_script(script)
            if not ok:
                print(f"\n❌ Failed at: {label}. Fix errors before continuing.")
                break
        else:
            print("\n✅ All steps completed successfully!")

if __name__ == "__main__":
    main()
