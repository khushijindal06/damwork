"""
SCRIPT 6 — Dataset Statistics + README Generator
==================================================
Run this LAST. Computes final dataset statistics and generates
a complete README.md ready for GitHub upload.

Usage:
    python 06_generate_readme.py
"""

import os
import csv
import json
from pathlib import Path
from datetime import date

BASE    = Path(__file__).parent.parent
META    = BASE / "metadata"
CLEAN   = BASE / "dataset" / "clean"
DEG     = BASE / "dataset" / "degraded"
ANNOT   = BASE / "dataset" / "annotations"

def count_images(folder):
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))

def get_dataset_stats():
    stats = {}
    stats["clean_count"] = count_images(CLEAN)

    stats["degraded"] = {}
    total_degraded = 0
    if DEG.exists():
        for variant_dir in sorted(DEG.iterdir()):
            if variant_dir.is_dir():
                c = count_images(variant_dir)
                stats["degraded"][variant_dir.name] = c
                total_degraded += c
    stats["total_degraded"] = total_degraded
    stats["total_pairs"]    = total_degraded  # 1 clean = 1 pair per variant
    stats["total_images"]   = stats["clean_count"] + total_degraded

    # Annotation stats
    masks_dir = ANNOT / "masks"
    boxes_dir = ANNOT / "boxes"
    stats["annotated_seg"]   = count_images(masks_dir) if masks_dir.exists() else 0
    stats["annotated_boxes"] = len(list(boxes_dir.glob("*.txt"))) if boxes_dir.exists() else 0

    return stats

def generate_readme(stats):
    today = str(date.today())
    clean  = stats["clean_count"]
    total  = stats["total_images"]
    pairs  = stats["total_pairs"]

    readme = f"""# DamVis — UAV Dam & Embankment Visibility Enhancement Dataset

**First large-scale paired multi-degradation UAV aerial image dataset  
for dam and embankment inspection under adverse visibility conditions.**

[![Dataset](https://img.shields.io/badge/dataset-DamVis-blue)]()
[![Licence](https://img.shields.io/badge/licence-CC_BY_4.0-green)]()
[![Images](https://img.shields.io/badge/images-{total:,}-orange)]()

---

## Overview

| Property | Value |
|---|---|
| **Total images** | {total:,} |
| **Clean source images** | {clean:,} |
| **Degraded training pairs** | {pairs:,} |
| **Degradation types** | 4 (Haze, Fog, Low-light, Mixed) |
| **Severity levels each** | 3 (Light, Medium, Heavy) |
| **Annotated images (segmentation)** | {stats["annotated_seg"]:,} |
| **Annotated images (detection)** | {stats["annotated_boxes"]:,} |
| **Image resolution** | 1024 × 1024 px |
| **Format** | JPEG (degraded) / PNG (masks) |
| **Licence** | CC BY 4.0 |

---

## Motivation

UAV-based dam and embankment inspection is a growing field, but all existing
methods assume clear-weather imagery. In practice, dams sit in river valleys
and mountain gorges — environments that are naturally prone to:

- **Valley fog** (winter mornings, especially in Himalayan/Deccan plateau dams)
- **Monsoon haze** (flood-season inspections under overcast/hazy conditions)
- **Pre-dawn low light** (emergency inspections after earthquakes or heavy rain)
- **Mixed conditions** (hazy dusk, foggy dawn)

No dataset or model currently addresses visibility enhancement for this domain.
DamVis fills that gap.

---

## Dataset Structure

```
DamVis/
│
├── dataset/
│   ├── clean/                     # {clean:,} clean source images
│   │   ├── dam_0001.jpg
│   │   └── ...
│   │
│   ├── degraded/
│   │   ├── haze_light/            # β=0.6
│   │   ├── haze_medium/           # β=1.2
│   │   ├── haze_heavy/            # β=2.0
│   │   ├── fog_light/             # density=0.35
│   │   ├── fog_medium/            # density=0.60
│   │   ├── fog_heavy/             # density=0.85
│   │   ├── lowlight_dusk/         # γ=2.0
│   │   ├── lowlight_dawn/         # γ=3.0
│   │   ├── lowlight_night/        # γ=4.5
│   │   ├── mixed_light/           # haze+lowlight, light
│   │   ├── mixed_medium/          # haze+lowlight, medium
│   │   └── mixed_heavy/           # haze+lowlight, heavy
│   │
│   └── annotations/
│       ├── masks/                 # PNG segmentation masks
│       └── boxes/                 # YOLO-format .txt detection labels
│
└── metadata/
    ├── clean_images_metadata.csv  # source, dam name, coordinates
    ├── quality_report.csv         # PSNR/SSIM for all pairs
    ├── dams.json                  # 25 dam coordinates used
    └── youtube_sources.csv        # CC video attribution log
```

---

## Degradation Image Counts

| Variant | Images | Beta / Gamma / Density |
|---|---|---|
"""

    for name, count in stats["degraded"].items():
        readme += f"| `{name}` | {count:,} | — |\n"

    readme += f"""
| **Total degraded** | **{stats['total_degraded']:,}** | |
| **Total (all)** | **{total:,}** | |

---

## Pairing Convention

Each degraded image has the **same filename** as its clean counterpart.

```python
clean   = "dataset/clean/dam_0042.jpg"
hazy    = "dataset/degraded/haze_medium/dam_0042.jpg"  # same stem
foggy   = "dataset/degraded/fog_heavy/dam_0042.jpg"
dark    = "dataset/degraded/lowlight_night/dam_0042.jpg"
```

---

## Annotation Classes

| Class | Segmentation | Detection |
|---|---|---|
| `dam_body` | ✅ | — |
| `embankment_slope` | ✅ | — |
| `water_surface` | ✅ | — |
| `spillway` | ✅ | ✅ |
| `vegetation` | ✅ | — |
| `bare_soil` | ✅ | — |
| `crack_defect` | — | ✅ |
| `seepage_stain` | — | ✅ |
| `access_road` | ✅ | — |

---

## Degradation Physics

### Haze Model
```
I(x) = J(x) · t(x) + A · (1 − t(x))
t(x) = exp(−β · d(x))
```
Where `d(x)` is the estimated depth, `A` is atmospheric light, `β` controls severity.
**Dam-specific:** haze density increases toward water surface (valley fog effect).

### Fog Model
Spatially varying fog generated with Perlin noise modulated by altitude gradient,
producing realistic non-uniform valley fog accumulation patterns.

### Low-Light Model
```
I_dark(x) = I_clean(x)^γ  +  Poisson noise  +  Gaussian read noise
```
Colour temperature shift applied for dusk/dawn variants.

---

## Benchmark Baselines

| Model | Haze PSNR | Fog PSNR | LL PSNR | Mixed PSNR |
|---|---|---|---|---|
| No enhancement (input) | — | — | — | — |
| DCP (He et al., 2009) | TBD | TBD | N/A | TBD |
| DehazeFormer | TBD | TBD | N/A | TBD |
| RetinexFormer | N/A | N/A | TBD | TBD |
| Zero-DCE | N/A | N/A | TBD | TBD |
| **Proposed (AeroVis-Net)** | **TBD** | **TBD** | **TBD** | **TBD** |

---

## Data Sources

- **Google Earth Pro** — high-resolution aerial screenshots of 25 Indian dams (academic use)
- **YouTube Creative Commons** — CC-BY licensed drone footage (see `metadata/youtube_sources.csv`)
- **Public datasets** — FloodNet, OpenAerialMap (subset filtered for dam/embankment scenes)

---

## Licence

The DamVis dataset is released under the  
**Creative Commons Attribution 4.0 International (CC BY 4.0)** licence.

You are free to use, share, and adapt this dataset for any purpose,  
provided you give appropriate credit.

---

## Citation

If you use DamVis in your research, please cite:

```bibtex
@dataset{{damvis2026,
  title   = {{DamVis: A Multi-Degradation UAV Aerial Image Dataset for Dam and Embankment Inspection}},
  author  = {{[Author Names]}},
  year    = {{2026}},
  note    = {{Dataset available at: [URL]}},
  licence = {{CC BY 4.0}}
}}
```

---

*Generated: {today}*
"""
    return readme

def run():
    print(f"\n{'='*60}")
    print(f"  DAMVIS — Final Statistics + README")
    print(f"{'='*60}")

    stats = get_dataset_stats()

    print(f"\n  Dataset Statistics:")
    print(f"  {'─'*40}")
    print(f"  Clean images       : {stats['clean_count']:,}")
    print(f"  Degraded pairs     : {stats['total_degraded']:,}")
    print(f"  Total images       : {stats['total_images']:,}")
    print(f"  Seg. annotations   : {stats['annotated_seg']:,}")
    print(f"  Det. annotations   : {stats['annotated_boxes']:,}")

    # Save stats JSON
    stats_path = META / "dataset_statistics.json"
    META.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    # Generate README
    readme_content = generate_readme(stats)
    readme_path = BASE / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"  README saved: {readme_path}")

    print(f"\n{'='*60}")
    print(f"  DATASET COMPLETE — ready for GitHub upload")
    print(f"{'='*60}")
    print(f"\n  Upload instructions:")
    print(f"  1. Create a GitHub repo named 'DamVis-Dataset'")
    print(f"  2. git init && git add . && git commit -m 'Initial dataset release'")
    print(f"  3. git remote add origin https://github.com/[you]/DamVis-Dataset")
    print(f"  4. git push origin main")
    print(f"\n  For large files (>100MB): use Git LFS or host on Zenodo/HuggingFace")
    print(f"  Zenodo (free, gives DOI): https://zenodo.org")
    print(f"  HuggingFace Datasets    : https://huggingface.co/datasets")

if __name__ == "__main__":
    run()
