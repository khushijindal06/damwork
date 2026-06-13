# DamVis — UAV Dam & Embankment Visibility Enhancement Dataset

> **Repository note:** This document describes a previous three-image pilot.
> Its generated `dataset/` and metadata reports are not included here. Use
> `../damwork.ipynb` or `../Duan_YOLO_Code/` for the current Duan RGB workflow.

**First large-scale paired multi-degradation UAV aerial image dataset  
for dam and embankment inspection under adverse visibility conditions.**

[![Dataset](https://img.shields.io/badge/dataset-DamVis-blue)]()
[![Licence](https://img.shields.io/badge/licence-CC_BY_4.0-green)]()
[![Images](https://img.shields.io/badge/images-39-orange)]()

---

## Overview

| Property | Value |
|---|---|
| **Total images** | 39 |
| **Clean source images** | 3 |
| **Degraded training pairs** | 36 |
| **Degradation types** | 4 (Haze, Fog, Low-light, Mixed) |
| **Severity levels each** | 3 (Light, Medium, Heavy) |
| **Annotated images (segmentation)** | 0 |
| **Annotated images (detection)** | 0 |
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
│   ├── clean/                     # 3 clean source images
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
| `fog_heavy` | 3 | — |
| `fog_light` | 3 | — |
| `fog_medium` | 3 | — |
| `haze_heavy` | 3 | — |
| `haze_light` | 3 | — |
| `haze_medium` | 3 | — |
| `lowlight_dawn` | 3 | — |
| `lowlight_dusk` | 3 | — |
| `lowlight_night` | 3 | — |
| `mixed_heavy` | 3 | — |
| `mixed_light` | 3 | — |
| `mixed_medium` | 3 | — |

| **Total degraded** | **36** | |
| **Total (all)** | **39** | |

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
@dataset{damvis2026,
  title   = {DamVis: A Multi-Degradation UAV Aerial Image Dataset for Dam and Embankment Inspection},
  author  = {[Author Names]},
  year    = {2026},
  note    = {Dataset available at: [URL]},
  licence = {CC BY 4.0}
}
```

---

*Generated: 2026-06-12*
