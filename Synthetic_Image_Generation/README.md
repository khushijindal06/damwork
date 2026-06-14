# Synthetic Image Generation

This folder contains the code and metadata for the 7,500 Duan RGB synthetic
images generated from 625 clean labelled source images.

## Files

- `generate_synthetic_images.py` is the exact code used to create all haze,
  fog, low-light, and mixed variants.
- `export_synthetic_metadata.py` builds the portable metadata CSV and verifies
  every referenced clean image, synthetic image, and YOLO label.
- `synthetic_images_metadata.csv` contains one row per synthetic image.
- `requirements.txt` lists the image-generation dependencies.

## Dataset Coverage

Each of the 625 source images has 12 generated versions:

- haze: light, medium, heavy
- fog: light, medium, heavy
- low light: dusk, dawn, night
- mixed haze and low light: light, medium, heavy

The synthetic images are stored locally under:

```text
E:/Downloads/10896178/Duan_RGB_Experiment/synthetic
```

They are not committed to GitHub because the image set is approximately
3.76 GB. The CSV uses paths relative to `Duan_RGB_Experiment`, so it remains
portable when the dataset root is moved.

## Metadata Columns

The CSV records:

- source filename, capture group, and train/validation/test split
- degradation family, variant, level, deterministic seed, and JPEG quality
- all applicable degradation parameters
- image dimensions and normalized YOLO bounding-box values
- source, derived-clean, synthetic-image, and label SHA-256 values
- synthetic file size and portable clean/image/label paths

Regenerate the metadata after generating the images:

```powershell
python Synthetic_Image_Generation/export_synthetic_metadata.py `
  --experiment E:/Downloads/10896178/Duan_RGB_Experiment
```
