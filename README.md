# UAV Piping Dataset and DamVis Pipeline

This repository contains UAV piping images, experiment metadata, model
detection results, and the DamVis dataset-generation pipeline.

## Repository Contents

| File | Description |
| --- | --- |
| `Data_Description.docx` | Short description of the UAV piping image dataset. |
| `Experiments_statistics.xlsx` | Experiment dates, sites, and simulated-scenario counts. |
| `Uav_piping_data_information.xlsx` | Per-image collection metadata, including weather, location, image name, and altitude values. |
| `UAV_piping_label_data/` | 986 labeled-data JPG images organized into visible and infrared folders. |
| `Model_detection_result/` | 341 model-result JPG images organized into visible and infrared folders. |
| `DamVis/` | DamVis pipeline scripts and documentation. |
| `damwork.ipynb` | End-to-end Duan RGB workflow: rendered-box conversion, leakage-free grouped splits, synthetic degradation, YOLO training, enhancement, and mAP recovery evaluation. |
| `Duan_YOLO_Code/` | Standalone review bundle containing the notebook, all four Duan workflow scripts, requirements, and execution notes. |

## Dataset Summary

- 104 simulated scenarios across 29 experiment-date rows.
- 106 unique image-name records.
- 1,327 included UAV piping JPG images.
- 11 metadata records from 2022 and 95 from 2023.
- Collection environments include pool, grassland, cropland, and bare land.
- Collection times include forenoon, afternoon, night, and before dawn.
- Weather values include sunny, rainy, and cloudy conditions.

The scenario total in `Experiments_statistics.xlsx` was checked against the
individual rows and matches the stated total of 104.

## Data Conventions

The spreadsheets use blank cells to mean "same as the preceding row" for
fields such as year, date, time, and weather. Forward-fill those columns before
grouping or filtering the data programmatically.

Some altitude cells contain multiple values separated by an ideographic comma,
while others contain a single numeric value. The altitude unit is not specified
in the current documentation.

Image names encode experimental conditions, but a complete definition of the
abbreviations is not included.

## Duan RGB YOLO Experiment

The notebook and `DamVis/scripts/07_*` through `10_*` use the official open
dataset at [Zenodo record 10896178](https://zenodo.org/records/10896178), DOI
`10.5281/zenodo.10896178` (CC BY 4.0).

The published visible JPGs contain the class text and rectangle rendered into
the pixels rather than separate annotation files. The preparation step extracts
that rectangle into a YOLO label and inpaints only the rendered overlay. It
then assigns complete site/date capture groups to train, validation, or test.
Synthetic images are generated only after splitting and are written outside
the clean data:

```text
Duan_RGB_Experiment/
|-- clean/
|-- synthetic/
|-- enhanced/
|-- metadata/
|-- lists/
`-- runs/
```

On the original Windows setup, the notebook defaults to:

```text
E:/Downloads/10896178/Duan_RGB_Experiment
```

## Known Gaps

- `Data_Description.docx` refers to
  `Uav_piping_data_collection_information.xlsx`; the file currently committed
  to this repository is named `Uav_piping_data_information.xlsx`.
- The UAV piping images are included, but their training/test split is not
  documented.
- `Uav_piping_data_information.xlsx` contains the value `Riany`, which appears
  to be a misspelling of `Rainy`.
- Several date labels use inconsistent ordinal suffixes, such as `May 31th` and
  `June 21th`.
- The altitude unit and the image-name abbreviation scheme need documentation
  before the metadata can be interpreted unambiguously.
- `DamVis/README.md` describes a previous 39-image pilot, but the uploaded
  DamVis folder contains only pipeline code and documentation, not its
  generated `dataset/` directory.

## Recommended Next Additions

1. Document the UAV piping training/test split and licensing terms.
2. Add a data dictionary for every spreadsheet column and image-name code.
3. Add DamVis dependency installation instructions and generated dataset files
   or a stable download location.
4. Normalize date, weather, and altitude fields in a separate processed file
   while preserving the uploaded source files unchanged.
