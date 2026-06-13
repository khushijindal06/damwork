# UAV Piping Field Experiment Metadata

This repository contains documentation and metadata for UAV piping-image
collection experiments conducted in 2022 and 2023.

The image files themselves are not currently included in this repository.

## Repository Contents

| File | Description |
| --- | --- |
| `Data_Description.docx` | Short description of the UAV piping image dataset. |
| `Experiments_statistics.xlsx` | Experiment dates, sites, and simulated-scenario counts. |
| `Uav_piping_data_information.xlsx` | Per-image collection metadata, including weather, location, image name, and altitude values. |

## Dataset Summary

- 104 simulated scenarios across 29 experiment-date rows.
- 106 unique image-name records.
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

Some altitude cells contain multiple values separated by the ideographic comma
(`、`), while others contain a single numeric value. The altitude unit is not
specified in the current documentation.

Image names encode experimental conditions, but a complete definition of the
abbreviations is not included.

## Known Gaps

- `Data_Description.docx` refers to
  `Uav_piping_data_collection_information.xlsx`; the file currently committed
  to this repository is named `Uav_piping_data_information.xlsx`.
- The Word description says the dataset contains JPG training and test images,
  but those images and their train/test split are not present here.
- `Uav_piping_data_information.xlsx` contains the value `Riany`, which appears
  to be a misspelling of `Rainy`.
- Several date labels use inconsistent ordinal suffixes, such as `May 31th` and
  `June 21th`.
- The altitude unit and the image-name abbreviation scheme need documentation
  before the metadata can be interpreted unambiguously.

## Recommended Next Additions

1. Add the image dataset or a stable download location.
2. Document the training/test split and licensing terms.
3. Add a data dictionary for every spreadsheet column and image-name code.
4. Normalize date, weather, and altitude fields in a separate processed file
   while preserving the uploaded source files unchanged.
