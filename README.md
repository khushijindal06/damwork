# Duan UAV Embankment Piping Dataset Files

This repository contains an unmodified copy of the files published with:

> Quntao Duan, Baili Chen, and Lihui Luo,
> **Data for UAV detection of river embankment piping**,
> Zenodo, 2024. https://doi.org/10.5281/zenodo.10896178

The source dataset is licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contents

| Path | Contents |
| --- | --- |
| `UAV_piping_label_data/` | 986 labelled JPG images: 625 visible and 361 infrared. |
| `Model_detection_result/` | 341 model-result JPG images: 222 visible and 119 infrared. |
| `Data_Description.docx` | Description supplied with the source dataset. |
| `Experiments_statistics.xlsx` | Experiment statistics supplied with the source dataset. |
| `Uav_piping_data_information.xlsx` | Image and collection metadata supplied with the source dataset. |

All 1,330 files above were verified as byte-for-byte identical to the files in
the downloaded Zenodo archive.

## Source Verification

```text
Archive:  UAV_piping_image.zip
Size:     290827639 bytes
MD5:      41a65d1914a0649bc96285211bdc83a0
Record:   https://zenodo.org/records/10896178
DOI:      10.5281/zenodo.10896178
License:  CC BY 4.0
```

## Privacy Scope

This repository does **not** include privately curated or derived work such as:

- converted YOLO labels or inpainted images
- train, validation, or test splits
- synthetic or enhanced images
- generated metadata manifests
- model weights, training runs, or evaluation results
- private experiment notebooks or processing scripts

Those materials should remain in private storage and are excluded by
`.gitignore`.
