"""Export portable, verifiable metadata for every generated synthetic image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


CONFIGS = {
    "haze_light": {"beta": 0.6, "atmosphere": 0.85},
    "haze_medium": {"beta": 1.2, "atmosphere": 0.90},
    "haze_heavy": {"beta": 2.0, "atmosphere": 0.95},
    "fog_light": {"density": 0.30, "blur": 71},
    "fog_medium": {"density": 0.55, "blur": 101},
    "fog_heavy": {"density": 0.80, "blur": 151},
    "lowlight_dusk": {"gamma": 1.8, "noise_sigma": 0.015},
    "lowlight_dawn": {"gamma": 2.6, "noise_sigma": 0.030},
    "lowlight_night": {"gamma": 4.0, "noise_sigma": 0.055},
    "mixed_light": {
        "beta": 0.6,
        "atmosphere": 0.85,
        "gamma": 1.8,
        "noise_sigma": 0.015,
    },
    "mixed_medium": {
        "beta": 1.1,
        "atmosphere": 0.90,
        "gamma": 2.5,
        "noise_sigma": 0.030,
    },
    "mixed_heavy": {
        "beta": 1.8,
        "atmosphere": 0.94,
        "gamma": 3.4,
        "noise_sigma": 0.050,
    },
}

FIELDS = [
    "synthetic_id",
    "source_name",
    "capture_group",
    "split",
    "family",
    "variant",
    "level",
    "generation_seed",
    "jpeg_quality",
    "width",
    "height",
    "class_id",
    "x_center",
    "y_center",
    "box_width",
    "box_height",
    "beta",
    "atmosphere",
    "density",
    "blur",
    "gamma",
    "noise_sigma",
    "source_original_sha256",
    "derived_clean_sha256",
    "synthetic_sha256",
    "label_sha256",
    "synthetic_size_bytes",
    "clean_image",
    "synthetic_image",
    "yolo_label",
    "config_json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def portable_path(path: Path, experiment: Path) -> str:
    return path.resolve().relative_to(experiment).as_posix()


def read_single_yolo_box(path: Path) -> tuple[str, str, str, str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) != 1:
        raise ValueError(f"Expected one YOLO box in {path}, found {len(lines)}")
    fields = lines[0].split()
    if len(fields) != 5:
        raise ValueError(f"Expected five YOLO fields in {path}: {lines[0]}")
    return tuple(fields)  # type: ignore[return-value]


def export(args: argparse.Namespace) -> None:
    experiment = args.experiment.resolve()
    metadata = experiment / "metadata"
    synthetic_manifest = metadata / "synthetic_manifest.csv"
    clean_manifest = metadata / "clean_manifest.csv"

    with clean_manifest.open(encoding="utf-8", newline="") as handle:
        clean_rows = {
            row["source_name"]: row
            for row in csv.DictReader(handle)
        }
    with synthetic_manifest.open(encoding="utf-8", newline="") as handle:
        synthetic_rows = list(csv.DictReader(handle))

    split_order = {"train": 0, "val": 1, "test": 2}
    synthetic_rows.sort(
        key=lambda row: (
            split_order[row["split"]],
            Path(row["source_image"]).name,
            row["variant"],
        )
    )

    output_rows = []
    for index, row in enumerate(synthetic_rows, start=1):
        source_name = Path(row["source_image"]).name
        clean = clean_rows[source_name]
        if clean["split"] != row["split"]:
            raise ValueError(f"Split mismatch for {source_name}")

        variant = row["variant"]
        config = CONFIGS[variant]
        synthetic_image = Path(row["synthetic_image"])
        label = Path(row["label"])
        clean_image = Path(row["source_image"])
        for path in (synthetic_image, label, clean_image):
            if not path.is_file():
                raise FileNotFoundError(path)

        class_id, x_center, y_center, box_width, box_height = (
            read_single_yolo_box(label)
        )
        level = variant.split("_", maxsplit=1)[1]
        output_rows.append(
            {
                "synthetic_id": f"syn_{index:06d}",
                "source_name": source_name,
                "capture_group": clean["group"],
                "split": row["split"],
                "family": row["family"],
                "variant": variant,
                "level": level,
                "generation_seed": stable_seed(
                    str(args.seed), row["split"], source_name
                ),
                "jpeg_quality": args.jpeg_quality,
                "width": row["width"],
                "height": row["height"],
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "box_width": box_width,
                "box_height": box_height,
                "beta": config.get("beta", ""),
                "atmosphere": config.get("atmosphere", ""),
                "density": config.get("density", ""),
                "blur": config.get("blur", ""),
                "gamma": config.get("gamma", ""),
                "noise_sigma": config.get("noise_sigma", ""),
                "source_original_sha256": clean["source_sha256"],
                "derived_clean_sha256": clean["derived_clean_sha256"],
                "synthetic_sha256": sha256(synthetic_image),
                "label_sha256": sha256(label),
                "synthetic_size_bytes": synthetic_image.stat().st_size,
                "clean_image": portable_path(clean_image, experiment),
                "synthetic_image": portable_path(synthetic_image, experiment),
                "yolo_label": portable_path(label, experiment),
                "config_json": json.dumps(
                    config, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows):,} rows to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("synthetic_images_metadata.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
