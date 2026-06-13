"""
Generate deterministic synthetic Duan RGB degradations after data splitting.

Output is deliberately separate from the derived clean data:

    <experiment>/clean/...
    <experiment>/synthetic/<variant>/images/{train,val,test}/...
    <experiment>/synthetic/<variant>/labels/{train,val,test}/...

No geometric operation is applied, so every YOLO label is copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


HAZE_CONFIGS = {
    "haze_light": {"beta": 0.6, "atmosphere": 0.85},
    "haze_medium": {"beta": 1.2, "atmosphere": 0.90},
    "haze_heavy": {"beta": 2.0, "atmosphere": 0.95},
}
FOG_CONFIGS = {
    "fog_light": {"density": 0.30, "blur": 71},
    "fog_medium": {"density": 0.55, "blur": 101},
    "fog_heavy": {"density": 0.80, "blur": 151},
}
LOWLIGHT_CONFIGS = {
    "lowlight_dusk": {"gamma": 1.8, "noise_sigma": 0.015},
    "lowlight_dawn": {"gamma": 2.6, "noise_sigma": 0.030},
    "lowlight_night": {"gamma": 4.0, "noise_sigma": 0.055},
}
MIXED_CONFIGS = {
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
ALL_CONFIGS = {
    **{name: ("haze", cfg) for name, cfg in HAZE_CONFIGS.items()},
    **{name: ("fog", cfg) for name, cfg in FOG_CONFIGS.items()},
    **{name: ("lowlight", cfg) for name, cfg in LOWLIGHT_CONFIGS.items()},
    **{name: ("mixed", cfg) for name, cfg in MIXED_CONFIGS.items()},
}


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def estimate_depth(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    vertical = np.linspace(1.0, 0.2, height, dtype=np.float32)[:, None]
    vertical = np.repeat(vertical, width, axis=1)
    gray = cv2.cvtColor(
        np.clip(image * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0
    dark_prior = 1.0 - cv2.GaussianBlur(gray, (15, 15), 0)
    depth = 0.6 * vertical + 0.4 * dark_prior
    depth = (depth - depth.min()) / (np.ptp(depth) + 1e-8)
    return 0.1 + 0.9 * depth


def add_haze(
    image: np.ndarray,
    depth: np.ndarray,
    beta: float,
    atmosphere: float,
) -> np.ndarray:
    transmission = np.exp(-beta * depth)[:, :, None]
    transmission = np.clip(transmission, 0.05, 1.0)
    return np.clip(
        image * transmission + atmosphere * (1.0 - transmission), 0, 1
    )


def make_fog_field(
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = shape[:2]
    small_h = max(8, height // 24)
    small_w = max(8, width // 24)
    noise = rng.normal(0.5, 0.22, (small_h, small_w)).astype(np.float32)
    mask = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=18, sigmaY=18)
    mask = (mask - mask.min()) / (np.ptp(mask) + 1e-8)
    valley = np.linspace(0.65, 1.15, height, dtype=np.float32)[:, None]
    return np.clip(mask * valley, 0, 1.2)


def add_fog(
    image: np.ndarray,
    fog_field: np.ndarray,
    density: float,
) -> np.ndarray:
    mask = np.clip(fog_field * density, 0, 0.92)[:, :, None]
    fog_color = np.array([0.98, 0.97, 0.95], dtype=np.float32)
    return np.clip(image * (1.0 - mask) + fog_color * mask, 0, 1)


def add_lowlight(
    image: np.ndarray,
    gamma: float,
    noise_sigma: float,
    noise_field: np.ndarray,
) -> np.ndarray:
    dark = np.power(image, gamma)
    output = dark + noise_field * noise_sigma
    if gamma <= 2.2:
        output[:, :, 2] *= 1.04
        output[:, :, 0] *= 0.97
    return np.clip(output, 0, 1)


def make_noise_field(
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = shape[:2]
    small = rng.normal(
        0,
        1,
        (max(8, height // 4), max(8, width // 4), 3),
    ).astype(np.float32)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def degrade_all(
    image: np.ndarray,
    variants: list[str],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    depth = estimate_depth(image)
    fog_field = make_fog_field(image.shape, rng)
    noise_field = make_noise_field(image.shape, rng)
    outputs = {}
    for variant in variants:
        family, config = ALL_CONFIGS[variant]
        if family == "haze":
            output = add_haze(
                image, depth, config["beta"], config["atmosphere"]
            )
        elif family == "fog":
            output = add_fog(image, fog_field, config["density"])
        elif family == "lowlight":
            output = add_lowlight(
                image, config["gamma"], config["noise_sigma"], noise_field
            )
        elif family == "mixed":
            hazy = add_haze(
                image, depth, config["beta"], config["atmosphere"]
            )
            output = add_lowlight(
                hazy, config["gamma"], config["noise_sigma"], noise_field
            )
        else:
            raise ValueError(f"Unknown degradation family: {family}")
        outputs[variant] = output
    return outputs


def write_yaml(
    variant_root: Path,
    train_list: Path | None = None,
    val_list: Path | None = None,
) -> None:
    if train_list and val_list:
        text = (
            f"path: {variant_root.parent.parent.as_posix()}\n"
            f"train: {train_list.as_posix()}\n"
            f"val: {val_list.as_posix()}\n"
            "test: clean/images/test\n"
            "names:\n"
            "  0: piping\n"
        )
    else:
        text = (
            f"path: {variant_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: piping\n"
        )
    (variant_root / "data.yaml").write_text(text, encoding="utf-8")


def generate(args: argparse.Namespace) -> None:
    experiment = args.experiment.resolve()
    clean_root = experiment / "clean"
    synthetic_root = experiment / "synthetic"
    if args.reset and synthetic_root.exists():
        shutil.rmtree(synthetic_root)
    synthetic_root.mkdir(parents=True, exist_ok=True)

    variants = (
        args.variants if args.variants else sorted(ALL_CONFIGS)
    )
    unknown = sorted(set(variants) - set(ALL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    rows = []
    for split in ("train", "val", "test"):
        image_paths = sorted((clean_root / "images" / split).glob("*"))
        if args.limit:
            image_paths = image_paths[: args.limit]
        for image_index, image_path in enumerate(image_paths, start=1):
            label_path = clean_root / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}")
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Cannot read {image_path}")
            image = image_bgr.astype(np.float32) / 255.0
            original_shape = image.shape
            source_label = label_path.read_bytes()
            rng = np.random.default_rng(
                stable_seed(str(args.seed), split, image_path.name)
            )
            degraded_outputs = degrade_all(image, variants, rng)
            for variant, output in degraded_outputs.items():
                family, config = ALL_CONFIGS[variant]
                image_out = (
                    synthetic_root / variant / "images" / split / image_path.name
                )
                label_out = (
                    synthetic_root
                    / variant
                    / "labels"
                    / split
                    / f"{image_path.stem}.txt"
                )
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(image_out),
                    np.clip(output * 255, 0, 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                label_out.write_bytes(source_label)
                if not image_out.exists() or image_out.stat().st_size == 0:
                    raise RuntimeError(f"Image write failed for {image_out}")
                if label_out.read_bytes() != source_label:
                    raise RuntimeError(f"Label changed for {label_out}")
                rows.append(
                    {
                        "split": split,
                        "variant": variant,
                        "family": family,
                        "source_image": str(image_path),
                        "synthetic_image": str(image_out),
                        "label": str(label_out),
                        "width": original_shape[1],
                        "height": original_shape[0],
                    }
                )
            if image_index % 25 == 0 or image_index == len(image_paths):
                print(
                    f"{split}: generated {image_index}/{len(image_paths)} "
                    f"source images x {len(variants)} variants"
                )

    for variant in variants:
        write_yaml(synthetic_root / variant)

    metadata_dir = experiment / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "synthetic_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Robust training sees clean and synthetic train/val, but never test.
    train_paths = sorted((clean_root / "images" / "train").glob("*"))
    val_paths = sorted((clean_root / "images" / "val").glob("*"))
    for variant in variants:
        train_paths.extend(
            sorted((synthetic_root / variant / "images" / "train").glob("*"))
        )
        val_paths.extend(
            sorted((synthetic_root / variant / "images" / "val").glob("*"))
        )
    lists_dir = experiment / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    train_list = lists_dir / "robust_train.txt"
    val_list = lists_dir / "robust_val.txt"
    train_list.write_text(
        "\n".join(path.resolve().as_posix() for path in train_paths) + "\n",
        encoding="utf-8",
    )
    val_list.write_text(
        "\n".join(path.resolve().as_posix() for path in val_paths) + "\n",
        encoding="utf-8",
    )
    robust_yaml = experiment / "robust_data.yaml"
    robust_yaml.write_text(
        f"path: {experiment.as_posix()}\n"
        f"train: {train_list.as_posix()}\n"
        f"val: {val_list.as_posix()}\n"
        "test: clean/images/test\n"
        "names:\n"
        "  0: piping\n",
        encoding="utf-8",
    )

    report = {
        "experiment": str(experiment),
        "synthetic_root": str(synthetic_root),
        "variants": variants,
        "variant_count": len(variants),
        "generated_images": len(rows),
        "labels_preserved_byte_for_byte": True,
        "geometry_preserved": True,
        "robust_train_images": len(train_paths),
        "robust_val_images": len(val_paths),
    }
    (metadata_dir / "synthetic_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--variants", nargs="*", choices=sorted(ALL_CONFIGS))
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
