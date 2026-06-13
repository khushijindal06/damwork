"""
Create enhanced copies of synthetic test images for mAP recovery evaluation.

Methods are classical, deterministic baselines:
  - clahe: local contrast enhancement in LAB space
  - gamma: automatic luminance-targeted gamma correction
  - retinex: multi-scale Retinex for low-light restoration
  - dcp: dark-channel-prior dehazing
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


METHODS = ("clahe", "gamma", "retinex", "dcp")


def clahe_enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(
        cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR
    )


def gamma_enhance(image: np.ndarray) -> np.ndarray:
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() / 255.0
    luminance = float(np.clip(luminance, 0.02, 0.98))
    target = 0.55
    gamma = np.log(target) / np.log(luminance)
    gamma = float(np.clip(gamma, 0.35, 1.25))
    table = np.clip(
        (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return cv2.LUT(image, table)


def retinex_enhance(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    if scale < 1.0:
        working = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = image
    source = working.astype(np.float32) + 1.0
    result = np.zeros_like(source)
    for sigma in (12, 55):
        blur = cv2.GaussianBlur(source, (0, 0), sigma)
        result += np.log(source) - np.log(blur + 1.0)
    result /= 2.0
    output = np.empty_like(result)
    for channel in range(3):
        plane = result[:, :, channel]
        low, high = np.percentile(plane, (1, 99))
        output[:, :, channel] = np.clip(
            (plane - low) / (high - low + 1e-6) * 255.0, 0, 255
        )
    output = output.astype(np.uint8)
    if scale < 1.0:
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)
    return output


def dcp_enhance(image: np.ndarray) -> np.ndarray:
    source = image.astype(np.float32) / 255.0
    dark = np.min(source, axis=2)
    dark = cv2.erode(dark, np.ones((15, 15), np.uint8))
    flat_dark = dark.reshape(-1)
    count = max(1, int(flat_dark.size * 0.001))
    indices = np.argpartition(flat_dark, -count)[-count:]
    flat_source = source.reshape(-1, 3)
    atmosphere = flat_source[indices].mean(axis=0)
    atmosphere = np.maximum(atmosphere, 0.55)
    normalized = source / atmosphere[None, None, :]
    transmission = 1.0 - 0.95 * cv2.erode(
        np.min(normalized, axis=2), np.ones((15, 15), np.uint8)
    )
    transmission = cv2.GaussianBlur(transmission, (21, 21), 0)
    transmission = np.clip(transmission, 0.15, 1.0)[:, :, None]
    restored = (source - atmosphere) / transmission + atmosphere
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def enhance(image: np.ndarray, method: str) -> np.ndarray:
    if method == "clahe":
        return clahe_enhance(image)
    if method == "gamma":
        return gamma_enhance(image)
    if method == "retinex":
        return retinex_enhance(image)
    if method == "dcp":
        return dcp_enhance(image)
    raise ValueError(method)


def run(args: argparse.Namespace) -> None:
    experiment = args.experiment.resolve()
    synthetic_root = experiment / "synthetic"
    enhanced_root = experiment / "enhanced"
    if args.reset and enhanced_root.exists():
        shutil.rmtree(enhanced_root)
    variants = (
        args.variants
        if args.variants
        else sorted(path.name for path in synthetic_root.iterdir() if path.is_dir())
    )
    rows = []
    for method in args.methods:
        for variant in variants:
            source_dir = synthetic_root / variant / "images" / "test"
            label_dir = synthetic_root / variant / "labels" / "test"
            image_paths = sorted(source_dir.glob("*"))
            if args.limit:
                image_paths = image_paths[: args.limit]
            for index, image_path in enumerate(image_paths, start=1):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot read {image_path}")
                output = enhance(image, method)
                image_out = (
                    enhanced_root
                    / method
                    / variant
                    / "images"
                    / "test"
                    / image_path.name
                )
                label_out = (
                    enhanced_root
                    / method
                    / variant
                    / "labels"
                    / "test"
                    / f"{image_path.stem}.txt"
                )
                source_label = label_dir / f"{image_path.stem}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(image_out),
                    output,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                shutil.copy2(source_label, label_out)
                rows.append(
                    {
                        "method": method,
                        "variant": variant,
                        "source_image": str(image_path),
                        "enhanced_image": str(image_out),
                        "label": str(label_out),
                    }
                )
                if index % 25 == 0 or index == len(image_paths):
                    print(
                        f"{method}/{variant}: {index}/{len(image_paths)} test images"
                    )
            variant_root = enhanced_root / method / variant
            (variant_root / "data.yaml").write_text(
                f"path: {variant_root.as_posix()}\n"
                "train: images/test\n"
                "val: images/test\n"
                "test: images/test\n"
                "names:\n"
                "  0: piping\n",
                encoding="utf-8",
            )

    metadata_dir = experiment / "metadata"
    manifest = metadata_dir / "enhancement_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "enhanced_root": str(enhanced_root),
        "methods": args.methods,
        "variants": variants,
        "enhanced_images": len(rows),
    }
    (metadata_dir / "enhancement_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
