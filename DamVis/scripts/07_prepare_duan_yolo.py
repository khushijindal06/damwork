"""
Prepare Duan et al.'s visible UAV piping images for YOLO experiments.

The Zenodo archive (10.5281/zenodo.10896178) contains visible JPG files with
the class name and bounding rectangle rendered into the pixels. This script:

1. Detects the rendered "Piping" annotation and black rectangle.
2. Converts the rectangle to a one-class YOLO label.
3. Inpaints only the rendered text and rectangle border to create a derived
   clean image. The original archive remains unchanged.
4. Assigns whole site/date capture groups to train, validation, or test.
5. Writes audit metadata and verifies that no group leaks across splits.

Example:
    python 07_prepare_duan_yolo.py ^
      --source "E:/Downloads/10896178/UAV_piping_image/UAV_piping_label_data/Visible" ^
      --output "E:/Downloads/10896178/Duan_RGB_Experiment"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
ZENODO_RECORD = "https://zenodo.org/records/10896178"
ZENODO_DOI = "10.5281/zenodo.10896178"
EXPECTED_ARCHIVE_MD5 = "41a65d1914a0649bc96285211bdc83a0"


@dataclass(frozen=True)
class Detection:
    text_x: int
    text_y: int
    text_w: int
    text_h: int
    box_x: int
    box_y: int
    box_w: int
    box_h: int
    score: float


def green_mask(image: np.ndarray) -> np.ndarray:
    """Return a mask for the rendered green annotation text."""
    b, g, r = cv2.split(image)
    mask = (
        (g >= 125)
        & (g.astype(np.int16) - r.astype(np.int16) >= 45)
        & (g.astype(np.int16) - b.astype(np.int16) >= 70)
        & (r <= 165)
        & (b <= 135)
    )
    return mask.astype(np.uint8) * 255


def text_candidates(image: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    mask = green_mask(image)
    joined = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        iterations=1,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    candidates = []
    for x, y, w, h, area in stats[1:count]:
        if 12 <= w <= 160 and 4 <= h <= 55 and area >= 35:
            candidates.append((int(x), int(y), int(w), int(h), int(area)))
    return candidates


def rectangle_candidates(
    image: np.ndarray,
    text: tuple[int, int, int, int, int],
) -> list[tuple[int, int, int, int, float]]:
    """Find black rectangular components below one green-text candidate."""
    tx, ty, tw, th, _ = text
    image_h, image_w = image.shape[:2]
    roi_x1 = max(0, tx - 50)
    roi_y1 = max(0, ty + th - 20)
    roi_x2 = min(image_w, tx + max(260, tw * 4))
    roi_y2 = min(image_h, ty + th + 320)
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    found: dict[tuple[int, int, int, int], float] = {}

    # Multiple thresholds handle boxes drawn over both bright soil and dark water.
    for threshold in (15, 20, 25, 30, 45, 60, 90):
        dark = (gray < threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark, connectivity=8
        )
        for component_id, (x, y, w, h, area) in enumerate(
            stats[1:count], start=1
        ):
            if not (8 <= w <= 320 and 8 <= h <= 320):
                continue
            if area < max(18, 0.4 * (w + h)):
                continue
            component = labels[y : y + h, x : x + w] == component_id
            border = np.concatenate(
                (
                    component[0, :],
                    component[-1, :],
                    component[:, 0],
                    component[:, -1],
                )
            )
            border_support = float(border.mean())
            if border_support < 0.20:
                continue
            score = 100.0 * (1.0 - border_support) + threshold * 0.05
            key = (int(x + roi_x1), int(y + roi_y1), int(w), int(h))
            found[key] = min(found.get(key, float("inf")), score)

    return [(*box, score) for box, score in found.items()]


def detect_annotation(image: np.ndarray) -> Detection:
    ranked = []
    for text in text_candidates(image):
        tx, ty, tw, th, area = text
        for x, y, w, h, component_score in rectangle_candidates(image, text):
            dx = abs(x - tx)
            vertical_gap = y - (ty + th)
            if dx > max(55, tw):
                continue
            if not (-15 <= vertical_gap <= 100):
                continue
            if w > max(260, tw * 4) or h > 320:
                continue
            aspect = w / h
            if not 0.15 <= aspect <= 6:
                continue
            score = (
                dx * 1.8
                + abs(vertical_gap - 8) * 1.2
                + component_score * 0.6
                + max(0, 20 - min(w, h)) * 2
                - min(area, 900) * 0.002
            )
            ranked.append((score, text, (x, y, w, h)))
    if not ranked:
        raise ValueError("Could not locate the rendered Piping annotation")
    score, text, box = min(ranked, key=lambda item: item[0])
    tx, ty, tw, th, _ = text
    x, y, w, h = box
    return Detection(tx, ty, tw, th, x, y, w, h, float(score))


def annotation_mask(image: np.ndarray, detection: Detection) -> np.ndarray:
    """Mask only the rendered green text and black rectangle border."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    text_x1 = max(0, min(detection.text_x, detection.box_x) - 15)
    text_y1 = max(0, detection.text_y - 10)
    text_x2 = min(
        image.shape[1],
        max(
            detection.text_x + detection.text_w,
            detection.box_x + detection.box_w,
        )
        + 30,
    )
    text_y2 = min(image.shape[0], detection.box_y + 5)
    cv2.rectangle(
        mask,
        (text_x1, text_y1),
        (max(text_x1, text_x2 - 1), max(text_y1, text_y2 - 1)),
        255,
        thickness=-1,
    )

    x, y, w, h = (
        detection.box_x,
        detection.box_y,
        detection.box_w,
        detection.box_h,
    )
    cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 255, thickness=5)
    return mask


def make_derived_clean(image: np.ndarray, detection: Detection) -> np.ndarray:
    mask = annotation_mask(image, detection)
    return cv2.inpaint(image, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


def group_from_filename(path: Path) -> str:
    """Use the site/date token, for example AH230615, as leakage group."""
    match = re.match(r"^P_([A-Z]{2}\d{6})_", path.stem)
    if not match:
        raise ValueError(f"Cannot derive site/date group from {path.name}")
    return match.group(1)


def stable_tie_break(seed: int, group: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def assign_groups(
    group_sizes: dict[str, int],
    seed: int,
    ratios: dict[str, float] = SPLIT_RATIOS,
) -> dict[str, str]:
    """Greedily balance complete groups toward requested image ratios."""
    total = sum(group_sizes.values())
    targets = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: 0 for split in ratios}
    assignments: dict[str, str] = {}
    groups = sorted(
        group_sizes,
        key=lambda group: (-group_sizes[group], stable_tie_break(seed, group)),
    )

    for group in groups:
        size = group_sizes[group]
        split = max(
            ratios,
            key=lambda name: (
                (targets[name] - counts[name]) / max(targets[name], 1),
                -counts[name],
            ),
        )
        assignments[group] = split
        counts[split] += size

    missing = [split for split in ratios if split not in assignments.values()]
    if missing:
        raise RuntimeError(f"Split assignment left empty splits: {missing}")
    return assignments


def yolo_line(detection: Detection, image_width: int, image_height: int) -> str:
    x_center = (detection.box_x + detection.box_w / 2) / image_width
    y_center = (detection.box_y + detection.box_h / 2) / image_height
    width = detection.box_w / image_width
    height = detection.box_h / image_height
    values = (x_center, y_center, width, height)
    if not all(0 < value <= 1 for value in values):
        raise ValueError(f"Invalid normalized bounding box: {values}")
    return "0 " + " ".join(f"{value:.8f}" for value in values) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    bits = resized >= resized.mean()
    return np.packbits(bits).tobytes().hex()


def write_dataset_yaml(output_root: Path) -> None:
    clean_root = output_root / "clean"
    yaml_text = (
        f"path: {clean_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: piping\n"
    )
    (clean_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def save_audit_image(
    source: np.ndarray,
    clean: np.ndarray,
    detection: Detection,
    destination: Path,
) -> None:
    annotated = source.copy()
    x, y, w, h = (
        detection.box_x,
        detection.box_y,
        detection.box_w,
        detection.box_h,
    )
    cv2.rectangle(annotated, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 3)
    canvas = np.concatenate((annotated, clean), axis=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def verify_no_leakage(rows: list[dict[str, object]]) -> dict[str, object]:
    group_splits: dict[str, set[str]] = {}
    exact_hash_splits: dict[str, set[str]] = {}
    average_hash_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(str(row["group"]), set()).add(str(row["split"]))
        exact_hash_splits.setdefault(
            str(row["derived_clean_sha256"]), set()
        ).add(str(row["split"]))
        average_hash_splits.setdefault(
            str(row["average_hash"]), set()
        ).add(str(row["split"]))

    leaking_groups = sorted(
        group for group, splits in group_splits.items() if len(splits) > 1
    )
    leaking_exact_hashes = sorted(
        value for value, splits in exact_hash_splits.items() if len(splits) > 1
    )
    cross_split_average_hashes = sorted(
        value for value, splits in average_hash_splits.items() if len(splits) > 1
    )
    if leaking_groups:
        raise RuntimeError(f"Capture-group leakage detected: {leaking_groups}")
    if leaking_exact_hashes:
        raise RuntimeError(
            f"Byte-identical derived images cross splits: "
            f"{len(leaking_exact_hashes)}"
        )
    return {
        "capture_groups": len(group_splits),
        "leaking_capture_groups": leaking_groups,
        "cross_split_exact_hashes": leaking_exact_hashes,
        "cross_split_average_hash_collisions": cross_split_average_hashes,
    }


def prepare(args: argparse.Namespace) -> None:
    source_dir = args.source.resolve()
    output_root = args.output.resolve()
    image_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    group_sizes = Counter(group_from_filename(path) for path in image_paths)
    group_assignments = assign_groups(dict(group_sizes), args.seed)
    metadata_dir = output_root / "metadata"
    audit_dir = metadata_dir / "audit_samples"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    randomizer = random.Random(args.seed)
    audit_names = set(
        path.name
        for path in randomizer.sample(
            image_paths, min(args.audit_samples, len(image_paths))
        )
    )

    for index, source_path in enumerate(image_paths, start=1):
        try:
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not read image")
            detection = detect_annotation(image)
            clean = make_derived_clean(image, detection)
            image_h, image_w = image.shape[:2]
            group = group_from_filename(source_path)
            split = group_assignments[group]
            image_out = output_root / "clean" / "images" / split / source_path.name
            label_out = (
                output_root / "clean" / "labels" / split / f"{source_path.stem}.txt"
            )
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(image_out),
                clean,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            label_out.write_text(
                yolo_line(detection, image_w, image_h), encoding="utf-8"
            )
            row = {
                "source_name": source_path.name,
                "source_sha256": sha256(source_path),
                "derived_clean_sha256": sha256(image_out),
                "average_hash": average_hash(clean),
                "group": group,
                "split": split,
                "image_width": image_w,
                "image_height": image_h,
                **asdict(detection),
                "clean_image": str(image_out),
                "yolo_label": str(label_out),
            }
            rows.append(row)
            if source_path.name in audit_names:
                save_audit_image(
                    image,
                    clean,
                    detection,
                    audit_dir / source_path.name,
                )
            if index % 25 == 0 or index == len(image_paths):
                print(f"Prepared {index}/{len(image_paths)} images")
        except Exception as exc:
            failures.append({"source_name": source_path.name, "error": str(exc)})
            if not args.allow_failures:
                raise RuntimeError(f"{source_path.name}: {exc}") from exc

    if failures:
        failure_path = metadata_dir / "conversion_failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_name", "error"])
            writer.writeheader()
            writer.writerows(failures)

    leakage = verify_no_leakage(rows)
    manifest_path = metadata_dir / "clean_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_dataset_yaml(output_root)
    split_counts = Counter(str(row["split"]) for row in rows)
    group_counts = Counter(group_assignments.values())
    report = {
        "source": {
            "zenodo_record": ZENODO_RECORD,
            "doi": ZENODO_DOI,
            "license": "CC BY 4.0",
            "expected_archive_md5": EXPECTED_ARCHIVE_MD5,
            "source_directory": str(source_dir),
        },
        "output_root": str(output_root),
        "images_discovered": len(image_paths),
        "images_prepared": len(rows),
        "conversion_failures": len(failures),
        "split_seed": args.seed,
        "requested_split_ratios": SPLIT_RATIOS,
        "split_image_counts": dict(split_counts),
        "split_group_counts": dict(group_counts),
        "group_assignments": group_assignments,
        "leakage_audit": leakage,
        "cleaning_note": (
            "Derived clean images were created by inpainting only the rendered "
            "green class text and black rectangle border. Original files were "
            "not modified."
        ),
    }
    (metadata_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Folder containing Duan visible annotated JPG files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Experiment root. Clean data is written below output/clean.",
    )
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--audit-samples", type=int, default=30)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Continue and report images whose annotations cannot be converted.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete an existing clean and metadata output before preparing.",
    )
    args = parser.parse_args()
    if args.reset:
        for child in (args.output / "clean", args.output / "metadata"):
            if child.exists():
                shutil.rmtree(child)
    return args


if __name__ == "__main__":
    prepare(parse_args())
