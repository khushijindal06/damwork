"""
Train clean-only and degradation-robust YOLO models, then evaluate mAP recovery.

The script expects an experiment prepared by scripts 07-09. Training data never
contains test capture groups. Results are written to:

    <experiment>/runs/
    <experiment>/metadata/yolo_metrics.csv
    <experiment>/metadata/yolo_recovery.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def metric_row(model_name: str, condition: str, result) -> dict[str, object]:
    box = result.box
    return {
        "model": model_name,
        "condition": condition,
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


def evaluate_dataset(
    model: YOLO,
    model_name: str,
    condition: str,
    data_yaml: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    result = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
    )
    row = metric_row(model_name, condition, result)
    print(row)
    return row


def train_models(args: argparse.Namespace) -> dict[str, Path]:
    experiment = args.experiment.resolve()
    runs = experiment / "runs"
    checkpoints: dict[str, Path] = {}
    specifications = {
        "clean": experiment / "clean" / "data.yaml",
        "robust": experiment / "robust_data.yaml",
    }
    for name, data_yaml in specifications.items():
        if name == "robust" and args.skip_robust:
            continue
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            project=str(runs),
            name=name,
            exist_ok=True,
            patience=args.patience,
            cache=args.cache,
        )
        checkpoint = runs / name / "weights" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Training did not produce {checkpoint}")
        checkpoints[name] = checkpoint
    return checkpoints


def existing_checkpoints(args: argparse.Namespace) -> dict[str, Path]:
    experiment = args.experiment.resolve()
    candidates = {
        "clean": args.clean_weights
        or experiment / "runs" / "clean" / "weights" / "best.pt",
        "robust": args.robust_weights
        or experiment / "runs" / "robust" / "weights" / "best.pt",
    }
    return {
        name: path.resolve()
        for name, path in candidates.items()
        if path and path.exists()
    }


def evaluate(args: argparse.Namespace, checkpoints: dict[str, Path]) -> None:
    experiment = args.experiment.resolve()
    synthetic_root = experiment / "synthetic"
    enhanced_root = experiment / "enhanced"
    metadata_dir = experiment / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    synthetic_variants = (
        sorted(path.name for path in synthetic_root.iterdir() if path.is_dir())
        if synthetic_root.exists()
        else []
    )
    enhanced_conditions = sorted(
        (method.name, variant.name)
        for method in enhanced_root.iterdir()
        if method.is_dir()
        for variant in method.iterdir()
        if variant.is_dir()
    ) if enhanced_root.exists() else []

    for model_name, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint))
        rows.append(
            evaluate_dataset(
                model,
                model_name,
                "clean",
                experiment / "clean" / "data.yaml",
                args,
            )
        )
        for variant in synthetic_variants:
            rows.append(
                evaluate_dataset(
                    model,
                    model_name,
                    variant,
                    synthetic_root / variant / "data.yaml",
                    args,
                )
            )
        for method, variant in enhanced_conditions:
            rows.append(
                evaluate_dataset(
                    model,
                    model_name,
                    f"{method}/{variant}",
                    enhanced_root / method / variant / "data.yaml",
                    args,
                )
            )

    metrics_path = metadata_dir / "yolo_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {
        (str(row["model"]), str(row["condition"])): float(row["map50_95"])
        for row in rows
    }
    recovery_rows = []
    for model_name in checkpoints:
        clean_map = lookup[(model_name, "clean")]
        for method, variant in enhanced_conditions:
            degraded_map = lookup[(model_name, variant)]
            enhanced_map = lookup[(model_name, f"{method}/{variant}")]
            denominator = clean_map - degraded_map
            recovery = (
                (enhanced_map - degraded_map) / denominator
                if denominator > 1e-12
                else None
            )
            recovery_rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "enhancement": method,
                    "clean_map50_95": clean_map,
                    "degraded_map50_95": degraded_map,
                    "enhanced_map50_95": enhanced_map,
                    "absolute_gain": enhanced_map - degraded_map,
                    "recovery_fraction": recovery,
                }
            )
    recovery_path = metadata_dir / "yolo_recovery.csv"
    with recovery_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "variant",
            "enhancement",
            "clean_map50_95",
            "degraded_map50_95",
            "enhanced_map50_95",
            "absolute_gain",
            "recovery_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recovery_rows)

    summary = {
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "metric_rows": len(rows),
        "recovery_rows": len(recovery_rows),
        "metrics_csv": str(metrics_path),
        "recovery_csv": str(recovery_path),
        "recovery_definition": (
            "(enhanced mAP50-95 - degraded mAP50-95) / "
            "(clean mAP50-95 - degraded mAP50-95)"
        ),
    }
    (metadata_dir / "yolo_evaluation_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train checkpoints without running the full evaluation matrix.",
    )
    parser.add_argument("--skip-robust", action="store_true")
    parser.add_argument("--clean-weights", type=Path)
    parser.add_argument("--robust-weights", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    weights = (
        existing_checkpoints(parsed)
        if parsed.skip_training
        else train_models(parsed)
    )
    if not weights:
        raise FileNotFoundError("No trained checkpoints are available")
    if not parsed.train_only:
        evaluate(parsed, weights)
