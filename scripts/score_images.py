#!/usr/bin/env python3
"""Compute the luminance metadata used for the balanced dataset split."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CSV_FIELDS = [
    "filename",
    "domain",
    "width",
    "height",
    "mean_luminance",
    "mean_luminance_normalized",
    "dark_frac",
    "bright_frac",
    "bright_to_dark_ratio",
    "error",
]
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
RESIZE_TO = (100, 100)
DARK_THRESHOLD = 0.22
BRIGHT_THRESHOLD = 0.75
EPSILON = 1e-8


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No supported images found in {root}")
    return paths


def luminance_stats(path: Path) -> tuple[int, int, float, float, float]:
    with Image.open(path) as image:
        width, height = image.size
        rgb = np.asarray(image.convert("RGB").resize(RESIZE_TO), dtype=np.float32) / 255.0

    luminance = (
        LUMA_WEIGHTS[0] * rgb[:, :, 0]
        + LUMA_WEIGHTS[1] * rgb[:, :, 1]
        + LUMA_WEIGHTS[2] * rgb[:, :, 2]
    )
    return (
        width,
        height,
        float(luminance.mean()),
        float((luminance < DARK_THRESHOLD).mean()),
        float((luminance > BRIGHT_THRESHOLD).mean()),
    )


def score_image(task: tuple[str, str, str]) -> dict[str, str | int | float]:
    path_text, relative_name, domain = task
    path = Path(path_text)
    try:
        width, height, mean, dark_frac, bright_frac = luminance_stats(path)
    except Exception as exc:  # Keep corrupt files visible without aborting the complete scan.
        return {
            "filename": relative_name,
            "domain": domain,
            "width": "",
            "height": "",
            "mean_luminance": "",
            "mean_luminance_normalized": "",
            "dark_frac": "",
            "bright_frac": "",
            "bright_to_dark_ratio": "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "filename": relative_name,
        "domain": domain,
        "width": width,
        "height": height,
        "mean_luminance": mean,
        "mean_luminance_normalized": "",
        "dark_frac": dark_frac,
        "bright_frac": bright_frac,
        "bright_to_dark_ratio": bright_frac / (dark_frac + EPSILON),
        "error": "",
    }


def normalize_luminance(rows: list[dict[str, str | int | float]]) -> None:
    valid = [row for row in rows if row["mean_luminance"] != ""]
    if not valid:
        return

    values = [float(row["mean_luminance"]) for row in valid]
    minimum = min(values)
    span = max(values) - minimum
    for row in valid:
        mean = float(row["mean_luminance"])
        row["mean_luminance_normalized"] = 1.0 if span == 0.0 else (mean - minimum) / span


def formatted(row: dict[str, str | int | float]) -> dict[str, str | int]:
    return {
        key: f"{value:.8f}" if isinstance(value, float) else value
        for key, value in row.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure image luminance and write the metadata CSV used by prepare_dataset.py."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", required=True, help="Domain label stored in the CSV, e.g. day.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    root = args.input_dir.resolve()
    paths = image_files(root)
    tasks = [(str(path), path.relative_to(root).as_posix(), args.domain) for path in paths]
    if args.workers == 1:
        rows = [score_image(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(score_image, tasks, chunksize=32))

    normalize_luminance(rows)
    rows.sort(
        key=lambda row: (
            row["mean_luminance_normalized"] == "",
            -float(row["mean_luminance_normalized"] or 0.0),
            str(row["filename"]),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(formatted(row) for row in rows)

    errors = sum(row["error"] != "" for row in rows)
    print(f"Scored {len(rows) - errors} images; errors={errors}; output={args.output}")


if __name__ == "__main__":
    main()
