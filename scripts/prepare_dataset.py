#!/usr/bin/env python3
"""Build reproducible, luminance-balanced unpaired day/night splits."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    filename: str
    source: Path
    score: float
    bucket: int


@dataclass(frozen=True)
class Domain:
    name: str
    csv_path: Path
    image_dir: Path
    train_folder: str
    test_folder: str


def read_rows(domain: Domain, score_column: str) -> list[tuple[str, Path, float]]:
    if not domain.csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {domain.csv_path}")
    if not domain.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {domain.image_dir}")

    rows: list[tuple[str, Path, float]] = []
    with domain.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"filename", score_column}
        if not required.issubset(fields):
            raise ValueError(f"{domain.csv_path} is missing columns: {sorted(required - fields)}")

        for row in reader:
            if (row.get("error") or "").strip():
                continue
            filename = (row.get("filename") or "").strip()
            score_text = (row.get(score_column) or "").strip()
            if not filename or not score_text:
                continue
            source = domain.image_dir / filename
            if not source.is_file():
                continue
            try:
                score = float(score_text)
            except ValueError:
                continue
            rows.append((filename, source, score))

    if not rows:
        raise RuntimeError(f"No usable rows found in {domain.csv_path}")
    return sorted(rows, key=lambda row: row[2])


def select_domain(
    domain: Domain,
    score_column: str,
    bucket_count: int,
    per_bucket_train: int,
    per_bucket_test: int,
    rng: random.Random,
) -> tuple[list[Sample], list[Sample]]:
    rows = read_rows(domain, score_column)
    buckets = [
        rows[(index * len(rows)) // bucket_count : ((index + 1) * len(rows)) // bucket_count]
        for index in range(bucket_count)
    ]

    train: list[Sample] = []
    test: list[Sample] = []
    needed = per_bucket_train + per_bucket_test
    for index, bucket in enumerate(buckets, start=1):
        if len(bucket) < needed:
            raise RuntimeError(
                f"{domain.name} bucket {index} contains {len(bucket)} usable images, "
                f"but {needed} are required."
            )

        picked = rng.sample(bucket, needed)
        rng.shuffle(picked)
        samples = [Sample(name, source, score, index) for name, source, score in picked]
        train.extend(samples[:per_bucket_train])
        test.extend(samples[per_bucket_train:])
        print(
            f"{domain.name} bucket {index}: available={len(bucket)}, "
            f"train={per_bucket_train}, test={per_bucket_test}, "
            f"score_range=[{bucket[0][2]:.6f}, {bucket[-1][2]:.6f}]"
        )

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def prepare_target(path: Path, overwrite: bool) -> None:
    if path.exists():
        entries = list(path.iterdir())
        if entries and not overwrite:
            raise FileExistsError(f"Target is not empty: {path}. Pass --overwrite to replace it.")
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                raise RuntimeError(f"Refusing to remove nested directory: {entry}")
            entry.unlink()
    path.mkdir(parents=True, exist_ok=True)


def materialize(
    samples: list[Sample],
    target: Path,
    prefix: str,
    mode: str,
) -> list[dict[str, str | int]]:
    manifest: list[dict[str, str | int]] = []
    for index, sample in enumerate(samples, start=1):
        destination_name = (
            f"{prefix}_{index:04d}_score{sample.score:.6f}_{Path(sample.filename).name}"
        )
        destination = target / destination_name
        if mode == "copy":
            shutil.copy2(sample.source, destination)
        else:
            os.symlink(sample.source.resolve(), destination)

        manifest.append(
            {
                "destination": destination_name,
                "source_filename": sample.filename,
                "score": f"{sample.score:.8f}",
                "bucket": sample.bucket,
            }
        )
    return manifest


def write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["destination", "source_filename", "score", "bucket"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create trainA/testA/trainB/testB from five luminance quantile buckets."
    )
    parser.add_argument("--day-csv", type=Path, required=True)
    parser.add_argument("--night-csv", type=Path, required=True)
    parser.add_argument("--day-dir", type=Path, required=True)
    parser.add_argument("--night-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--score-column", default="mean_luminance_normalized")
    parser.add_argument("--buckets", type=int, default=5)
    parser.add_argument("--per-bucket-train", type=int, default=1000)
    parser.add_argument("--per-bucket-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=6311)
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files or links in the four target folders.",
    )
    args = parser.parse_args()

    if min(args.buckets, args.per_bucket_train, args.per_bucket_test) < 1:
        raise ValueError("--buckets and per-bucket counts must all be >= 1")

    domains = [
        Domain("day", args.day_csv, args.day_dir, "trainA", "testA"),
        Domain("night", args.night_csv, args.night_dir, "trainB", "testB"),
    ]
    rng = random.Random(args.seed)

    # Select and validate both domains before writing any images.
    selected: dict[str, tuple[list[Sample], list[Sample]]] = {}
    for domain in domains:
        selected[domain.name] = select_domain(
            domain,
            args.score_column,
            args.buckets,
            args.per_bucket_train,
            args.per_bucket_test,
            rng,
        )

    folders = ("trainA", "testA", "trainB", "testB")
    for folder in folders:
        prepare_target(args.out_dir / folder, args.overwrite)

    manifest_dir = args.out_dir / "manifests"
    if manifest_dir.exists() and any(manifest_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Manifest directory is not empty: {manifest_dir}. Pass --overwrite to replace it."
        )
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        train, test = selected[domain.name]
        for samples, folder in (
            (train, domain.train_folder),
            (test, domain.test_folder),
        ):
            manifest = materialize(samples, args.out_dir / folder, domain.name, args.mode)
            write_manifest(manifest_dir / f"{folder}.csv", manifest)
            print(f"{folder}: wrote {len(samples)} images and a manifest")

    print(f"Dataset ready in {args.out_dir}")


if __name__ == "__main__":
    main()
