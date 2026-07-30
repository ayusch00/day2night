#!/usr/bin/env python3
"""Separate BDD100K images into day and night source folders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


DAY_LABELS = {"daytime"}
NIGHT_LABELS = {"night"}


def load_annotations(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"BDD100K annotation file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        return payload["images"]
    raise ValueError("Expected a JSON list or an object containing an 'images' list.")


def classified_images(images_dir: Path, annotations: list[dict]) -> dict[str, list[Path]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"BDD100K image directory not found: {images_dir}")

    domains: dict[str, list[Path]] = {"day": [], "night": []}
    missing = 0
    for entry in annotations:
        filename = entry.get("name") or entry.get("filename")
        time_of_day = (entry.get("attributes") or {}).get("timeofday")
        if not filename:
            continue
        if time_of_day in DAY_LABELS:
            domain = "day"
        elif time_of_day in NIGHT_LABELS:
            domain = "night"
        else:
            continue

        source = images_dir / filename
        if source.is_file():
            domains[domain].append(source)
        else:
            missing += 1

    for paths in domains.values():
        paths.sort(key=lambda path: path.name)
    if not domains["day"] or not domains["night"]:
        raise RuntimeError(
            "The annotations and image directory did not yield both daytime and night images."
        )
    if missing:
        print(f"Skipped {missing} annotated images that were not present in {images_dir}")
    return domains


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


def place_images(paths: list[Path], target: Path, mode: str) -> None:
    for source in paths:
        destination = target / source.name
        if mode == "copy":
            shutil.copy2(source, destination)
        else:
            os.symlink(source.resolve(), destination)


def write_manifest(path: Path, domains: dict[str, list[Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "timeofday"])
        writer.writeheader()
        for domain in ("day", "night"):
            for source in domains[domain]:
                writer.writerow({"filename": source.name, "timeofday": domain})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split BDD100K images using the official attributes.timeofday metadata."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="BDD100K image split, e.g. images/100k/train.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Matching bdd100k_labels_images_<split>.json.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/bdd100k_domains"))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic maximum number of images per domain.",
    )
    parser.add_argument("--mode", choices=("copy", "symlink"), default="symlink")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files or links in the day and night target folders.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    domains = classified_images(args.images.resolve(), load_annotations(args.labels))
    if args.limit is not None:
        domains = {domain: paths[: args.limit] for domain, paths in domains.items()}

    for domain in ("day", "night"):
        target = args.out_dir / domain
        prepare_target(target, args.overwrite)
        place_images(domains[domain], target, args.mode)
        print(f"{domain}: wrote {len(domains[domain])} {args.mode}s to {target}")

    write_manifest(args.out_dir / "manifest.csv", domains)


if __name__ == "__main__":
    main()
