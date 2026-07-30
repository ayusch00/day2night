from __future__ import annotations

import csv
import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image

from scripts.prepare_dataset import Domain, materialize, select_domain, write_manifest
from scripts.score_images import normalize_luminance, score_image


class DataToolTests(unittest.TestCase):
    def test_luminance_scoring_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dark = root / "dark.png"
            bright = root / "bright.png"
            Image.new("RGB", (16, 16), color=(0, 0, 0)).save(dark)
            Image.new("RGB", (16, 16), color=(255, 255, 255)).save(bright)

            rows = [
                score_image((str(dark), dark.name, "day")),
                score_image((str(bright), bright.name, "day")),
            ]
            normalize_luminance(rows)

            self.assertEqual(float(rows[0]["mean_luminance_normalized"]), 0.0)
            self.assertEqual(float(rows[1]["mean_luminance_normalized"]), 1.0)
            self.assertEqual(float(rows[0]["dark_frac"]), 1.0)
            self.assertEqual(float(rows[1]["bright_frac"]), 1.0)

    def test_bucket_split_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            csv_path = root / "scores.csv"

            rows = []
            for index in range(12):
                filename = f"image_{index:02d}.png"
                Image.new("RGB", (8, 8), color=(index * 20,) * 3).save(image_dir / filename)
                rows.append(
                    {
                        "filename": filename,
                        "mean_luminance_normalized": f"{index / 11:.8f}",
                        "error": "",
                    }
                )
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["filename", "mean_luminance_normalized", "error"],
                )
                writer.writeheader()
                writer.writerows(rows)

            domain = Domain("day", csv_path, image_dir, "trainA", "testA")
            with redirect_stdout(io.StringIO()):
                first = select_domain(domain, "mean_luminance_normalized", 3, 2, 1, random.Random(7))
                second = select_domain(domain, "mean_luminance_normalized", 3, 2, 1, random.Random(7))
            train, test = first

            self.assertEqual([sample.filename for sample in train], [sample.filename for sample in second[0]])
            self.assertEqual([sample.filename for sample in test], [sample.filename for sample in second[1]])
            self.assertEqual(len(train), 6)
            self.assertEqual(len(test), 3)
            self.assertTrue({sample.filename for sample in train}.isdisjoint(sample.filename for sample in test))

            output = root / "output"
            output.mkdir()
            manifest = materialize(train, output, "day", mode="symlink")
            write_manifest(root / "manifest.csv", manifest)
            self.assertEqual(len(list(output.iterdir())), 6)
            with (root / "manifest.csv").open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 7)


if __name__ == "__main__":
    unittest.main()
