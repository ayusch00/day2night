from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.utils.common import load_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
SEG_CONFIG_PATH = REPO_ROOT / "configs" / "segmentation.yaml"
SEMGAN_CONFIG_PATH = REPO_ROOT / "configs" / "semgan.yaml"


class ConfigTests(unittest.TestCase):
    def test_final_segmentation_configuration(self) -> None:
        config = load_cfg(SEG_CONFIG_PATH)
        self.assertEqual(config["exp_name"], "seg_prop")
        self.assertEqual(config["data"]["dataset"], "cityscapes")
        self.assertEqual(config["data"]["sample_n_train"], 5000)
        self.assertEqual(config["transforms"]["resize"], 572)
        self.assertEqual(config["transforms"]["crop"], 512)
        self.assertEqual(config["train"]["epochs"], 300)

    def test_final_semgan_configuration(self) -> None:
        config = load_cfg(SEMGAN_CONFIG_PATH)
        generator = config["model"]["generator"]
        self.assertEqual(config["exp_name"], "semgan_prop5k")
        self.assertEqual(config["data"]["sample_limit"], 5000)
        self.assertEqual(generator["encoder_checkpoint"], "checkpoints/encoder_GE.pth")
        self.assertEqual(generator["encoder_transfer_directions"], ["day2night"])
        self.assertTrue(generator["freeze_encoder"])
        self.assertEqual(generator["decoder_res_blocks"], 0)
        self.assertEqual(config["semgan"]["apply_on"], ["A2B", "B2A"])
        self.assertEqual(config["semgan"]["lambda_sem"], 1.0)
        self.assertEqual(config["train"]["epochs"], 300)

    def test_flat_configs_reject_obsolete_section_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "flat.yaml"
            path.write_text(yaml.safe_dump({"exp_name": "legacy"}), encoding="utf-8")
            with self.assertRaises(KeyError):
                load_cfg(path, section="translation")


if __name__ == "__main__":
    unittest.main()
