from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.models.cyclegan import CycleGANGenerator, PatchDiscriminator
from src.models.seg_unet_resnet import Encoder9Res


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_encoder_transfer_freezes_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "encoder.pth"
            encoder = Encoder9Res(in_c=3, base=8, n_res=1)
            torch.save(encoder.state_dict(), checkpoint)

            generator = CycleGANGenerator(
                base_channels=8,
                n_res_blocks=1,
                encoder_checkpoint=str(checkpoint),
                freeze_encoder=True,
                decoder_res_blocks=0,
            )
            self.assertTrue(all(not parameter.requires_grad for parameter in generator.encoder.parameters()))
            for name, expected in encoder.state_dict().items():
                self.assertTrue(torch.equal(generator.encoder.state_dict()[name], expected))

    def test_generator_and_discriminator_shapes(self) -> None:
        image = torch.randn(1, 3, 64, 64)
        generator = CycleGANGenerator(base_channels=8, n_res_blocks=1, decoder_res_blocks=0)
        generated = generator(image)
        discriminator = PatchDiscriminator(base_channels=8, n_layers=4, max_channels=64)
        prediction = discriminator(generated)

        self.assertEqual(generated.shape, image.shape)
        self.assertEqual(prediction.ndim, 4)
        self.assertEqual(prediction.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
