import os
from typing import Optional

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from src.models.seg_unet_resnet import Encoder9Res


def init_decoder_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm2d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def upsample_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm2d(out_c, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True),
    )


class GeneratorDecoder(nn.Module):
    def __init__(
        self,
        base: int = 64,
        out_channels: int = 3,
        use_skip: bool = True,
        n_res_blocks: int = 3,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.res_blocks = (
            nn.Sequential(*[ResBlock(base * 4) for _ in range(n_res_blocks)])
            if n_res_blocks > 0
            else nn.Identity()
        )
        self.up1 = upsample_block(base * 4, base * 2)
        self.up2 = upsample_block(base * 2, base)

        if use_skip:
            self.fuse1 = nn.Sequential(
                nn.Conv2d(base * 2 + base * 2, base * 2, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(base * 2, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
            )
            self.fuse2 = nn.Sequential(
                nn.Conv2d(base + base, base, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(base, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
            )
        else:
            self.fuse1 = nn.Identity()
            self.fuse2 = nn.Identity()

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.use_skip:
            if skips is None:
                raise ValueError("Skip connections required but none were provided.")
            e2, e1 = skips
        else:
            e2 = e1 = None

        bottleneck = self.res_blocks(bottleneck)
        x = self.up1(bottleneck)
        if self.use_skip:
            x = self.fuse1(torch.cat([x, e2], dim=1))
        else:
            x = self.fuse1(x)

        x = self.up2(x)
        if self.use_skip:
            x = self.fuse2(torch.cat([x, e1], dim=1))
        else:
            x = self.fuse2(x)

        return self.head(x)


class CycleGANGenerator(nn.Module):
    """
    CycleGAN generator that re-uses the segmentation encoder weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        n_res_blocks: int = 9,
        use_skip: bool = True,
        encoder_checkpoint: Optional[str] = None,
        freeze_encoder: bool = False,
        decoder_res_blocks: int = 3,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.encoder = Encoder9Res(in_c=in_channels, base=base_channels, n_res=n_res_blocks)
        self.decoder = GeneratorDecoder(
            base=base_channels,
            out_channels=out_channels,
            use_skip=use_skip,
            n_res_blocks=decoder_res_blocks,
        )
        self.decoder.apply(init_decoder_weights)

        if encoder_checkpoint:
            self.load_encoder_weights(encoder_checkpoint)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = True) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.encoder.load_state_dict(state, strict=strict)
        if missing or unexpected:
            raise RuntimeError(
                f"Mismatch when loading encoder weights. Missing: {missing}, Unexpected: {unexpected}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder(x)
        if not self.use_skip:
            skips = None
        return self.decoder(bottleneck, skips)


class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        max_channels: int = 512,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        def conv_layer(*args, **kwargs):
            layer = nn.Conv2d(*args, **kwargs)
            return spectral_norm(layer) if use_spectral_norm else layer

        layers = [
            nn.Sequential(
                conv_layer(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]

        in_c = base_channels
        for i in range(1, n_layers):
            out_c = min(base_channels * 2**i, max_channels)
            stride = 1 if i == n_layers - 1 else 2
            layers.append(
                nn.Sequential(
                    conv_layer(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False),
                    nn.InstanceNorm2d(out_c, affine=False, track_running_stats=False),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_c = out_c

        layers.append(conv_layer(in_c, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GANLoss(nn.Module):
    """
    Least Squares GAN loss used in CycleGAN.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)
