from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.apply_cyclegan import build_inference_transform, find_checkpoint
from src.models.cyclegan import CycleGANGenerator, PatchDiscriminator
from src.utils.common import load_cfg


def _make_generator(cfg: dict, device: torch.device) -> CycleGANGenerator:
    gen_cfg = cfg["model"]["generator"]
    return CycleGANGenerator(
        in_channels=gen_cfg.get("in_channels", 3),
        out_channels=gen_cfg.get("out_channels", 3),
        base_channels=gen_cfg.get("base_channels", 64),
        n_res_blocks=gen_cfg.get("n_res_blocks", 9),
        use_skip=gen_cfg.get("use_skip", True),
        encoder_checkpoint=None,
        freeze_encoder=gen_cfg.get("freeze_encoder", False),
        decoder_res_blocks=gen_cfg.get("decoder_res_blocks", 3),
    ).to(device)


def _make_discriminator(cfg: dict, device: torch.device) -> PatchDiscriminator:
    disc_cfg = cfg["model"]["discriminator"]
    return PatchDiscriminator(
        in_channels=disc_cfg.get("in_channels", 3),
        base_channels=disc_cfg.get("base_channels", 64),
        n_layers=disc_cfg.get("n_layers", 3),
        max_channels=disc_cfg.get("max_channels", 512),
        use_spectral_norm=disc_cfg.get("use_spectral_norm", False),
    ).to(device)


def load_models(cfg: dict, checkpoint: Path, device: torch.device, direction: str):
    """
    Build CycleGAN components and load weights from a checkpoint.
    Returns the generator to inspect, plus both discriminators.
    """
    state = torch.load(checkpoint, map_location="cpu")
    G = _make_generator(cfg, device)
    F = _make_generator(cfg, device)
    D_day = _make_discriminator(cfg, device)
    D_night = _make_discriminator(cfg, device)

    G.load_state_dict(state["G"])
    F.load_state_dict(state["F"])
    D_day.load_state_dict(state["D_day"])
    D_night.load_state_dict(state["D_night"])

    for model in (G, F, D_day, D_night):
        model.eval()

    generator = G if direction == "day2night" else F
    return generator, D_day, D_night


def preprocess_image(image_path: Path, transform: transforms.Compose, auto_resize: bool = False) -> torch.Tensor:
    expected_size: tuple[int, int] | None = getattr(transform, "expected_size", None)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if expected_size and img.size != expected_size:
            if not auto_resize:
                raise ValueError(
                    f"Input image {image_path} has size {img.size}, expected {expected_size}. "
                    "Pass --auto-resize to resize automatically."
                )
            img = img.resize(expected_size, Image.BICUBIC)
        tensor = transform(img).unsqueeze(0)
    return tensor


@torch.inference_mode()
def evaluate_image(
    image_path: Path,
    generator: CycleGANGenerator,
    disc_day: PatchDiscriminator,
    disc_night: PatchDiscriminator,
    transform: transforms.Compose,
    device: torch.device,
    auto_resize: bool,
) -> dict[str, float]:
    inp = preprocess_image(image_path, transform, auto_resize).to(device)
    day_score = torch.sigmoid(disc_day(inp)).mean().item()
    night_score = torch.sigmoid(disc_night(inp)).mean().item()
    identity_l1 = torch.mean(torch.abs(generator(inp) - inp)).item()
    return {
        "day_score": day_score,
        "night_score": night_score,
        "identity_l1": identity_l1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check how the CycleGAN discriminators rate an image (day vs night) and how much the generator would change it."
    )
    parser.add_argument("image", type=Path, help="Path to a single image to inspect.")
    parser.add_argument("--config", "-c", default="configs/cyclegan.yaml", help="Config used to build the models.")
    parser.add_argument(
        "--checkpoint",
        "-k",
        help="Path to epoch_xxxx.pt. Defaults to the latest checkpoint of the configured experiment.",
    )
    parser.add_argument(
        "--direction",
        "-d",
        choices=("day2night", "night2day"),
        default="day2night",
        help="Which generator to probe (G=day2night or F=night2day).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. cuda or cpu. Default: auto (cuda if available).",
    )
    parser.add_argument(
        "--auto-resize",
        action="store_true",
        help="Resize the input to the configured size if it does not match exactly.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = find_checkpoint(args.checkpoint, cfg)

    transform = build_inference_transform(cfg)
    generator, disc_day, disc_night = load_models(cfg, Path(checkpoint), device, args.direction)
    results = evaluate_image(
        args.image,
        generator,
        disc_day,
        disc_night,
        transform,
        device,
        auto_resize=args.auto_resize,
    )

    predicted = "day" if results["day_score"] >= results["night_score"] else "night"
    domain_hint = "night" if args.direction == "day2night" else "day"
    print(f"Checkpoint: {checkpoint}")
    print(f"Image:      {args.image}")
    print(f"D_day   sigmoid mean:   {results['day_score']:.4f}")
    print(f"D_night sigmoid mean:   {results['night_score']:.4f}")
    print(f"Identity L1 (generator output vs input): {results['identity_l1']:.4f}")
    print(
        f"Discriminator perception: {predicted.upper()} "
        f"({'expected target' if predicted == domain_hint else 'looks like source'} for {args.direction})"
    )
    print(
        "Note: A very low identity L1 means the generator would leave the image almost unchanged, "
        f"which suggests it already looks like {domain_hint} to the chosen generator."
    )


if __name__ == "__main__":
    main()
