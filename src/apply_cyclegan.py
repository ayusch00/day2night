from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from torchvision import transforms

from src.models.cyclegan import CycleGANGenerator
from src.utils.common import load_cfg

SUPPORTED_EXTENSIONS: Sequence[str] = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")


def build_inference_transform(cfg: dict) -> transforms.Compose:
    ops: list = []
    tcfg = cfg.get("transforms", {})
    center_crop = tcfg.get("center_crop")
    if center_crop:
        ops.append(transforms.CenterCrop(center_crop))
    resize = tcfg.get("resize")
    if resize:
        ops.append(transforms.Resize((resize, resize), antialias=True))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)


def gather_image_paths(root: Path, extensions: Sequence[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    allowed = {ext.lower().lstrip(".") for ext in extensions}
    images = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower().lstrip(".") in allowed
    ]
    return images


def find_checkpoint(checkpoint: str | None, cfg: dict) -> Path:
    if checkpoint:
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt
    exp_root = Path(cfg["logging"]["out_dir"])
    pattern = exp_root.glob(f"{cfg['exp_name']}_*/epoch_*.pt")
    candidates = sorted(pattern)
    if not candidates:
        raise RuntimeError(
            f"No checkpoints found in {exp_root} for experiment prefix {cfg['exp_name']}_*"
        )
    return candidates[-1]


def build_generator(cfg: dict, device: torch.device) -> CycleGANGenerator:
    gen_cfg = cfg["model"]["generator"]
    return CycleGANGenerator(
        in_channels=gen_cfg.get("in_channels", 3),
        out_channels=gen_cfg.get("out_channels", 3),
        base_channels=gen_cfg.get("base_channels", 64),
        n_res_blocks=gen_cfg.get("n_res_blocks", 9),
        use_skip=gen_cfg.get("use_skip", True),
        encoder_checkpoint=gen_cfg.get("encoder_checkpoint"),
        freeze_encoder=gen_cfg.get("freeze_encoder", False),
        decoder_res_blocks=gen_cfg.get("decoder_res_blocks", 3),
    ).to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = tensor.mul(0.5).add(0.5).clamp(0.0, 1.0)
    return transforms.ToPILImage()(tensor)


def apply_generator(
    generator: CycleGANGenerator,
    image_paths: Iterable[Path],
    transform: transforms.Compose,
    output_dir: Path,
    input_root: Path,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.inference_mode():
        for count, image_path in enumerate(image_paths, start=1):
            img = Image.open(image_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            output_tensor = generator(input_tensor)
            result_img = tensor_to_pil(output_tensor)
            rel_path = image_path.relative_to(input_root)
            destination = output_dir / rel_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            result_img.save(destination)
            if count % 10 == 0:
                print(f"Converted {count}/{len(image_paths)} images", end="\r")
    print(f"\n{len(image_paths)} images saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a trained CycleGAN generator to convert between day/night domains."
    )
    parser.add_argument("--config", "-c", default="configs/cyclegan.yaml")
    parser.add_argument(
        "--checkpoint",
        "-k",
        help="Path to a CycleGAN checkpoint (defaults to latest run of the configured experiment).",
    )
    parser.add_argument(
        "--direction",
        "-d",
        choices=("day2night", "night2day"),
        default="day2night",
        help="Which mapping to use.",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        help="Directory with input images. Defaults to testA/testB inside the bundled dataset.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results/cyclegan_inference",
        help="Base directory for generated images.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="Image extensions to consider (overrides config).",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = find_checkpoint(args.checkpoint, cfg)
    transform = build_inference_transform(cfg)
    extensions = (
        [ext.lower().lstrip(".") for ext in args.extensions]
        if args.extensions
        else cfg["data"].get("extensions", SUPPORTED_EXTENSIONS)
    )

    default_inputs = {
        "day2night": Path("data/night2day/night_to_day/testA"),
        "night2day": Path("data/night2day/night_to_day/testB"),
    }
    input_dir = Path(args.input_dir) if args.input_dir else default_inputs[args.direction]
    image_paths = gather_image_paths(input_dir, extensions)
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir} with extensions {extensions}")

    output_dir = Path(args.output_dir) / args.direction
    generator = build_generator(cfg, device)
    state = torch.load(checkpoint, map_location="cpu")
    key = "G" if args.direction == "day2night" else "F"
    generator.load_state_dict(state[key])

    print(f"Using checkpoint {checkpoint} to convert {len(image_paths)} images from {input_dir}")
    apply_generator(generator, image_paths, transform, output_dir, input_dir, device)


if __name__ == "__main__":
    main()
