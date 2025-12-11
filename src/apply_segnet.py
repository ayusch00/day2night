from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from src.models.seg_unet_resnet import SegNet9ResUNet
from src.utils.common import load_cfg

SUPPORTED_EXTENSIONS: Sequence[str] = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")

# Cityscapes trainId palette (0-18) with readable names.
CITYSCAPES_CLASSES: list[tuple[str, tuple[int, int, int]]] = [
    ("road", (128, 64, 128)),
    ("sidewalk", (244, 35, 232)),
    ("building", (70, 70, 70)),
    ("wall", (102, 102, 156)),
    ("fence", (190, 153, 153)),
    ("pole", (153, 153, 153)),
    ("traffic light", (250, 170, 30)),
    ("traffic sign", (220, 220, 0)),
    ("vegetation", (107, 142, 35)),
    ("terrain", (152, 251, 152)),
    ("sky", (70, 130, 180)),
    ("person", (220, 20, 60)),
    ("rider", (255, 0, 0)),
    ("car", (0, 0, 142)),
    ("truck", (0, 0, 70)),
    ("bus", (0, 60, 100)),
    ("train", (0, 80, 100)),
    ("motorcycle", (0, 0, 230)),
    ("bicycle", (119, 11, 32)),
]


def build_inference_transform(cfg: dict | None) -> transforms.Compose:
    tf_cfg = cfg.get("transforms", {}) if cfg else {}
    resize = tf_cfg.get("resize")
    crop = tf_cfg.get("crop")
    ops: list = []
    if resize:
        if isinstance(resize, int):
            resize_arg = resize
        elif isinstance(resize, (list, tuple)) and len(resize) == 2:
            resize_arg = tuple(int(x) for x in resize)
        else:
            raise ValueError("transforms.resize must be an int or a tuple/list of two ints.")
        ops.append(transforms.Resize(resize_arg, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
    if crop:
        if not isinstance(crop, int):
            raise ValueError("transforms.crop must be a single int.")
        ops.append(transforms.CenterCrop((crop, crop)))
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
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower().lstrip(".") in allowed
    ]


def resolve_checkpoint(
    checkpoint: str | None,
    experiments_root: str | Path,
    filename: str = "segnet_full.pth",
    run_prefix: str = "seg_",
) -> Path:
    if checkpoint:
        candidate = Path(checkpoint)
        if candidate.is_dir():
            candidate = candidate / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate
    root = Path(experiments_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Experiments directory not found: {root}")
    pattern = root.glob(f"{run_prefix}*/{filename}")
    candidates = sorted(pattern, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No '{filename}' found under {root} with pattern {run_prefix}*")
    return candidates[-1]


def infer_model_channels(state_dict: dict) -> tuple[int, int]:
    num_classes = None
    base_channels = None
    for k, v in state_dict.items():
        if num_classes is None and k.endswith("decoder.head.1.weight"):
            num_classes = int(v.shape[0])
        if base_channels is None and k.endswith("encoder.c7s1_64.1.weight"):
            base_channels = int(v.shape[0])
        if num_classes is not None and base_channels is not None:
            break
    if num_classes is None:
        raise RuntimeError("Unable to infer number of classes from checkpoint.")
    if base_channels is None:
        base_channels = 64
    return num_classes, base_channels


def load_segnet(checkpoint: Path, device: torch.device, cfg: dict | None) -> SegNet9ResUNet:
    print(f"Loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    num_classes, base_channels = infer_model_channels(state)
    n_res = cfg.get("model", {}).get("n_resblocks", 9) if cfg else 9
    model = SegNet9ResUNet(num_classes=num_classes, base=base_channels, n_res=n_res).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_palette(num_classes: int) -> list[tuple[int, int, int]]:
    palette = list(color for _, color in CITYSCAPES_CLASSES)
    if num_classes > len(palette):
        rng = np.random.default_rng(0)
        extra = num_classes - len(palette)
        for _ in range(extra):
            palette.append(tuple(int(x) for x in rng.integers(low=0, high=255, size=3)))
    return palette[:num_classes]


def make_class_names(num_classes: int) -> list[str]:
    names = [name for name, _ in CITYSCAPES_CLASSES]
    if num_classes > len(names):
        names.extend([f"class_{i}" for i in range(len(names), num_classes)])
    return names[:num_classes]


def tensor_to_mask(
    model: SegNet9ResUNet,
    image: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
    out_size: tuple[int, int],
) -> np.ndarray:
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        preds = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return preds


def colorize_mask(mask: np.ndarray, palette: Sequence[tuple[int, int, int]]) -> Image.Image:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in enumerate(palette):
        color[mask == idx] = rgb
    return Image.fromarray(color)


def closest_16by9_size(width: int, height: int, target_ratio: float = 16 / 9) -> tuple[int, int]:
    ratio = width / height
    if abs(ratio - target_ratio) < 1e-3:
        return width, height
    target_w = max(1, int(round(height * target_ratio)))
    target_h = max(1, int(round(width / target_ratio)))
    delta_w = abs(target_w - width) / max(1, width)
    delta_h = abs(target_h - height) / max(1, height)
    if delta_w <= delta_h:
        return target_w, height
    return width, target_h


def render_legend(
    class_names: Sequence[str],
    palette: Sequence[tuple[int, int, int]],
    width: int,
    swatch: int = 18,
    padding: int = 8,
) -> Image.Image:
    font = ImageFont.load_default()
    entries = []
    draw_dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for idx, (name, color) in enumerate(zip(class_names, palette)):
        label = f"{idx}: {name}"
        bbox = draw_dummy.textbbox((0, 0), label, font=font)
        entries.append((label, color, bbox[2] - bbox[0], bbox[3] - bbox[1]))
    line_height = max(h for _, _, _, h in entries) + padding // 2
    x = padding
    y = padding
    max_width = width - padding
    lines: list[list[tuple[str, tuple[int, int, int]]]] = [[]]
    for label, color, text_w, _ in entries:
        needed = swatch + padding + text_w
        if x + needed > max_width and lines[-1]:
            x = padding
            y += line_height
            lines.append([])
        lines[-1].append((label, color))
        x += needed + padding
    legend_height = y + line_height + padding
    canvas = Image.new("RGB", (width, legend_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    x = padding
    y = padding
    for line in lines:
        for label, color in line:
            draw.rectangle([x, y, x + swatch, y + swatch], fill=color)
            draw.rectangle([x, y, x + swatch, y + swatch], outline=(255, 255, 255))
            draw.text((x + swatch + padding // 2, y), label, fill=(235, 235, 235), font=font)
            text_w = draw.textlength(label, font=font)
            x += swatch + padding // 2 + int(text_w) + padding
        x = padding
        y += line_height
    return canvas


def side_by_side(
    original: Image.Image,
    segmented: Image.Image,
    legend: Image.Image | None = None,
    pad: int = 12,
    bg_color: tuple[int, int, int] = (20, 20, 20),
) -> Image.Image:
    # Resize original to match segmented height for a cleaner comparison.
    target_h = segmented.height
    new_w = int(round(original.width * target_h / max(1, original.height)))
    original_resized = original.resize((new_w, target_h), resample=Image.BICUBIC)
    row_height = max(original_resized.height, segmented.height)
    row_width = original_resized.width + segmented.width + pad
    legend_height = legend.height + pad if legend else 0
    canvas = Image.new("RGB", (row_width, row_height + legend_height), color=bg_color)
    canvas.paste(original_resized, (0, 0))
    canvas.paste(segmented, (original_resized.width + pad, 0))
    if legend:
        legend_x = max(0, (row_width - legend.width) // 2)
        canvas.paste(legend, (legend_x, row_height + pad // 2))
    return canvas


def process_images(
    model: SegNet9ResUNet,
    image_paths: Iterable[Path],
    transform: transforms.Compose,
    palette: Sequence[tuple[int, int, int]],
    class_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
    stretch: bool = True,
) -> None:
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    legend = None
    pad = 12

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        mask_np = tensor_to_mask(model, image, transform, device, out_size=(image.height, image.width))
        color_mask = colorize_mask(mask_np, palette)
        if stretch:
            target_w, target_h = closest_16by9_size(color_mask.width, color_mask.height)
            color_mask = color_mask.resize((target_w, target_h), resample=Image.NEAREST)
        if legend is None:
            resized_w = int(round(image.width * color_mask.height / max(1, image.height)))
            row_width = resized_w + color_mask.width + pad
            legend = render_legend(class_names, palette, width=row_width)
        combined = side_by_side(image, color_mask, legend=legend, pad=pad)

        stem = img_path.stem
        viz_path = viz_dir / f"{stem}_viz.png"
        combined.save(viz_path)
        print(f"Saved {viz_path.relative_to(output_dir.parent)}")

    legend_path = output_dir / "legend.png"
    legend.save(legend_path)
    print(f"Legend saved to {legend_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SegNet inference on a handful of images.")
    parser.add_argument("--config", "-c", default="configs/seg.yaml", help="Config file (for transforms).")
    parser.add_argument("--checkpoint", "-k", help="Path to segnet_full.pth (defaults to latest experiments/seg_*/).")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory with input images.")
    parser.add_argument("--output-dir", "-o", default="results/segnet_notebook", help="Base output directory.")
    parser.add_argument("--num-samples", "-n", type=int, default=5, help="Number of random images to process.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda or cpu (defaults to auto).")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional subfolder name inside output-dir. Defaults to checkpoint run name or timestamp.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config) if Path(args.config).is_file() else {}
    transform = build_inference_transform(cfg)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments_root = cfg.get("logging", {}).get("out_dir", "experiments")
    checkpoint = resolve_checkpoint(args.checkpoint, experiments_root=experiments_root)
    model = load_segnet(checkpoint, device=device, cfg=cfg)

    input_dir = Path(args.input_dir)
    extensions = cfg.get("data", {}).get("extensions", SUPPORTED_EXTENSIONS)
    images = gather_image_paths(input_dir, extensions)
    if not images:
        raise RuntimeError(f"No images found in {input_dir} with extensions {extensions}")
    k = min(args.num_samples, len(images))
    random.Random(args.seed).shuffle(images)
    selected = images[:k]
    print(f"Processing {k} images from {input_dir} using {checkpoint.name}")

    palette = make_palette(model.decoder.head[1].weight.shape[0])
    class_names = make_class_names(len(palette))
    default_run = checkpoint.parent.name if checkpoint.parent else None
    run_id = args.run_name or default_run or time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    process_images(
        model=model,
        image_paths=selected,
        transform=transform,
        palette=palette,
        class_names=class_names,
        device=device,
        output_dir=output_dir,
        stretch=True,
    )
    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
