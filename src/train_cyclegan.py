import glob
import importlib
import inspect
import json
import math
import os
import random
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from typing import Sequence

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as Fnn
import torch.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm.auto import tqdm

from src.models.cyclegan import CycleGANGenerator, PatchDiscriminator, GANLoss
from src.models.seg_unet_resnet import SegNet9ResUNet
from src.utils.common import load_cfg, set_seed, make_run_dirs, resolve_encoder_checkpoint


DEFAULT_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")


def build_transform(cfg: dict) -> transforms.Compose:
    ops: list = []
    resize = cfg.get("resize")
    if resize:
        if isinstance(resize, int):
            ops.append(transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True))
        elif isinstance(resize, (list, tuple)) and len(resize) == 2 and all(isinstance(x, int) for x in resize):
            ops.append(transforms.Resize(tuple(resize), interpolation=InterpolationMode.BICUBIC, antialias=True))
        else:
            raise ValueError("transforms.resize must be an int (shorter side) or a tuple/list of two ints (h, w).")
    crop = cfg.get("random_crop")
    if crop:
        if not isinstance(crop, int):
            raise ValueError("transforms.random_crop must be a single int for square crops.")
        ops.append(transforms.RandomCrop((crop, crop)))
    if cfg.get("random_flip", True):
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: transforms.Compose,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        sample_limit: int | None = None,
    ):
        self.root = root
        self.transform = transform
        paths: list[str] = []
        for ext in extensions:
            pattern = os.path.join(root, f"**/*.{ext}")
            paths.extend(glob.glob(pattern, recursive=True))
        paths = sorted(set(paths))
        if not paths:
            raise RuntimeError(f"No images found in {root}.")
        if sample_limit:
            if len(paths) > sample_limit:
                paths = random.sample(paths, sample_limit)
            else:
                paths = paths[:sample_limit]
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img)


def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


@dataclass
class LossMeters:
    g_total: float = 0.0
    g_adv: float = 0.0
    g_cycle: float = 0.0
    g_id: float = 0.0
    g_sky: float = 0.0
    g_sem: float = 0.0
    d_total: float = 0.0

    def update(self, losses: dict[str, float], batch_size: int):
        for k, v in losses.items():
            setattr(self, k, getattr(self, k) + v * batch_size)

    def average(self, samples: int) -> dict[str, float]:
        keys = ["g_total", "g_adv", "g_cycle", "g_id", "g_sky", "g_sem", "d_total"]
        return {k: getattr(self, k) / samples for k in keys}


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_distributed_if_needed() -> tuple[bool, int]:
    distributed = distributed_world_size() > 1
    local_rank = 0
    if distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    return distributed, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def broadcast_dirs(exp_dir: str | None, results_dir: str | None) -> tuple[str, str]:
    payload = [exp_dir, results_dir]
    dist.broadcast_object_list(payload, src=0)
    return payload[0], payload[1]


def sync_meter_totals(meters: LossMeters, samples: int, device: torch.device, distributed: bool) -> tuple[dict[str, float], int]:
    keys = ["g_total", "g_adv", "g_cycle", "g_id", "g_sky", "g_sem", "d_total"]
    totals = torch.tensor([getattr(meters, k) for k in keys], device=device)
    sample_tensor = torch.tensor([samples], device=device)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    total_samples = int(sample_tensor.item())
    avg = {k: totals[i].item() / max(total_samples, 1) for i, k in enumerate(keys)}
    return avg, total_samples


def set_linear_lr(optimizer, base_lr, epoch, decay_start, total_epochs):
    if total_epochs <= decay_start or epoch <= decay_start:
        factor = 1.0
    else:
        decay_epochs = max(1, total_epochs - decay_start)
        factor = max(0.0, 1.0 - (epoch - decay_start) / decay_epochs)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    if img.numel() == 0:
        return torch.zeros((), device=img.device, dtype=img.dtype)
    loss_h = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    loss_w = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w


def gaussian_blur_tensor(img: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
    if sigma <= 0.0 or ksize <= 1:
        return img
    radius = ksize // 2
    coords = torch.arange(ksize, device=img.device, dtype=img.dtype) - radius
    kernel1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel1d = kernel1d / kernel1d.sum().clamp(min=1e-12)
    kernel2d = torch.einsum("i,j->ij", kernel1d, kernel1d)
    kernel2d = kernel2d.expand(img.shape[1], 1, ksize, ksize)
    return Fnn.conv2d(img, kernel2d, padding=radius, groups=img.shape[1])


def parse_hw_size(value: object, field_name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(x, int) and x > 0 for x in value)
    ):
        return int(value[0]), int(value[1])
    raise ValueError(f"{field_name} must be null or [H, W] with positive ints.")


def resolve_path_from_cfg(path_spec: str, cfg_path: str) -> str:
    candidate = os.path.expanduser(str(path_spec))
    if os.path.isabs(candidate):
        return candidate
    cwd_path = os.path.abspath(candidate)
    if os.path.exists(cwd_path):
        return cwd_path
    cfg_dir_path = os.path.abspath(os.path.join(os.path.dirname(cfg_path), candidate))
    if os.path.exists(cfg_dir_path):
        return cfg_dir_path
    # Return cwd-based absolute path for clear downstream error reporting.
    return cwd_path


def normalize_segmentor_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Some UDA checkpoints store multiple branches; prefer the actual train model.
    if any(k.startswith("model.") for k in state_dict):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}

    for prefix in ("module.", "model."):
        while state_dict and all(k.startswith(prefix) for k in state_dict):
            state_dict = {k.replace(prefix, "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("DTBS checkpoint must be a state_dict or dict containing one.")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state_dict = state["model"]
    else:
        state_dict = state
    state_dict = normalize_segmentor_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"DTBS checkpoint is missing model keys (first keys): {missing[:5]}")
    if unexpected:
        print(f"[SemGAN] Loaded DTBS checkpoint with unexpected keys (first keys): {unexpected[:5]}")


def build_dtbs_segmenter(
    seg_config_json: str,
    seg_checkpoint: str,
    num_classes: int,
) -> nn.Module:
    if not os.path.isfile(seg_config_json):
        raise FileNotFoundError(f"semgan.seg_config_json not found: {seg_config_json}")
    if not os.path.isfile(seg_checkpoint):
        raise FileNotFoundError(f"semgan.seg_checkpoint not found: {seg_checkpoint}")

    with open(seg_config_json, encoding="utf-8") as fh:
        cfg_json = json.load(fh)
    model_cfg = cfg_json.get("model")
    if not isinstance(model_cfg, dict):
        raise RuntimeError(f"DTBS config JSON must contain a top-level 'model' dict: {seg_config_json}")

    attempt_errors: list[str] = []

    # Prefer local DTBS vendored stack if present.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    local_dtbs_root = os.path.join(repo_root, "third_party", "DTBS")
    if os.path.isdir(local_dtbs_root) and local_dtbs_root not in sys.path:
        sys.path.insert(0, local_dtbs_root)

    # Preferred path: mmseg API if available.
    try:
        try:
            from mmseg.apis import init_model as mmseg_init_model  # type: ignore
        except Exception:
            from mmseg.apis import init_segmentor as mmseg_init_model  # type: ignore

        init_kwargs = {"device": "cpu"}
        if "revise_checkpoint" in inspect.signature(mmseg_init_model).parameters:
            init_kwargs["revise_checkpoint"] = [(r"^model\.", ""), (r"^module\.", "")]
        model = mmseg_init_model(seg_config_json, None, **init_kwargs)
        load_checkpoint_into_model(model, seg_checkpoint)
        return model
    except Exception as exc:  # noqa: BLE001
        attempt_errors.append(f"mmseg.apis init_model/init_segmentor failed: {exc}")

    # Fallback: build mmseg model manually and load checkpoint.
    manual_model_cfg = deepcopy(model_cfg)
    # DTBS/mmseg v0.x expects segmentor-level pretrained to be None at inference build time.
    manual_model_cfg["pretrained"] = None
    decode_head = manual_model_cfg.get("decode_head")
    if isinstance(decode_head, dict):
        decode_head["num_classes"] = num_classes
    elif isinstance(decode_head, list):
        for head in decode_head:
            if isinstance(head, dict):
                head["num_classes"] = num_classes

    try:
        from mmseg.models import build_segmentor  # type: ignore

        model = build_segmentor(manual_model_cfg)
        try:
            from mmengine.runner import load_checkpoint  # type: ignore

            load_checkpoint(model, seg_checkpoint, map_location="cpu")
            return model
        except Exception as mmengine_exc:  # noqa: BLE001
            attempt_errors.append(f"mmengine.runner.load_checkpoint failed: {mmengine_exc}")
        try:
            from mmcv.runner import load_checkpoint  # type: ignore

            load_checkpoint(model, seg_checkpoint, map_location="cpu")
            return model
        except Exception as mmcv_exc:  # noqa: BLE001
            attempt_errors.append(f"mmcv.runner.load_checkpoint failed: {mmcv_exc}")
        load_checkpoint_into_model(model, seg_checkpoint)
        return model
    except Exception as exc:  # noqa: BLE001
        attempt_errors.append(f"manual mmseg build/load failed: {exc}")

    # Fallback: potential local DTBS repo module with build_segmentor().
    local_builder_candidates = [
        "dtbs.models.builder",
        "dtbs.modeling.builder",
        "dtbs.segmentation.builder",
    ]
    for module_name in local_builder_candidates:
        try:
            mod = importlib.import_module(module_name)
            build_segmentor = getattr(mod, "build_segmentor")
            model = build_segmentor(manual_model_cfg)
            load_checkpoint_into_model(model, seg_checkpoint)
            return model
        except Exception as exc:  # noqa: BLE001
            attempt_errors.append(f"{module_name}.build_segmentor failed: {exc}")

    details = "\n".join(f"  - {msg}" for msg in attempt_errors)
    raise RuntimeError(
        "SemGAN DTBS model could not be constructed.\n"
        f"Config JSON: {seg_config_json}\n"
        f"Checkpoint: {seg_checkpoint}\n"
        "Expected DTBS modules were not found or failed to initialize.\n"
        "Expected options:\n"
        "  1) MMSeg stack (`mmseg` + `mmengine` or legacy `mmcv`) available in Python env.\n"
        "  2) Local DTBS code under a subfolder (e.g. `dtbs/...`) exposing `build_segmentor`.\n"
        "Build attempts:\n"
        f"{details}"
    )


def prepare_semgan_seg_input(
    image: torch.Tensor,
    seg_input_norm: str,
    seg_inference_size: tuple[int, int] | None,
) -> torch.Tensor:
    if seg_input_norm not in {"imagenet", "none", "minus1_1"}:
        raise ValueError("semgan.seg_input_norm must be one of: imagenet, none, minus1_1.")

    if seg_input_norm == "minus1_1":
        seg_input = image
    else:
        seg_input = image.mul(0.5).add(0.5).clamp(0.0, 1.0)
        if seg_input_norm == "imagenet":
            mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            seg_input = (seg_input - mean) / std

    if seg_inference_size is not None:
        seg_input = Fnn.interpolate(
            seg_input,
            size=seg_inference_size,
            mode="bilinear",
            align_corners=False,
        )
    return seg_input


def run_segmenter_for_semgan(
    seg_model: nn.Module,
    image: torch.Tensor,
    seg_input_norm: str,
    seg_inference_size: tuple[int, int] | None,
) -> torch.Tensor:
    seg_input = prepare_semgan_seg_input(image, seg_input_norm, seg_inference_size)

    # DTBS/mmseg segmentors are usually called with img+img_metas; encode_decode
    # returns per-class logits directly, which matches the SemGAN loss usage here.
    if hasattr(seg_model, "encode_decode"):
        h, w = int(seg_input.shape[-2]), int(seg_input.shape[-1])
        img_metas = [
            {
                "ori_shape": (h, w, 3),
                "img_shape": (h, w, 3),
                "pad_shape": (h, w, 3),
                "scale_factor": 1.0,
                "flip": False,
                "flip_direction": "horizontal",
            }
            for _ in range(seg_input.shape[0])
        ]
        return seg_model.encode_decode(seg_input, img_metas)  # type: ignore[attr-defined]

    return seg_model(seg_input)


def masked_channel_mean(value: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    mask = mask_hw.unsqueeze(1).to(dtype=value.dtype)
    denom = (mask.sum() * value.shape[1]).clamp_min(1.0)
    return (value * mask).sum() / denom


def semantic_consistency_loss(
    logits_real: torch.Tensor,
    logits_fake: torch.Tensor,
    sem_loss_type: str,
    ignore_index: int,
) -> torch.Tensor:
    sem_loss_type = sem_loss_type.lower()
    if sem_loss_type not in {"l1_prob", "kl_prob", "ce_pseudo"}:
        raise ValueError("semgan.sem_loss_type must be one of: l1_prob, kl_prob, ce_pseudo.")

    pseudo_labels = logits_real.argmax(dim=1)
    valid_mask = pseudo_labels != ignore_index
    if not torch.any(valid_mask):
        return logits_fake.new_zeros(())

    if sem_loss_type == "ce_pseudo":
        ce_map = Fnn.cross_entropy(logits_fake, pseudo_labels, reduction="none")
        return ce_map[valid_mask].mean()

    probs_real = torch.softmax(logits_real, dim=1)
    probs_fake = torch.softmax(logits_fake, dim=1)

    if sem_loss_type == "l1_prob":
        return masked_channel_mean((probs_fake - probs_real).abs(), valid_mask)

    eps = 1e-8
    kl_map = probs_real * ((probs_real + eps).log() - (probs_fake + eps).log())
    return masked_channel_mean(kl_map, valid_mask)


class ImagePool:
    """Stores previously generated images to stabilize discriminator training."""

    def __init__(self, size: int):
        self.size = max(0, size)
        self.buffer: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out: list[torch.Tensor] = []
        for img in images:
            img = img.unsqueeze(0).detach()
            if len(self.buffer) < self.size:
                self.buffer.append(img.clone())
                out.append(img)
            else:
                if random.random() < 0.5:
                    idx = random.randrange(self.size)
                    cached = self.buffer[idx].clone()
                    self.buffer[idx] = img.clone()
                    out.append(cached)
                else:
                    out.append(img)
        return torch.cat(out, dim=0)


def train(cfg_path: str = "configs/cyclegan.yaml", resume: str | None = None):
    cfg_path = os.path.abspath(cfg_path)
    cfg = load_cfg(cfg_path)
    # SemGAN usage:
    #   python -m src.train_cyclegan --config configs/semgan.yaml
    # Requires semgan.use_semgan=true plus semgan.seg_config_json and semgan.seg_checkpoint.
    resume_ckpt = resume or cfg.get("train", {}).get("resume_checkpoint")
    distributed, local_rank = init_distributed_if_needed()
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0
    seed = cfg["train"].get("seed", 42) + rank
    set_seed(seed)

    if distributed and not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA devices.")
    device = torch.device("cuda", local_rank) if distributed else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = device.type
    use_amp = cfg["train"].get("amp", True) and amp_device == "cuda"

    img_t = build_transform(cfg["transforms"])
    data_cfg = cfg["data"]

    day_ds = ImageFolderDataset(
        data_cfg["day_dir"],
        img_t,
        extensions=data_cfg.get("extensions", DEFAULT_EXTENSIONS),
        sample_limit=data_cfg.get("sample_limit"),
    )
    night_ds = ImageFolderDataset(
        data_cfg["night_dir"],
        img_t,
        extensions=data_cfg.get("extensions", DEFAULT_EXTENSIONS),
        sample_limit=data_cfg.get("sample_limit"),
    )

    if distributed:
        day_sampler = DistributedSampler(day_ds, shuffle=True, drop_last=True)
        night_sampler = DistributedSampler(night_ds, shuffle=True, drop_last=True)
    else:
        day_sampler = night_sampler = None

    dl_kwargs = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("workers", 4),
        shuffle=not distributed,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=cfg["train"].get("workers", 4) > 0,
    )
    day_loader = DataLoader(day_ds, sampler=day_sampler, **dl_kwargs)
    night_loader = DataLoader(night_ds, sampler=night_sampler, **dl_kwargs)

    gen_cfg = cfg["model"]["generator"]
    disc_cfg = cfg["model"]["discriminator"]

    freeze_encoder = gen_cfg.get("freeze_encoder", False)
    encoder_ckpt = (
        resolve_encoder_checkpoint(
            gen_cfg.get("encoder_checkpoint"),
            experiments_root=cfg["logging"].get("out_dir", "experiments"),
            default_run_prefix=gen_cfg.get("encoder_run_prefix", "seg"),
            filename=gen_cfg.get("encoder_filename", "encoder_GE.pth"),
        )
        if freeze_encoder
        else None
    )
    decoder_res_blocks = gen_cfg.get("decoder_res_blocks", 3)

    # G uses the segmentation encoder only when freeze_encoder is true; otherwise it starts random.
    g_encoder_ckpt = encoder_ckpt
    g_freeze_encoder = freeze_encoder
    f_encoder_ckpt = None
    f_freeze_encoder = False
    if is_main:
        print(f"[CycleGAN] G (day->night) encoder: {g_encoder_ckpt or 'None'} | freeze={g_freeze_encoder}")
        print(f"[CycleGAN] F (night->day) encoder: {f_encoder_ckpt or 'None'} | freeze={f_freeze_encoder}")

    G = CycleGANGenerator(
        in_channels=gen_cfg.get("in_channels", 3),
        out_channels=gen_cfg.get("out_channels", 3),
        base_channels=gen_cfg.get("base_channels", 64),
        n_res_blocks=gen_cfg.get("n_res_blocks", 9),
        use_skip=gen_cfg.get("use_skip", True),
        encoder_checkpoint=g_encoder_ckpt,
        freeze_encoder=g_freeze_encoder,
        decoder_res_blocks=decoder_res_blocks,
    ).to(device)

    F = CycleGANGenerator(
        in_channels=gen_cfg.get("in_channels", 3),
        out_channels=gen_cfg.get("out_channels", 3),
        base_channels=gen_cfg.get("base_channels", 64),
        n_res_blocks=gen_cfg.get("n_res_blocks", 9),
        use_skip=gen_cfg.get("use_skip", True),
        encoder_checkpoint=f_encoder_ckpt,
        freeze_encoder=f_freeze_encoder,
        decoder_res_blocks=decoder_res_blocks,
    ).to(device)

    D_day = PatchDiscriminator(
        in_channels=disc_cfg.get("in_channels", 3),
        base_channels=disc_cfg.get("base_channels", 64),
        n_layers=disc_cfg.get("n_layers", 3),
        max_channels=disc_cfg.get("max_channels", 512),
        use_spectral_norm=disc_cfg.get("use_spectral_norm", False),
    ).to(device)

    D_night = PatchDiscriminator(
        in_channels=disc_cfg.get("in_channels", 3),
        base_channels=disc_cfg.get("base_channels", 64),
        n_layers=disc_cfg.get("n_layers", 3),
        max_channels=disc_cfg.get("max_channels", 512),
        use_spectral_norm=disc_cfg.get("use_spectral_norm", False),
    ).to(device)

    sky_cfg = cfg.get("sky_loss", {})
    use_sky_loss = bool(sky_cfg.get("enabled", False))
    sky_class_ids: list[int] = sky_cfg.get("class_ids", []) or []
    sky_class_ids = [int(c) for c in sky_class_ids]
    sky_grad_weight = float(sky_cfg.get("grad_weight", sky_cfg.get("weight", sky_cfg.get("lambda", sky_cfg.get("lambda_sky", 0.0)))))
    sky_peak_weight = float(sky_cfg.get("peak_weight", sky_cfg.get("light_weight", sky_cfg.get("lambda_light", 0.0))))
    sky_peak_sigma = float(sky_cfg.get("peak_sigma", 3.0))
    sky_peak_delta = float(sky_cfg.get("peak_delta", sky_cfg.get("light_delta", 0.0)))
    sky_tv_weight = float(sky_cfg.get("tv_weight", 0.0))
    sky_input_size = sky_cfg.get("seg_input_size")
    if sky_input_size is not None:
        sky_input_size = tuple(int(x) for x in sky_input_size)
    sky_seg_model = None

    sem_cfg = cfg.get("semgan", {})
    use_semgan = bool(sem_cfg.get("use_semgan", False))
    sem_lambda = float(sem_cfg.get("lambda_sem", 0.1))
    sem_loss_type = str(sem_cfg.get("sem_loss_type", "l1_prob")).lower()
    valid_sem_loss_types = {"l1_prob", "kl_prob", "ce_pseudo"}
    if use_semgan and sem_loss_type not in valid_sem_loss_types:
        raise ValueError(
            f"semgan.sem_loss_type must be one of {sorted(valid_sem_loss_types)}, got '{sem_loss_type}'."
        )
    sem_apply_raw = sem_cfg.get("apply_on", ["A2B", "B2A"])
    if isinstance(sem_apply_raw, str):
        sem_apply_raw = [sem_apply_raw]
    sem_apply_on = {str(direction).upper() for direction in sem_apply_raw}
    valid_sem_apply = {"A2B", "B2A"}
    invalid_apply = sem_apply_on - valid_sem_apply
    if invalid_apply:
        raise ValueError(
            f"semgan.apply_on supports only {sorted(valid_sem_apply)}, got invalid entries: {sorted(invalid_apply)}"
        )
    if use_semgan and not sem_apply_on:
        raise ValueError("semgan.apply_on must contain at least one direction when semgan.use_semgan=true.")
    sem_num_classes = int(sem_cfg.get("num_classes", 19) or 19)
    sem_ignore_index = int(sem_cfg.get("ignore_index", 255))
    sem_seg_input_norm = str(sem_cfg.get("seg_input_norm", "imagenet")).lower()
    sem_seg_inference_size = parse_hw_size(sem_cfg.get("seg_inference_size"), "semgan.seg_inference_size")
    sem_cache_seg_logits = bool(sem_cfg.get("cache_seg_logits", False))
    sem_seg_model = None

    if distributed:
        ddp_kwargs = dict(device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
        G = DDP(G, **ddp_kwargs)
        F = DDP(F, **ddp_kwargs)
        D_day = DDP(D_day, **ddp_kwargs)
        D_night = DDP(D_night, **ddp_kwargs)
    elif torch.cuda.device_count() > 1:
        G = nn.DataParallel(G)
        F = nn.DataParallel(F)
        D_day = nn.DataParallel(D_day)
        D_night = nn.DataParallel(D_night)

    if use_sky_loss:
        seg_ckpt_spec = sky_cfg.get("seg_checkpoint") or "latest"
        seg_ckpt = None
        num_classes = int(sky_cfg.get("num_classes", 19) or 19)
        sky_seg_filename = sky_cfg.get("seg_filename", "segnet_full.pth") or "segnet_full.pth"
        experiments_root = cfg["logging"].get("out_dir", "experiments")
        sky_run_prefix = sky_cfg.get("seg_run_prefix", "seg") or "seg"
        if not sky_class_ids:
            if is_main:
                print("[SkyLoss] Disabled: sky_loss.class_ids is empty.")
            use_sky_loss = False
        else:
            try:
                seg_ckpt = resolve_encoder_checkpoint(
                    seg_ckpt_spec,
                    experiments_root=experiments_root,
                    default_run_prefix=sky_run_prefix,
                    filename=sky_seg_filename,
                )
            except FileNotFoundError:
                if is_main:
                    print(
                        f"[SkyLoss] Disabled: no seg checkpoint found for spec='{seg_ckpt_spec}' "
                        f"(pattern {sky_run_prefix}_*/{sky_seg_filename})."
                    )
                use_sky_loss = False
            if use_sky_loss:
                if is_main:
                    print(f"[SkyLoss] Using seg checkpoint: {seg_ckpt}")
                sky_seg_model = SegNet9ResUNet(num_classes=num_classes).to(device)
                try:
                    state = torch.load(seg_ckpt, map_location="cpu")
                    if not isinstance(state, dict):
                        raise RuntimeError("Checkpoint must be a state_dict or dict with 'state_dict'.")
                    state_dict = state["state_dict"] if "state_dict" in state else state
                    if state_dict and all(k.startswith("module.") for k in state_dict):
                        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                    missing, unexpected = sky_seg_model.load_state_dict(state_dict, strict=False)
                    if missing:
                        raise RuntimeError(
                            f"Missing keys when loading seg checkpoint (need full encoder+decoder): {missing[:5]}"
                        )
                    if unexpected and is_main:
                        print(f"[SkyLoss] Loaded seg checkpoint with unexpected={unexpected}")
                except Exception as exc:  # noqa: BLE001
                    if is_main:
                        print(f"[SkyLoss] Disabled: failed to load seg checkpoint ({exc}).")
                    sky_seg_model = None
                    use_sky_loss = False
                else:
                    sky_seg_model.eval()
                    for p in sky_seg_model.parameters():
                        p.requires_grad = False
                    if is_main:
                        print(
                            f"[SkyLoss] Enabled: classes={sky_class_ids}, "
                            f"grad_weight={sky_grad_weight}, peak_weight={sky_peak_weight}, "
                            f"peak_sigma={sky_peak_sigma}, peak_delta={sky_peak_delta}, "
                            f"tv_weight={sky_tv_weight}, ckpt={seg_ckpt}"
                        )

    if use_semgan:
        sem_seg_config = sem_cfg.get("seg_config_json")
        sem_seg_ckpt = sem_cfg.get("seg_checkpoint")
        if not sem_seg_config:
            raise ValueError("semgan.use_semgan=true requires semgan.seg_config_json.")
        if not sem_seg_ckpt:
            raise ValueError("semgan.use_semgan=true requires semgan.seg_checkpoint.")
        sem_seg_config = resolve_path_from_cfg(str(sem_seg_config), cfg_path)
        sem_seg_ckpt = resolve_path_from_cfg(str(sem_seg_ckpt), cfg_path)
        sem_seg_model = build_dtbs_segmenter(
            seg_config_json=sem_seg_config,
            seg_checkpoint=sem_seg_ckpt,
            num_classes=sem_num_classes,
        ).to(device)
        sem_seg_model.eval()
        for param in sem_seg_model.parameters():
            param.requires_grad = False
        if is_main:
            print(
                f"[SemGAN] Enabled with loss={sem_loss_type}, lambda_sem={sem_lambda}, apply_on={sorted(sem_apply_on)}, "
                f"num_classes={sem_num_classes}, ignore_index={sem_ignore_index}, "
                f"seg_input_norm={sem_seg_input_norm}, seg_inference_size={sem_seg_inference_size}, "
                f"cache_seg_logits={sem_cache_seg_logits}"
            )
            print(f"[SemGAN] DTBS config={sem_seg_config}, checkpoint={sem_seg_ckpt}")

    gan_loss = GANLoss().to(device)
    l1_loss = nn.L1Loss()

    optim_cfg = cfg["optim"]
    gen_lr = optim_cfg["generator"]["lr"]
    disc_lr = optim_cfg["discriminator"]["lr"]
    gen_params = [p for p in chain(G.parameters(), F.parameters()) if p.requires_grad]
    disc_params = [p for p in chain(D_day.parameters(), D_night.parameters()) if p.requires_grad]
    opt_G = torch.optim.Adam(
        gen_params,
        lr=gen_lr,
        betas=tuple(optim_cfg["generator"].get("betas", [0.5, 0.999])),
    )
    opt_D = torch.optim.Adam(
        disc_params,
        lr=disc_lr,
        betas=tuple(optim_cfg["discriminator"].get("betas", [0.5, 0.999])),
    )

    lambda_cycle = cfg["loss"].get("lambda_cycle", 10.0)
    lambda_id = cfg["loss"].get("lambda_identity", 5.0)

    epochs = cfg["train"]["epochs"]
    decay_start = cfg["train"].get("lr_decay_start", epochs // 2)
    steps_per_epoch = max(len(day_loader), len(night_loader))
    save_every = int(cfg["logging"].get("save_every", 40))
    if save_every <= 0:
        raise ValueError("logging.save_every must be >= 1.")
    latest_ckpt_name = cfg["logging"].get("latest_checkpoint_name", "latest.pt")
    best_ckpt_name = cfg["logging"].get("best_checkpoint_name", "best.pt")
    keep_epoch_checkpoints = bool(cfg["logging"].get("keep_epoch_checkpoints", False))
    best_metric_name = str(cfg["logging"].get("best_metric", "g_total"))
    best_metric_mode = str(cfg["logging"].get("best_metric_mode", "min")).lower()
    valid_metric_names = {"g_total", "g_adv", "g_cycle", "g_id", "g_sky", "g_sem", "d_total"}
    if best_metric_name not in valid_metric_names:
        raise ValueError(
            f"logging.best_metric must be one of {sorted(valid_metric_names)}, got '{best_metric_name}'."
        )
    if best_metric_mode not in {"min", "max"}:
        raise ValueError("logging.best_metric_mode must be 'min' or 'max'.")
    log_every = cfg["logging"].get("log_interval", 50)

    config_filename = os.path.basename(cfg_path)
    if resume_ckpt:
        exp_dir = os.path.abspath(os.path.join(resume_ckpt, os.pardir))
        results_dir = cfg["logging"].get("results_dir", "results")
        if is_main:
            print(f"[Resume] Using existing run directory: {exp_dir}")
        if distributed:
            exp_dir, results_dir = broadcast_dirs(exp_dir, results_dir)
    else:
        if distributed:
            if is_main:
                exp_dir, results_dir = make_run_dirs(cfg)
                shutil.copy(cfg_path, os.path.join(exp_dir, config_filename))
            else:
                exp_dir = results_dir = None
            exp_dir, results_dir = broadcast_dirs(exp_dir, results_dir)
        else:
            exp_dir, results_dir = make_run_dirs(cfg)
            shutil.copy(cfg_path, os.path.join(exp_dir, config_filename))
    if is_main:
        # Ensure run dirs exist before any logging occurs.
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    loss_log_path = os.path.join(exp_dir, "loss_log.txt")
    if is_main and not os.path.exists(loss_log_path):
        with open(loss_log_path, "w") as log_f:
            log_f.write("")

    scaler_G = amp.GradScaler(amp_device, enabled=use_amp)
    scaler_D = amp.GradScaler(amp_device, enabled=use_amp)
    pool_size = cfg["train"].get("image_pool_size", 50)
    fake_day_pool = ImagePool(pool_size)
    fake_night_pool = ImagePool(pool_size)

    if is_main:
        for domain, dataset in (("day", day_ds), ("night", night_ds)):
            filelist_path = os.path.join(exp_dir, f"{domain}_files.txt")
            with open(filelist_path, "w") as fh:
                for path in dataset.paths:
                    fh.write(f"{path}\n")
            print(f"Saved list of {len(dataset.paths)} {domain} images to {filelist_path}")

    best_metric_value: float | None = None
    best_metric_epoch = 0

    def metric_improved(current: float, best: float | None) -> bool:
        if best is None:
            return True
        if best_metric_mode == "min":
            return current < best
        return current > best

    start_epoch = 1
    if resume_ckpt:
        state = torch.load(resume_ckpt, map_location="cpu")
        target = lambda m: m.module if isinstance(m, (nn.DataParallel, DDP)) else m
        target(G).load_state_dict(state["G"])
        target(F).load_state_dict(state["F"])
        target(D_day).load_state_dict(state["D_day"])
        target(D_night).load_state_dict(state["D_night"])
        opt_G.load_state_dict(state["opt_G"])
        opt_D.load_state_dict(state["opt_D"])
        start_epoch = int(state.get("epoch", 0)) + 1
        state_metric_name = state.get("best_metric_name")
        if state_metric_name == best_metric_name:
            state_metric_value = state.get("best_metric_value")
            if isinstance(state_metric_value, (int, float)):
                best_metric_value = float(state_metric_value)
                best_metric_epoch = int(state.get("best_metric_epoch", state.get("epoch", 0)))
        elif is_main and state_metric_name is not None:
            print(
                f"[Resume] Checkpoint best metric '{state_metric_name}' differs from configured "
                f"'{best_metric_name}'. Best tracking will restart."
            )
        if is_main:
            print(f"[Resume] Loaded checkpoint {resume_ckpt}, restarting from epoch {start_epoch}.")

    if best_metric_value is None:
        best_ckpt_path = os.path.join(exp_dir, best_ckpt_name)
        if os.path.isfile(best_ckpt_path):
            try:
                best_state = torch.load(best_ckpt_path, map_location="cpu")
                metric_name = best_state.get("best_metric_name", best_metric_name)
                metric_value = best_state.get("best_metric_value")
                if metric_name == best_metric_name and isinstance(metric_value, (int, float)):
                    best_metric_value = float(metric_value)
                    best_metric_epoch = int(best_state.get("best_metric_epoch", best_state.get("epoch", 0)))
                    if is_main:
                        print(
                            f"[Resume] Loaded historical best {best_metric_name}={best_metric_value:.4f} "
                            f"(epoch {best_metric_epoch}) from {best_ckpt_path}."
                        )
            except Exception as exc:
                if is_main:
                    print(f"[Resume] Warning: failed to read {best_ckpt_path}: {exc}")

    for epoch in range(start_epoch, epochs + 1):
        set_linear_lr(opt_G, gen_lr, epoch, decay_start, epochs)
        set_linear_lr(opt_D, disc_lr, epoch, decay_start, epochs)
        if day_sampler:
            day_sampler.set_epoch(epoch)
        if night_sampler:
            night_sampler.set_epoch(epoch)
        G.train()
        F.train()
        D_day.train()
        D_night.train()

        day_iter = iter(day_loader)
        night_iter = iter(night_loader)

        meters = LossMeters()
        samples = 0
        iterator = range(steps_per_epoch)
        if is_main:
            iterator = tqdm(iterator, desc=f"Epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True)

        for step in iterator:
            day_batch, day_iter = next_batch(day_iter, day_loader)
            night_batch, night_iter = next_batch(night_iter, night_loader)

            day = day_batch.to(device)
            night = night_batch.to(device)
            bsz = day.size(0)
            samples += bsz

            # --- Train discriminators ---
            opt_D.zero_grad(set_to_none=True)

            with amp.autocast(device_type=amp_device, enabled=use_amp):
                fake_night = G(day).detach()
                fake_day = F(night).detach()
                fake_night_buf = fake_night_pool.query(fake_night)
                fake_day_buf = fake_day_pool.query(fake_day)

                loss_d_night = 0.5 * (
                    gan_loss(D_night(night), True) + gan_loss(D_night(fake_night_buf), False)
                )
                loss_d_day = 0.5 * (
                    gan_loss(D_day(day), True) + gan_loss(D_day(fake_day_buf), False)
                )
                loss_d = loss_d_day + loss_d_night
            scaler_D.scale(loss_d).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # --- Train generators ---
            opt_G.zero_grad(set_to_none=True)

            with amp.autocast(device_type=amp_device, enabled=use_amp):
                fake_night = G(day)
                fake_day = F(night)

                adv_loss = gan_loss(D_night(fake_night), True) + gan_loss(D_day(fake_day), True)

                rec_day = F(fake_night)
                rec_night = G(fake_day)
                cycle_loss = l1_loss(rec_day, day) + l1_loss(rec_night, night)

                id_day = F(day)
                id_night = G(night)
                id_loss = l1_loss(id_day, day) + l1_loss(id_night, night)

                sky_grad_term = fake_night.new_zeros(())
                sky_tv_term = fake_night.new_zeros(())
                sky_peak_term = fake_night.new_zeros(())
                if use_sky_loss and sky_seg_model is not None:
                    with torch.no_grad(), amp.autocast(device_type=amp_device, enabled=False):
                        seg_in = day
                        if sky_input_size is not None:
                            seg_in = Fnn.interpolate(
                                seg_in,
                                size=sky_input_size,
                                mode="bilinear",
                                align_corners=False,
                            )
                        seg_logits = sky_seg_model(seg_in)
                        seg_pred = seg_logits.argmax(dim=1)
                        mask = torch.zeros_like(seg_pred, dtype=torch.float32)
                        for cid in sky_class_ids:
                            if cid < 0:
                                continue
                            mask = mask + (seg_pred == cid).float()
                        mask = mask.clamp(max=1.0)
                        sky_mask = Fnn.interpolate(mask.unsqueeze(1), size=fake_night.shape[2:], mode="nearest")
                        sky_mask = sky_mask.to(dtype=fake_night.dtype)
                    if torch.any(sky_mask):
                        fake_gray = fake_night.mean(dim=1, keepdim=True)
                        day_gray = day.mean(dim=1, keepdim=True)

                        if sky_grad_weight > 0.0:
                            grad_x_fake = fake_gray[:, :, :, 1:] - fake_gray[:, :, :, :-1]
                            grad_x_day = day_gray[:, :, :, 1:] - day_gray[:, :, :, :-1]
                            grad_y_fake = fake_gray[:, :, 1:, :] - fake_gray[:, :, :-1, :]
                            grad_y_day = day_gray[:, :, 1:, :] - day_gray[:, :, :-1, :]
                            grad_x_fake = Fnn.pad(grad_x_fake, (0, 1, 0, 0))
                            grad_x_day = Fnn.pad(grad_x_day, (0, 1, 0, 0))
                            grad_y_fake = Fnn.pad(grad_y_fake, (0, 0, 0, 1))
                            grad_y_day = Fnn.pad(grad_y_day, (0, 0, 0, 1))
                            grad_diff = (grad_x_fake - grad_x_day).abs() + (grad_y_fake - grad_y_day).abs()
                            sky_grad_term = (grad_diff * sky_mask).mean()

                        if sky_peak_weight > 0.0:
                            sigma = max(sky_peak_sigma, 1e-4)
                            ksize = int(max(3, math.ceil(sigma * 3) * 2 + 1))
                            blurred = gaussian_blur_tensor(fake_gray, ksize, sigma)
                            peaks = (fake_gray - blurred - sky_peak_delta).clamp(min=0.0)
                            sky_peak_term = (peaks * sky_mask).mean()

                        if sky_tv_weight > 0.0:
                            sky_tv_term = total_variation_loss(fake_night * sky_mask)

                sem_loss_total = fake_night.new_zeros(())
                if use_semgan and sem_seg_model is not None and sem_apply_on:
                    real_logits_cache: dict[str, torch.Tensor] = {}

                    def get_real_logits(key: str, domain_tensor: torch.Tensor) -> torch.Tensor:
                        if sem_cache_seg_logits and key in real_logits_cache:
                            return real_logits_cache[key]
                        with torch.no_grad(), amp.autocast(device_type=amp_device, enabled=False):
                            logits = run_segmenter_for_semgan(
                                sem_seg_model,
                                domain_tensor.float(),
                                seg_input_norm=sem_seg_input_norm,
                                seg_inference_size=sem_seg_inference_size,
                            )
                        if sem_cache_seg_logits:
                            real_logits_cache[key] = logits
                        return logits

                    if "A2B" in sem_apply_on:
                        logits_real_a = get_real_logits("A", day)
                        with amp.autocast(device_type=amp_device, enabled=False):
                            logits_fake_b = run_segmenter_for_semgan(
                                sem_seg_model,
                                fake_night.float(),
                                seg_input_norm=sem_seg_input_norm,
                                seg_inference_size=sem_seg_inference_size,
                            )
                        sem_loss_total = sem_loss_total + semantic_consistency_loss(
                            logits_real=logits_real_a,
                            logits_fake=logits_fake_b,
                            sem_loss_type=sem_loss_type,
                            ignore_index=sem_ignore_index,
                        )

                    if "B2A" in sem_apply_on:
                        logits_real_b = get_real_logits("B", night)
                        with amp.autocast(device_type=amp_device, enabled=False):
                            logits_fake_a = run_segmenter_for_semgan(
                                sem_seg_model,
                                fake_day.float(),
                                seg_input_norm=sem_seg_input_norm,
                                seg_inference_size=sem_seg_inference_size,
                            )
                        sem_loss_total = sem_loss_total + semantic_consistency_loss(
                            logits_real=logits_real_b,
                            logits_fake=logits_fake_a,
                            sem_loss_type=sem_loss_type,
                            ignore_index=sem_ignore_index,
                        )

                total_g = adv_loss + lambda_cycle * cycle_loss + lambda_id * id_loss
                if use_sky_loss and sky_seg_model is not None:
                    total_g = total_g + sky_grad_weight * sky_grad_term + sky_peak_weight * sky_peak_term
                    total_g = total_g + sky_tv_weight * sky_tv_term
                if use_semgan and sem_seg_model is not None:
                    total_g = total_g + sem_lambda * sem_loss_total
            scaler_G.scale(total_g).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            batch_losses = {
                "g_total": total_g.item(),
                "g_adv": adv_loss.item(),
                "g_cycle": cycle_loss.item(),
                "g_id": id_loss.item(),
                "g_sky": (
                    sky_grad_weight * sky_grad_term + sky_peak_weight * sky_peak_term + sky_tv_weight * sky_tv_term
                ).item()
                if use_sky_loss and sky_seg_model is not None
                else 0.0,
                "g_sem": (sem_lambda * sem_loss_total).item() if use_semgan and sem_seg_model is not None else 0.0,
                "d_total": loss_d.item(),
            }
            meters.update(batch_losses, bsz)

            if is_main and (step + 1) % log_every == 0 and hasattr(iterator, "set_postfix"):
                avg_local = meters.average(samples)
                postfix = {k: f"{v:.4f}" for k, v in avg_local.items()}
                if not use_sky_loss:
                    postfix.pop("g_sky", None)
                if not use_semgan:
                    postfix.pop("g_sem", None)
                iterator.set_postfix(postfix)

        avg_all, _ = sync_meter_totals(meters, samples, device, distributed)
        if is_main:
            detail = (
                f"adv={avg_all['g_adv']:.4f}, cycle={avg_all['g_cycle']:.4f}, id={avg_all['g_id']:.4f}"
            )
            if use_sky_loss:
                detail += f", sky={avg_all['g_sky']:.4f}"
            if use_semgan:
                detail += f", sem={avg_all['g_sem']:.4f}"
            msg = f"[{epoch}/{epochs}] D={avg_all['d_total']:.4f} G={avg_all['g_total']:.4f} ({detail})"
            print(msg)
            with open(loss_log_path, "a") as log_f:
                log_f.write(msg + "\n")

            if epoch % save_every == 0 or epoch == epochs:
                current_metric = float(avg_all[best_metric_name])
                improved = metric_improved(current_metric, best_metric_value)
                if improved:
                    best_metric_value = current_metric
                    best_metric_epoch = epoch
                state = {
                    "epoch": epoch,
                    "G": (G.module if isinstance(G, (nn.DataParallel, DDP)) else G).state_dict(),
                    "F": (F.module if isinstance(F, (nn.DataParallel, DDP)) else F).state_dict(),
                    "D_day": (D_day.module if isinstance(D_day, (nn.DataParallel, DDP)) else D_day).state_dict(),
                    "D_night": (
                        D_night.module if isinstance(D_night, (nn.DataParallel, DDP)) else D_night
                    ).state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                    "cfg": cfg,
                    "best_metric_name": best_metric_name,
                    "best_metric_mode": best_metric_mode,
                    "best_metric_value": best_metric_value,
                    "best_metric_epoch": best_metric_epoch,
                }
                latest_path = os.path.join(exp_dir, latest_ckpt_name)
                torch.save(state, latest_path)
                print(f"[Checkpoint] Saved latest checkpoint: {latest_path}")

                if improved:
                    best_path = os.path.join(exp_dir, best_ckpt_name)
                    torch.save(state, best_path)
                    print(
                        f"[Checkpoint] Updated best checkpoint ({best_metric_name}={best_metric_value:.4f}, "
                        f"epoch={best_metric_epoch}): {best_path}"
                    )

                if keep_epoch_checkpoints:
                    torch.save(state, os.path.join(exp_dir, f"epoch_{epoch:04d}.pt"))

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/cyclegan.yaml")
    parser.add_argument(
        "--resume",
        help="Path to checkpoint (e.g. latest.pt or best.pt) to resume training.",
    )
    args = parser.parse_args()
    train(args.config, resume=args.resume)
