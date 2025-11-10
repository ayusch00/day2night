import glob
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from src.models.cyclegan import CycleGANGenerator, PatchDiscriminator, GANLoss
from src.utils.common import load_cfg, set_seed, make_run_dirs


DEFAULT_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")


def build_transform(cfg: dict) -> transforms.Compose:
    ops: list = []
    resize = cfg.get("resize")
    if resize:
        ops.append(transforms.Resize((resize, resize)))
    crop = cfg.get("random_crop")
    if crop:
        ops.append(transforms.RandomCrop(crop))
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
    d_total: float = 0.0

    def update(self, losses: dict[str, float], batch_size: int):
        for k, v in losses.items():
            setattr(self, k, getattr(self, k) + v * batch_size)

    def average(self, samples: int) -> dict[str, float]:
        return {k: getattr(self, k) / samples for k in ["g_total", "g_adv", "g_cycle", "g_id", "d_total"]}


def set_linear_lr(optimizer, base_lr, epoch, decay_start, total_epochs):
    if total_epochs <= decay_start or epoch <= decay_start:
        factor = 1.0
    else:
        decay_epochs = max(1, total_epochs - decay_start)
        factor = max(0.0, 1.0 - (epoch - decay_start) / decay_epochs)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor


def train(cfg_path: str = "configs/cyclegan.yaml"):
    cfg = load_cfg(cfg_path)
    set_seed(cfg["train"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    dl_kwargs = dict(
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("workers", 4),
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    day_loader = DataLoader(day_ds, **dl_kwargs)
    night_loader = DataLoader(night_ds, **dl_kwargs)

    gen_cfg = cfg["model"]["generator"]
    disc_cfg = cfg["model"]["discriminator"]

    encoder_ckpt = gen_cfg.get("encoder_checkpoint")
    freeze_encoder = gen_cfg.get("freeze_encoder", False)
    decoder_res_blocks = gen_cfg.get("decoder_res_blocks", 3)

    G = CycleGANGenerator(
        in_channels=gen_cfg.get("in_channels", 3),
        out_channels=gen_cfg.get("out_channels", 3),
        base_channels=gen_cfg.get("base_channels", 64),
        n_res_blocks=gen_cfg.get("n_res_blocks", 9),
        use_skip=gen_cfg.get("use_skip", True),
        encoder_checkpoint=encoder_ckpt,
        freeze_encoder=freeze_encoder,
        decoder_res_blocks=decoder_res_blocks,
    ).to(device)

    F = CycleGANGenerator(
        in_channels=gen_cfg.get("in_channels", 3),
        out_channels=gen_cfg.get("out_channels", 3),
        base_channels=gen_cfg.get("base_channels", 64),
        n_res_blocks=gen_cfg.get("n_res_blocks", 9),
        use_skip=gen_cfg.get("use_skip", True),
        encoder_checkpoint=encoder_ckpt,
        freeze_encoder=freeze_encoder,
        decoder_res_blocks=decoder_res_blocks,
    ).to(device)

    D_day = PatchDiscriminator(
        in_channels=disc_cfg.get("in_channels", 3),
        base_channels=disc_cfg.get("base_channels", 64),
        n_layers=disc_cfg.get("n_layers", 3),
        max_channels=disc_cfg.get("max_channels", 512),
    ).to(device)

    D_night = PatchDiscriminator(
        in_channels=disc_cfg.get("in_channels", 3),
        base_channels=disc_cfg.get("base_channels", 64),
        n_layers=disc_cfg.get("n_layers", 3),
        max_channels=disc_cfg.get("max_channels", 512),
    ).to(device)

    if torch.cuda.device_count() > 1:
        G = nn.DataParallel(G)
        F = nn.DataParallel(F)
        D_day = nn.DataParallel(D_day)
        D_night = nn.DataParallel(D_night)

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
    save_every = cfg["logging"].get("save_every", 10)
    log_every = cfg["logging"].get("log_interval", 50)

    exp_dir, _ = make_run_dirs(cfg)

    # Persist the resolved training image lists for each domain.
    for domain, dataset in (("day", day_ds), ("night", night_ds)):
        filelist_path = os.path.join(exp_dir, f"{domain}_files.txt")
        with open(filelist_path, "w") as fh:
            for path in dataset.paths:
                fh.write(f"{path}\n")
        print(f"Saved list of {len(dataset.paths)} {domain} images to {filelist_path}")

    for epoch in range(1, epochs + 1):
        set_linear_lr(opt_G, gen_lr, epoch, decay_start, epochs)
        set_linear_lr(opt_D, disc_lr, epoch, decay_start, epochs)
        G.train()
        F.train()
        D_day.train()
        D_night.train()

        day_iter = iter(day_loader)
        night_iter = iter(night_loader)

        meters = LossMeters()
        samples = 0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        for step in pbar:
            day_batch, day_iter = next_batch(day_iter, day_loader)
            night_batch, night_iter = next_batch(night_iter, night_loader)

            day = day_batch.to(device)
            night = night_batch.to(device)
            bsz = day.size(0)
            samples += bsz

            # --- Train discriminators ---
            opt_D.zero_grad(set_to_none=True)

            fake_night = G(day).detach()
            fake_day = F(night).detach()

            loss_d_night = 0.5 * (
                gan_loss(D_night(night), True) + gan_loss(D_night(fake_night), False)
            )
            loss_d_day = 0.5 * (
                gan_loss(D_day(day), True) + gan_loss(D_day(fake_day), False)
            )
            loss_d = loss_d_day + loss_d_night
            loss_d.backward()
            opt_D.step()

            # --- Train generators ---
            opt_G.zero_grad(set_to_none=True)

            fake_night = G(day)
            fake_day = F(night)

            adv_loss = gan_loss(D_night(fake_night), True) + gan_loss(D_day(fake_day), True)

            rec_day = F(fake_night)
            rec_night = G(fake_day)
            cycle_loss = l1_loss(rec_day, day) + l1_loss(rec_night, night)

            id_day = F(day)
            id_night = G(night)
            id_loss = l1_loss(id_day, day) + l1_loss(id_night, night)

            total_g = adv_loss + lambda_cycle * cycle_loss + lambda_id * id_loss
            total_g.backward()
            opt_G.step()

            batch_losses = {
                "g_total": total_g.item(),
                "g_adv": adv_loss.item(),
                "g_cycle": cycle_loss.item(),
                "g_id": id_loss.item(),
                "d_total": loss_d.item(),
            }
            meters.update(batch_losses, bsz)

            if (step + 1) % log_every == 0:
                avg = meters.average(samples)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in avg.items()})

        avg = meters.average(samples)
        print(
            f"[{epoch}/{epochs}] "
            f"D={avg['d_total']:.4f} G={avg['g_total']:.4f} "
            f"(adv={avg['g_adv']:.4f}, cycle={avg['g_cycle']:.4f}, id={avg['g_id']:.4f})"
        )

        if epoch % save_every == 0 or epoch == epochs:
            state = {
                "epoch": epoch,
                "G": G.state_dict(),
                "F": F.state_dict(),
                "D_day": D_day.state_dict(),
                "D_night": D_night.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "cfg": cfg,
            }
            torch.save(state, os.path.join(exp_dir, f"epoch_{epoch:04d}.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/cyclegan.yaml")
    args = parser.parse_args()
    train(args.config)
