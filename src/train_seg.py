# src/train_seg_min.py
import os, torch, torch.nn as nn, random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from tqdm.auto import tqdm
from src.utils.common import load_cfg, set_seed
from src.utils.transforms import make_transforms
from src.utils.data_loading import SegDataset
from src.models.seg_unet_resnet import SegNet9ResUNet

def init_distributed() -> tuple[bool, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = 0
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requested but CUDA is unavailable.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    return distributed, local_rank

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def main(cfg_path="configs/seg.yaml"):
    cfg = load_cfg(cfg_path)
    distributed, local_rank = init_distributed()
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0
    base_seed = cfg["train"]["seed"]
    set_seed(base_seed + rank)

    img_t, mask_t = make_transforms(
        crop=cfg["transforms"]["crop"],
        resize=cfg["transforms"]["resize"],
        hflip=cfg["transforms"].get("hflip", True),
    )

    train_ds = SegDataset(
        img_root=cfg["data"]["train_images"],
        mask_root=cfg["data"]["train_masks"],
        img_t=img_t, mask_t=mask_t,
        ignore_index=cfg["data"]["ignore_index"],
    )
    n = cfg["data"].get("sample_n_train")
    if n and n < len(train_ds):
        idx = list(range(len(train_ds)))
        random.Random(base_seed).shuffle(idx)
        train_ds = Subset(train_ds, idx[:n])

    # Persist the exact list of images that participate in training for reproducibility.
    if isinstance(train_ds, Subset):
        base_imgs = train_ds.dataset.imgs
        selected_imgs = [base_imgs[i] for i in train_ds.indices]
    else:
        selected_imgs = train_ds.imgs
    out_dir = os.path.join(cfg.get("logging", {}).get("out_dir", "experiments"), cfg["exp_name"])
    os.makedirs(out_dir, exist_ok=True)
    filelist_path = os.path.join(out_dir, "train_files.txt")
    if is_main:
        with open(filelist_path, "w") as fh:
            for path in selected_imgs:
                fh.write(f"{path}\n")
        print(f"Saved list of {len(selected_imgs)} training images to {filelist_path}")

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False) if distributed else None
    workers = cfg["train"]["workers"]
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=not distributed,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        sampler=train_sampler,
        persistent_workers=workers > 0,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for segmentation training but none was found.")
    device = torch.device(f"cuda:{local_rank}") if distributed else torch.device("cuda")
    model = SegNet9ResUNet(
        num_classes=cfg["data"]["num_classes"],
        base=cfg["model"]["base_channels"],
        n_res=cfg["model"]["n_resblocks"],
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("optim", {}).get("lr", 2e-4),
        betas=tuple(cfg.get("optim", {}).get("betas", [0.5, 0.999])),
    )
    crit = nn.CrossEntropyLoss(ignore_index=cfg["data"]["ignore_index"])

    for ep in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total = 0.0
        processed = 0
        if distributed:
            train_sampler.set_epoch(ep)
        pbar = tqdm(
            train_dl,
            desc=f"Epoch {ep}/{cfg['train']['epochs']}",
            leave=False,
            dynamic_ncols=True,
            disable=not is_main,
        )
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            processed += x.size(0)
            if is_main:
                pbar.set_postfix(loss=loss.item())
        totals = torch.tensor([total, processed], device=device)
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss = totals[0].item()
        total_samples = max(1, int(totals[1].item()))
        if is_main:
            print(f"[{ep}/{cfg['train']['epochs']}] loss={total_loss/total_samples:.4f}")

    out = f"experiments/{cfg['exp_name']}_encoder_GE.pth"
    core_model = model.module if isinstance(model, DDP) else model
    if is_main:
        torch.save(core_model.encoder.state_dict(), out)
        print(f"Saved encoder to {out}")
    cleanup_distributed()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="configs/seg.yaml")
    args = p.parse_args()
    main(args.config)
