# src/train_seg_min.py
import os, torch, torch.nn as nn, random, time, shutil
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
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

def compute_class_weights(mask_paths, num_classes, ignore_index, desc="Class hist"):
    counts = np.zeros(num_classes, dtype=np.float64)
    for path in tqdm(mask_paths, desc=desc, leave=False):
        mask = np.array(Image.open(path), dtype=np.int64)
        valid = mask != ignore_index
        if not np.any(valid):
            continue
        vals, freq = np.unique(mask[valid], return_counts=True)
        for v, f in zip(vals, freq):
            if 0 <= v < num_classes:
                counts[v] += f
    total = counts.sum()
    if total == 0:
        return np.ones(num_classes, dtype=np.float32)
    freq = counts / total
    weights = 1.0 / (freq + 1e-6)
    weights[counts == 0] = 0.0
    positive = weights[weights > 0]
    if positive.size > 0:
        weights /= positive.mean()
    return weights.astype(np.float32)

def infer_num_classes(mask_paths, ignore_index):
    max_class = -1
    for path in tqdm(mask_paths, desc="Infer classes", leave=False):
        mask = np.array(Image.open(path), dtype=np.int64)
        valid = mask != ignore_index
        if not np.any(valid):
            continue
        max_class = max(max_class, mask[valid].max())
    if max_class < 0:
        raise RuntimeError(
            "Unable to infer number of classes. Ensure masks contain labels other than the ignore_index."
        )
    return int(max_class + 1)

@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index, distributed):
    model.eval()
    conf = torch.zeros((num_classes, num_classes), device=device)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(1)
        valid = labels != ignore_index
        if not torch.any(valid):
            continue
        y = labels[valid].view(-1)
        p = preds[valid].view(-1)
        k = (y * num_classes + p).long()
        binc = torch.bincount(k, minlength=num_classes ** 2)
        conf += binc.view(num_classes, num_classes)
    if distributed:
        dist.all_reduce(conf)
    inter = torch.diag(conf)
    union = conf.sum(1) + conf.sum(0) - inter
    iu = torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(inter))
    miou = iu.mean().item()
    pix_acc = inter.sum().item() / conf.sum().clamp_min(1.0).item()
    return miou, pix_acc

def main(cfg_path="configs/seg.yaml"):
    cfg = load_cfg(cfg_path)
    cfg_path = os.path.abspath(cfg_path)
    distributed, local_rank = init_distributed()
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0
    base_seed = cfg["train"]["seed"]
    set_seed(base_seed + rank)
    use_amp = cfg["train"].get("amp", False)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_cfg = cfg.get("logging", {})
    val_every = max(1, int(log_cfg.get("val_every", 1)))
    out_root = log_cfg.get("out_dir", "experiments")
    run_dir = os.path.join(out_root, f"{cfg['exp_name']}_{ts}") if is_main else None
    config_filename = os.path.basename(cfg_path)
    if distributed:
        payload = [run_dir]
        if is_main:
            os.makedirs(run_dir, exist_ok=True)
            shutil.copy(cfg_path, os.path.join(run_dir, config_filename))
        dist.broadcast_object_list(payload, src=0)
        run_dir = payload[0]
    else:
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(cfg_path, os.path.join(run_dir, config_filename))
    loss_log_path = os.path.join(run_dir, "loss_log.txt")

    train_tf = make_transforms(
        split="train",
        crop=cfg["transforms"]["crop"],
        resize=cfg["transforms"]["resize"],
        hflip=cfg["transforms"].get("hflip", True),
    )
    val_tf = make_transforms(
        split="val",
        crop=cfg["transforms"]["crop"],
        resize=cfg["transforms"]["resize"],
        hflip=False,
    )

    data_cfg = cfg["data"]
    dataset_kind = data_cfg.get("dataset", "cityscapes")
    extensions = data_cfg.get("extensions")

    train_ds = SegDataset(
        img_root=data_cfg["train_images"],
        mask_root=data_cfg["train_masks"],
        img_t=None,
        mask_t=None,
        pair_t=train_tf,
        ignore_index=data_cfg["ignore_index"],
        dataset=dataset_kind,
        extensions=extensions,
    )
    n = data_cfg.get("sample_n_train")
    if n and n < len(train_ds):
        idx = list(range(len(train_ds)))
        random.Random(base_seed).shuffle(idx)
        train_ds = Subset(train_ds, idx[:n])

    # Persist the exact list of images that participate in training for reproducibility.
    if isinstance(train_ds, Subset):
        base_dataset = train_ds.dataset
        indices = train_ds.indices
        selected_imgs = [base_dataset.imgs[i] for i in indices]
    else:
        base_dataset = train_ds
        indices = list(range(len(train_ds)))
        selected_imgs = base_dataset.imgs
    filelist_path = os.path.join(run_dir, "train_files.txt")
    if is_main:
        with open(filelist_path, "w") as fh:
            for path in selected_imgs:
                fh.write(f"{path}\n")
        print(f"Saved list of {len(selected_imgs)} training images to {filelist_path}")

    mask_paths = None
    if is_main and (data_cfg.get("use_class_weights", False) or data_cfg.get("num_classes") is None):
        mask_paths = [base_dataset._mask_path(img_path) for img_path in selected_imgs]

    if data_cfg.get("num_classes") is None:
        inferred = None
        if is_main:
            inferred = infer_num_classes(mask_paths, data_cfg["ignore_index"])
            print(f"Inferred {inferred} segmentation classes from masks.")
        if distributed:
            payload = [inferred]
            dist.broadcast_object_list(payload, src=0)
            inferred = payload[0]
        data_cfg["num_classes"] = int(inferred)
    num_classes = int(data_cfg["num_classes"])

    class_weight_list = None
    if data_cfg.get("use_class_weights", False):
        if is_main:
            class_weight_list = compute_class_weights(
                mask_paths,
                num_classes=num_classes,
                ignore_index=data_cfg["ignore_index"],
                desc="Class weights",
            ).tolist()
        if distributed:
            payload = [class_weight_list]
            dist.broadcast_object_list(payload, src=0)
            class_weight_list = payload[0]

    world_size = dist.get_world_size() if distributed else 1
    val_ds = SegDataset(
        img_root=data_cfg["val_images"],
        mask_root=data_cfg["val_masks"],
        img_t=None,
        mask_t=None,
        pair_t=val_tf,
        ignore_index=data_cfg["ignore_index"],
        dataset=dataset_kind,
        extensions=extensions,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if distributed else None
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
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        sampler=val_sampler,
        persistent_workers=False,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for segmentation training but none was found.")
    device = torch.device(f"cuda:{local_rank}") if distributed else torch.device("cuda")
    if use_amp and device.type != "cuda":
        raise RuntimeError("AMP requested but CUDA device not available.")
    model = SegNet9ResUNet(
        num_classes=cfg["data"]["num_classes"],
        base=cfg["model"]["base_channels"],
        n_res=cfg["model"]["n_resblocks"],
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    global_batch = cfg["train"]["batch_size"] * world_size
    ref_batch = max(1, cfg["train"].get("global_batch_ref", global_batch))
    base_lr = cfg.get("optim", {}).get("lr", 2e-4)
    scaled_lr = base_lr * (global_batch / ref_batch)
    if is_main:
        print(f"SegNet global batch={global_batch}, ref={ref_batch}, lr={scaled_lr:.6f}")
    opt = torch.optim.Adam(
        model.parameters(),
        lr=scaled_lr,
        betas=tuple(cfg.get("optim", {}).get("betas", [0.5, 0.999])),
    )
    sched_cfg = cfg.get("scheduler", {})
    scheduler = None
    if sched_cfg:
        policy = sched_cfg.get("policy", "").lower()
        if policy == "linear_decay_after_warm":
            warm_epochs = int(sched_cfg.get("warm_epochs", 0))
            decay_epochs = max(1, int(sched_cfg.get("decay_epochs", 1)))
            min_lr = float(sched_cfg.get("min_lr", 0.0))
            min_factor = 0.0 if scaled_lr <= 0 else min_lr / scaled_lr
            min_factor = max(0.0, min(min_factor, 1.0))

            def lr_lambda(epoch):
                if epoch < warm_epochs:
                    return 1.0
                progress = min(1.0, (epoch - warm_epochs) / decay_epochs)
                return max(min_factor, 1.0 - progress * (1.0 - min_factor))

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        elif policy == "cosine":
            t_max = int(sched_cfg.get("t_max", cfg["train"]["epochs"]))
            eta_min = float(sched_cfg.get("min_lr", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)

    class_weights = (
        torch.tensor(class_weight_list, dtype=torch.float32, device=device)
        if class_weight_list is not None else None
    )
    crit = nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg["data"]["ignore_index"])
    scaler = GradScaler(enabled=use_amp)

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
            with autocast(enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
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
            msg = f"[{ep}/{cfg['train']['epochs']}] loss={total_loss/total_samples:.4f}"
            print(msg)
            with open(loss_log_path, "a") as log_f:
                log_f.write(msg + "\n")

        run_validation = (ep % val_every == 0)
        if run_validation:
            if val_sampler is not None:
                val_sampler.set_epoch(ep)
            miou, pix_acc = evaluate(
                model,
                val_dl,
                device=device,
                num_classes=cfg["data"]["num_classes"],
                ignore_index=cfg["data"]["ignore_index"],
                distributed=distributed,
            )
            if is_main:
                val_msg = f"[val:{ep}] mIoU={miou:.4f}, pixAcc={pix_acc:.4f}"
                print(val_msg)
                with open(loss_log_path, "a") as log_f:
                    log_f.write(val_msg + "\n")
        if scheduler is not None:
            scheduler.step()

    artifacts_cfg = cfg.get("artifacts", {})
    enc_name = artifacts_cfg.get("encoder_weights", "encoder_GE.pth")
    full_name = artifacts_cfg.get("full_model", "segnet_full.pth")
    core_model = model.module if isinstance(model, DDP) else model
    if is_main:
        enc_out = os.path.join(run_dir, enc_name)
        torch.save(core_model.encoder.state_dict(), enc_out)
        print(f"Saved encoder to {enc_out}")
        if full_name:
            full_out = os.path.join(run_dir, full_name)
            torch.save(core_model.state_dict(), full_out)
            print(f"Saved full SegNet to {full_out}")
    cleanup_distributed()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="configs/seg.yaml")
    args = p.parse_args()
    main(args.config)
