# src/train_seg_min.py
import torch, torch.nn as nn, random
from torch.utils.data import DataLoader, Subset
from src.utils.common import load_cfg, set_seed
from src.utils.transforms import make_transforms
from src.utils.data_loading import SegDataset
from src.models.seg_unet_resnet import SegNet9ResUNet

def main(cfg_path="configs/seg.yaml"):
    cfg = load_cfg(cfg_path)
    set_seed(cfg["train"]["seed"])

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
        random.Random(cfg["train"]["seed"]).shuffle(idx)
        train_ds = Subset(train_ds, idx[:n])

    train_dl = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=cfg["train"]["workers"], pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegNet9ResUNet(
        num_classes=cfg["data"]["num_classes"],
        base=cfg["model"]["base_channels"],
        n_res=cfg["model"]["n_resblocks"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("optim", {}).get("lr", 2e-4),
        betas=tuple(cfg.get("optim", {}).get("betas", [0.5, 0.999])),
    )
    crit = nn.CrossEntropyLoss(ignore_index=cfg["data"]["ignore_index"])

    for ep in range(1, cfg["train"]["epochs"] + 1):
        model.train(); total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y)
            loss.backward(); opt.step()
            total += loss.item() * x.size(0)
        print(f"[{ep}/{cfg['train']['epochs']}] loss={total/len(train_ds):.4f}")

    out = f"experiments/{cfg['exp_name']}_encoder_GE.pth"
    torch.save(model.encoder.state_dict(), out)
    print(f"Saved encoder to {out}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="configs/seg.yaml")
    args = p.parse_args()
    main(args.config)
