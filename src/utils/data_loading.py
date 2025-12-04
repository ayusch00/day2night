import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

CITYSCAPES_SUFFIX = "_leftImg8bit.png"
DEFAULT_EXTENSIONS = ("jpg", "jpeg", "png")

def cityscapes_mask_path(img_path, mask_root):
    # .../leftImg8bit/train/<city>/<base>_leftImg8bit.png
    # -> .../gtFine/train/<city>/<base>_gtFine_labelTrainIds.png
    city = os.path.basename(os.path.dirname(img_path))
    base = os.path.basename(img_path).replace(CITYSCAPES_SUFFIX, "_gtFine_labelTrainIds.png")
    return os.path.join(mask_root, city, base)

def bdd100k_mask_path(img_path, mask_root, strict=True):
    base = os.path.splitext(os.path.basename(img_path))[0]
    candidates = [f"{base}.png", f"{base}_train_id.png"]
    for name in candidates:
        mask_path = os.path.join(mask_root, name)
        if os.path.isfile(mask_path):
            return mask_path
    if strict:
        raise FileNotFoundError(f"Missing mask for {img_path}: expected one of {candidates}")
    return None

class SegDataset(Dataset):
    def __init__(
        self,
        img_root,
        mask_root,
        img_t,
        mask_t,
        pair_t=None,
        ignore_index=255,
        dataset="cityscapes",
        extensions=None,
    ):
        self.mask_root = mask_root
        self.mask_lookup = None
        self.img_t, self.mask_t = img_t, mask_t
        self.pair_t = pair_t
        self.ignore_index = ignore_index
        self.dataset = dataset
        self.extensions = tuple(extensions) if extensions else DEFAULT_EXTENSIONS

        if dataset == "cityscapes":
            pattern = os.path.join(img_root, "*", f"*{CITYSCAPES_SUFFIX}")
            self.imgs = sorted(glob.glob(pattern))
            self.mask_resolver = cityscapes_mask_path
        elif dataset == "bdd100k":
            # Allow nested dirs (train/val) and different extensions.
            paths = []
            for ext in self.extensions:
                paths.extend(glob.glob(os.path.join(img_root, f"**/*.{ext}"), recursive=True))
            unique = sorted(set(paths))
            pairs = []
            missing = 0
            for img_path in unique:
                mask_path = bdd100k_mask_path(img_path, mask_root, strict=False)
                if mask_path:
                    pairs.append((img_path, mask_path))
                else:
                    missing += 1
            self.imgs = [img for img, _ in pairs]
            self.mask_lookup = {img: mask for img, mask in pairs}
            if missing > 0:
                print(
                    f"[SegDataset] Skipped {missing} BDD images without masks under {mask_root}."
                )
            self.mask_resolver = bdd100k_mask_path
        else:
            raise ValueError(f"Unsupported segmentation dataset '{dataset}'.")

        if not self.imgs:
            raise RuntimeError(f"No images found for dataset '{dataset}' under {img_root}.")

    def _mask_path(self, img_path):
        if self.mask_lookup is not None:
            mask_path = self.mask_lookup.get(img_path)
            if mask_path is not None:
                return mask_path
        return self.mask_resolver(img_path, self.mask_root)

    def __getitem__(self, i):
        from PIL import Image
        img_path = self.imgs[i]
        mask_path = self._mask_path(img_path)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing mask for {img_path}: expected {mask_path}")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.pair_t:
            img, mask = self.pair_t(img, mask)
            return img, mask
        img = self.img_t(img) if self.img_t else img
        mask = self.mask_t(mask) if self.mask_t else mask
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask

    def __len__(self):
        return len(self.imgs)
