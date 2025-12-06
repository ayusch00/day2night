import random
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class SegPairTransform:
    def __init__(self, split, resize, crop, hflip):
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}' for segmentation transforms.")
        if resize is None:
            self.resize = None
        elif isinstance(resize, int):
            self.resize = resize
        elif isinstance(resize, (list, tuple)) and len(resize) == 2 and all(isinstance(x, int) for x in resize):
            self.resize = tuple(resize)
        else:
            raise ValueError("resize must be null, an int (shorter side), or a tuple/list of two ints (h, w).")
        if crop is not None and not isinstance(crop, int):
            raise ValueError("crop must be a single int for square crops or null to disable cropping.")
        self.split = split
        self.crop = crop
        self.hflip = hflip and split == "train"

    def _resize(self, img, mask):
        if self.resize is None:
            return img, mask
        img = F.resize(img, self.resize, interpolation=InterpolationMode.BICUBIC, antialias=True)
        mask = F.resize(mask, self.resize, interpolation=InterpolationMode.NEAREST, antialias=False)
        return img, mask

    def _crop(self, img, mask):
        if self.crop is None:
            return img, mask
        if self.split == "train":
            i, j, h, w = T.RandomCrop.get_params(img, (self.crop, self.crop))
            img = F.crop(img, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)
        else:
            img = F.center_crop(img, self.crop)
            mask = F.center_crop(mask, self.crop)
        return img, mask

    def _maybe_hflip(self, img, mask):
        if self.hflip and random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

    def __call__(self, img, mask):
        img, mask = self._resize(img, mask)
        img, mask = self._crop(img, mask)
        img, mask = self._maybe_hflip(img, mask)
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask


def make_transforms(split, crop=512, resize=572, hflip=True):
    """
    Resize (optional) -> optional crop -> optional flip -> [-1,1] norm.
    Accepts integer resize (keeps aspect ratio), (h, w) tuple for absolute sizing, or None to skip resizing.
    """
    return SegPairTransform(split, resize, crop, hflip)
