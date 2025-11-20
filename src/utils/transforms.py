import random
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class SegPairTransform:
    def __init__(self, split, crop, final, hflip):
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}' for segmentation transforms.")
        self.split = split
        self.crop = crop
        self.final = final
        self.hflip = hflip and split == "train"

    def _crop(self, img, mask):
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
        img, mask = self._crop(img, mask)
        img, mask = self._maybe_hflip(img, mask)
        img = F.resize(img, (self.final, self.final), interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = F.resize(mask, (self.final, self.final), interpolation=InterpolationMode.NEAREST)
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img, mask


def make_transforms(split, crop=512, final=256, hflip=True):
    """
    Match the paper: RandomCrop/CenterCrop -> Resize -> optional flip -> [-1,1] norm.
    Returns a callable that receives (img, mask) and returns aligned tensors.
    """
    return SegPairTransform(split, crop, final, hflip)
