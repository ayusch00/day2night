import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def cityscapes_mask_path(img_path, mask_root):
    # .../leftImg8bit/train/<city>/<base>_leftImg8bit.png
    # -> .../gtFine/train/<city>/<base>_gtFine_labelTrainIds.png
    city = os.path.basename(os.path.dirname(img_path))
    base = os.path.basename(img_path).replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
    return os.path.join(mask_root, city, base)

class SegDataset(Dataset):
    def __init__(self, img_root, mask_root, img_t, mask_t, ignore_index=255):
        self.imgs = sorted(glob.glob(os.path.join(img_root, "*", "*_leftImg8bit.png")))
        self.mask_root = mask_root
        self.img_t, self.mask_t = img_t, mask_t
        self.ignore_index = ignore_index
    def __getitem__(self, i):
        from PIL import Image
        img_path = self.imgs[i]
        mask_path = cityscapes_mask_path(img_path, self.mask_root)
        img = self.img_t(Image.open(img_path).convert("RGB"))
        mask = Image.open(mask_path)
        mask = torch.from_numpy(np.array(self.mask_t(mask), dtype=np.int64))
        return img, mask
    def __len__(self):
        return len(self.imgs)
