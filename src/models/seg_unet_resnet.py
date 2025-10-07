import os, glob, random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------- Building blocks ----------

def conv7(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_c, out_c, kernel_size=7, stride=stride, padding=0, bias=False),
        nn.InstanceNorm2d(out_c, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True),
    )

def downsample(in_c, out_c):  # 3x3, stride=2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_c, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True),
    )

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(c, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(c, affine=False, track_running_stats=False),
        )
    def forward(self, x):
        return x + self.block(x)

def up_block(in_c, out_c):
    # Nearest-Neighbor Upsample + 3x3 Conv (stabiler als TransposedConv)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
        nn.InstanceNorm2d(out_c, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True),
    )

# ---------- Encoder (SE) = CycleGAN-kompatibel ----------
class Encoder9Res(nn.Module):
    def __init__(self, in_c=3, base=64, n_res=9):
        super().__init__()
        self.c7s1_64 = conv7(in_c, base)         # -> 64, HxW
        self.d128    = downsample(base, base*2)  # -> 128, H/2 x W/2
        self.d256    = downsample(base*2, base*4)# -> 256, H/4 x W/4

        self.res = nn.Sequential(*[ResBlock(base*4) for _ in range(n_res)])  # 9 ResBlocks @256ch

    def forward(self, x):
        e1 = self.c7s1_64(x)     # skip1 (64ch)
        e2 = self.d128(e1)       # skip2 (128ch)
        e3 = self.d256(e2)       # bottleneck in (256ch, H/4)
        r  = self.res(e3)        # bottleneck out (256ch)
        return r, (e2, e1)       # return skips (from high→low)

# ---------- Decoder (SD) mit U-Net Skips ----------
class DecoderUNet(nn.Module):
    def __init__(self, base=64, num_classes=19):
        super().__init__()
        # Up von 256->128
        self.up1 = up_block(base*4, base*2)     # 256->128, H/4->H/2
        self.fuse1 = nn.Sequential(             # fuse mit skip e2 (128ch)
            nn.Conv2d(base*2 + base*2, base*2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(base*2, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        # Up von 128->64
        self.up2 = up_block(base*2, base)       # 128->64, H/2->H
        self.fuse2 = nn.Sequential(             # fuse mit skip e1 (64ch)
            nn.Conv2d(base + base, base, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(base, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        # Head (Logits)
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base, num_classes, kernel_size=7, stride=1, padding=0)  # logits (no softmax)
        )

    def forward(self, bottleneck, skips):
        e2, e1 = skips          # e2: 128ch @ H/2, e1: 64ch @ H
        x = self.up1(bottleneck)
        x = self.fuse1(torch.cat([x, e2], dim=1))
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, e1], dim=1))
        logits = self.head(x)
        return logits
    
# ---------- Full Model ----------    
class SegNet9ResUNet(nn.Module):
    def __init__(self, num_classes=19, base=64, n_res=9):
        super().__init__()
        self.encoder = Encoder9Res(base=base, n_res=n_res)
        self.decoder = DecoderUNet(base=base, num_classes=num_classes)
    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        return self.decoder(bottleneck, skips)