from torchvision import transforms
from PIL import Image

def make_transforms(crop, resize, hflip=True):
    img_t = [transforms.Resize((crop, crop))]
    if hflip: img_t.append(transforms.RandomHorizontalFlip())
    img_t += [
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1] wie CycleGAN
    ]
    mask_resize = transforms.Compose([
        transforms.Resize((crop, crop), interpolation=Image.NEAREST),
        transforms.Resize((resize, resize), interpolation=Image.NEAREST)
    ])
    return transforms.Compose(img_t), mask_resize
