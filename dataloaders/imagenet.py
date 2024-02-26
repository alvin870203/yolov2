"""
Training & validation dataloaders of ImageNet2012 classification dataset.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2


@dataclass
class ImageNetConfig:
    img_h: int = 224
    img_w: int = 224
    imgs_mean: Tuple = (0.485, 0.456, 0.406)
    imgs_std: Tuple = (0.229, 0.224, 0.225)
    brightness: float = 0.75
    contrast: float = 0.75
    saturation: float = 0.75
    hue: float = 0.1
    degrees: float = 7.0
    scale_min: float = 0.25
    scale_max: float = 1.25
    ratio_min: float = 3.0 / 4.0
    ratio_max: float = 4.0 / 3.0


class ImageNetTrainDataLoader(DataLoader):
    def __init__(self, config: ImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True):
        self.config = config
        dataset = torchvision.datasets.ImageNet(
            data_dir, split='train',
            transform=v2.Compose([
                v2.ToImage(),
                v2.Resize(size=(config.img_h, config.img_w), antialias=True),
                v2.ColorJitter(brightness=config.brightness, contrast=config.contrast,
                               saturation=config.saturation, hue=config.hue),
                v2.Pad(padding=(int(config.img_h * (config.scale_max - 1) / 2),
                                int(config.img_w * (config.scale_max - 1) / 2)),
                       fill=0, padding_mode='constant'),
                v2.RandomRotation(degrees=config.degrees),
                v2.RandomResizedCrop(size=(config.img_h, config.img_w), scale=(config.scale_min, 1.0),
                                     ratio=(config.ratio_min, config.ratio_max), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory)


class ImageNetValDataLoader(DataLoader):
    # Default shuffle=True since only eval partial data
    def __init__(self, config: ImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True):
        self.config = config
        dataset = torchvision.datasets.ImageNet(
            data_dir, split='val',
            transform=v2.Compose([
                v2.ToImage(),
                v2.Resize(size=(config.img_h, config.img_w), antialias=True),
                v2.CenterCrop(size=(config.img_h, config.img_w)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory)



if __name__ == '__main__':
    # Test the dataloaders by `python -m dataloaders.imagenet` from the workspace directory
    import matplotlib.pyplot as plt
    data_dir = 'data/imagenet2012'
    config = ImageNetConfig()
    dataloader_train = ImageNetTrainDataLoader(config, data_dir, batch_size=32, num_workers=4)
    dataloader_val = ImageNetValDataLoader(config, data_dir, batch_size=32, shuffle=False, num_workers=4)
    print(f"{len(dataloader_train)=}")
    print(f"{len(dataloader_val)=}")
    example_img = next(iter(dataloader_train))[0][0]
    # Unnormalize the image for plotting
    example_img = example_img * torch.tensor(config.imgs_std).reshape(3, 1, 1) + torch.tensor(config.imgs_mean).reshape(3, 1, 1)
    print(f"{example_img.shape=}")
    print(f"{example_img.mean()=}, {example_img.std()=}")
    print(f"{example_img.min()=}, {example_img.max()=}")
    plt.imshow(example_img.permute(1, 2, 0))
    plt.show()
