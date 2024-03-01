"""
Training & validation dataloaders of ImageNet2012 classification dataset.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._utils import _FillType


class ZoomOut(nn.Module):
    def __init__(self,
                 scale: float,
                 fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = 0,
                 padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",) -> None:
        super().__init__()
        self.scale = scale
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = img.shape[-2:]
        new_h, new_w = int(orig_h * self.scale), int(orig_w * self.scale)
        padding = ((new_w - orig_w) // 2, (new_h - orig_h) // 2)
        return F.pad(img, padding, self.fill, self.padding_mode)


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
    flip_p: float = 0.5


class ImageNetTrainDataLoader(DataLoader):
    def __init__(self, config: ImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True):
        self.config = config
        dataset = torchvision.datasets.ImageNet(
            data_dir, split='train',
            transform=v2.Compose([
                v2.ToImage(),

                # TODO: test simple augmentation
                # v2.Resize(size=max(config.img_h, config.img_w), antialias=True),
                # v2.ColorJitter(brightness=config.brightness, contrast=config.contrast,
                #                saturation=config.saturation, hue=config.hue),
                # ZoomOut(scale=config.scale_max, fill=0, padding_mode='constant'),
                # v2.RandomRotation(degrees=config.degrees),

                v2.RandomResizedCrop(size=(config.img_h, config.img_w), scale=(config.scale_min, 1.0),
                                     ratio=(config.ratio_min, config.ratio_max), antialias=True),
                v2.RandomHorizontalFlip(p=config.flip_p),
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

                # TODO: test simple augmentation
                # v2.Resize(size=max(config.img_h, config.img_w), antialias=True),
                v2.Resize(size=int(max(config.img_h, config.img_w) * 256 / 224), antialias=True),

                v2.CenterCrop(size=(config.img_h, config.img_w)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory)



@dataclass
class BlankImageNetConfig:
    img_h: int = 224
    img_w: int = 224
    imgs_mean: Tuple = (0.485, 0.456, 0.406)
    imgs_std: Tuple = (0.229, 0.224, 0.225)


class BlankImageNetTrainDataLoader(DataLoader):
    """All images are set to zeros. Used for setting input-independent baseline."""
    def __init__(self, config: BlankImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True):
        self.config = config
        dataset = torchvision.datasets.ImageNet(
            data_dir, split='train',
            transform=v2.Compose([
                v2.Lambda(lambda img: torch.zeros((3, config.img_h, config.img_w), dtype=torch.float32)),
                v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
            ])
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin_memory)


class BlankImageNetValDataLoader(DataLoader):
    """All images are set to zeros. Used for setting input-independent baseline."""
    # Default shuffle=True since only eval partial data
    def __init__(self, config: BlankImageNetConfig, data_dir, batch_size, num_workers, shuffle=True, pin_memory=True):
        self.config = config
        dataset = torchvision.datasets.ImageNet(
            data_dir, split='val',
            transform=v2.Compose([
                v2.Lambda(lambda img: torch.zeros((3, config.img_h, config.img_w), dtype=torch.float32)),
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
    example_imgs, example_labels = next(iter(dataloader_train))
    print(f"{example_imgs.shape=}; {example_labels.shape=}")
    # Unnormalize the image for plotting
    example_img = example_imgs[0]
    example_img = example_img * torch.tensor(config.imgs_std).reshape(3, 1, 1) + torch.tensor(config.imgs_mean).reshape(3, 1, 1)
    print(f"{example_img.shape=}")
    print(f"{example_img.mean()=}, {example_img.std()=}")
    print(f"{example_img.min()=}, {example_img.max()=}")
    # plt.imshow(example_img.permute(1, 2, 0))
    # plt.show()
