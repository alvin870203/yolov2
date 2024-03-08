"""
Training & validation dataloaders of ImageNet2012 classification dataset.
"""

from dataclasses import dataclass
import warnings
from typing import Any, Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F, InterpolationMode
from torchvision.transforms.v2.functional._utils import _FillType


class RandomResize(v2.RandomResize):
    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.85, 1.05),
        scale_p: float = 0.9,
        ratio: Tuple[float, float] = (0.9, 1.11),
        ratio_p: float = 0.9,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> None:
        super().__init__(None, None, interpolation, antialias)
        self.size = v2._utils._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        scale = cast(Tuple[float, float], scale)
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        ratio = cast(Tuple[float, float], ratio)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        self.scale = scale
        self.ratio = ratio
        self.scale_p = scale_p
        self.ratio_p = ratio_p
        self._log_ratio = torch.log(torch.tensor(ratio))
        self._min_factor = min((1 / self.ratio[0], self.ratio[0],
                                1 / self.ratio[1], self.ratio[1])) / math.sqrt(max(*self.scale))
        if self._min_factor < 0.5:
            raise ValueError("The minimum factor is less than 0.5, please decrease scale_max or adjust ratio.")


    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        img_h, img_w = img_size = v2._utils.query_size(flat_inputs)
        target_h, target_w = self.size
        ratio_h = img_h / target_h
        ratio_w = img_w / target_w
        ratio = min(ratio_h, ratio_w)

        if torch.rand(1) < self.scale_p:
            ratio_factor = torch.empty(1).uniform_(*self.scale)
            ratio_factor = 1.0 / math.sqrt(ratio_factor)
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1.0, 1.0)

        if torch.rand(1) < self.ratio_p:
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(
                    self._log_ratio[0],
                    self._log_ratio[1],
                )
            ).item()
            aspect_factor = math.sqrt(aspect_ratio)
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)

        size = [round(x * f / ratio) for x, f in zip(img_size, ratio_factor)]
        return dict(size=size)


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
                v2.RandomResizedCrop(size=max(config.img_h, config.img_w),
                                     scale=(config.scale_min ** 2, config.scale_max ** 2),
                                     ratio=(config.ratio_min, config.ratio_max), antialias=True),
                v2.RandomHorizontalFlip(p=config.flip_p),
                v2.TrivialAugmentWide(),
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
