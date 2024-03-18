"""
Training & validation dataloaders of VOC detection dataset.
"""

from dataclasses import dataclass
import warnings
from typing import Any, Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F, InterpolationMode
from torchvision.transforms.v2.functional._utils import _FillType
from torchvision.datasets import wrap_dataset_for_transforms_v2


class Resize(v2.Resize):
    def __init__(
            self,
            letterbox: bool,
            fill:  Union[_FillType, Dict[Union[Type, str], _FillType]] = 0,
            **kwargs,  # params for v2.Resize
        ) -> None:
        super().__init__(**kwargs)
        self.size = self.size + self.size if len(self.size) == 1 else self.size
        self.letterbox = letterbox
        self.fill = fill
        self._fill = v2._utils._setup_fill_arg(fill)
        self.padding_mode = 'constant'  # only support constant padding mode for bounding boxes

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = v2._utils.query_size(flat_inputs)
        new_h, new_w = self.size
        if not self.letterbox:
            return dict(size=(new_h, new_w))
        else:  # do letterbox
            r_h, r_w = new_h / orig_h, new_w / orig_w
            r = min(r_h, r_w)
            new_unpad_h, new_unpad_w = round(orig_h * r), round(orig_w * r)
            pad_left = pad_right = pad_top = pad_bottom = 0
            if r_w < r_h:
                diff = new_h - new_unpad_h
                pad_top += (diff // 2)
                pad_bottom += (diff - pad_top)
            else:  # r_h <= r_w:
                diff = new_w - new_unpad_w
                pad_left += (diff // 2)
                pad_right += (diff - pad_left)
            padding = [pad_left, pad_top, pad_right, pad_bottom]
            return dict(size=(new_unpad_h, new_unpad_w), padding=padding)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = self._call_kernel(F.resize, inpt, size=params['size'],
                                 interpolation=self.interpolation, antialias=self.antialias)
        if self.letterbox:
            fill = v2._utils._get_fill(self._fill, type(inpt))
            inpt = self._call_kernel(F.pad, inpt, padding=params["padding"], fill=fill, padding_mode=self.padding_mode)
        return inpt


class Voc2Yolov2(nn.Module):
    """
    x (Tensor): size(3, img_h, img_w), RGB, 0~255
    y_yolov2 (Tensor): size(n_grid_h, n_grid_w, n_box_per_cell, 6)  # TODO: Only consider n_box_per_cell ?
        targets[j, k, l, 0:4] is is the box coordinates for the l-th box in the j,k-th grid cell, 0.0~1.0
            targets[j, k, l, 0:2] is the cx,cy relative to top-left corner of the j,k-th grid cell
                               and normalized by the grid cell width,height,
                               i.e., ground truth of sigmoid(t_x), sigmoid(t_y) in the paper
            targets[j, k, l, 2:4] is the w,h normalized by the grid cell width,height,
                               i.e., ground truth of p_w * exp(t_w), p_h * exp(t_h) in the paper
        targets[j, k, l, 4] is the is_obj for the l-th box in the j,k-th grid cell, 1.0 if there is an object, 0.0 otherwise
        targets[j, k, l, 5] is the class index for the l-th box in the j,k-th grid cell, 0.0~float(n_class-1)
    y_voc['boxes'] (tv_tensors.BoundingBoxes): size(n_box=, 4), format='XYXY'
        y_voc['boxes'] is not normalized
    """
    def __init__(self, n_box_per_cell):
        super().__init__()
        self.n_box_per_cell = n_box_per_cell

    # The following three functions are for cumcount
    # Ref: https://stackoverflow.com/questions/40602269/how-to-use-numpy-to-get-the-cumulative-count-by-unique-values-in-linear-time
    def _dfill(self, a):
        n = a.size
        b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
        return np.arange(n)[b[:-1]].repeat(np.diff(b))

    def _argunsort(self, s):
        n = s.size
        u = np.empty(n, dtype=np.int64)
        u[s] = np.arange(n)
        return u

    def _cumcount(self, a):
        n = a.size
        s = a.argsort(kind='mergesort')
        i = self._argunsort(s)
        b = a[s]
        return (np.arange(n) - self._dfill(b))[i]

    def forward(self, x, y_voc):
        img_h, img_w = x.shape[-2:]
        n_grid_h, n_grid_w = img_h * 13 // 416, img_w * 13 // 416
        boxes_yolov2 = y_voc['boxes'].clone()
        # Transform the bounding boxes from xyxy to cxcywh, boxes_yolov2.dtype is int64
        boxes_yolov2 = box_convert(boxes_yolov2, in_fmt='xyxy', out_fmt='cxcywh')
        # Normalized the bounding boxes by the image width,height, boxes_yolov2.dtype become float32
        boxes_yolov2[:, [0, 2]] /= img_w
        boxes_yolov2[:, [1, 3]] /= img_h
        # Normalized the bounding boxes by the grid cell width,height
        boxes_yolov2[:, [0, 2]] *= n_grid_w
        boxes_yolov2[:, [1, 3]] *= n_grid_h
        # Randomly shuffle the bounding boxes and labels, since only n_box_per_cell object can be assigned to a grid cell
        idx = torch.randperm(len(boxes_yolov2))
        y_voc['boxes'] = y_voc['boxes'][idx]
        boxes_yolov2 = boxes_yolov2[idx]
        y_voc['labels'] = y_voc['labels'][idx] - 1  # remove background class
        y_yolov2 = torch.zeros((n_grid_h, n_grid_w, self.n_box_per_cell, 6), dtype=torch.float32)
        cx_yolov2, cy_yolov2, w_yolov2, h_yolov2 = torch.unbind(boxes_yolov2, dim=1)
        grid_x = torch.clamp_max(torch.floor(cx_yolov2), (n_grid_w - 1)).to(torch.int64)
        grid_y = torch.clamp_max(torch.floor(cy_yolov2), (n_grid_h - 1)).to(torch.int64)
        _, obj_grid_idx = torch.unique(torch.stack((grid_x, grid_y)), return_inverse=True, dim=1)
        grid_idx_box = self._cumcount(obj_grid_idx.numpy())
        grid_idx_box = np.minimum(grid_idx_box, (self.n_box_per_cell - 1))
        y_yolov2[grid_y, grid_x, grid_idx_box, 4] = 1.0  # set the is_obj to 1.0
        y_yolov2[grid_y, grid_x, grid_idx_box, 5] = y_voc['labels'].to(torch.float32)  # set the class index to label
        # Set the box coordinates, all normalized by the grid size
        y_yolov2[grid_y, grid_x, grid_idx_box, 0] = cx_yolov2 - grid_x
        y_yolov2[grid_y, grid_x, grid_idx_box, 1] = cy_yolov2 - grid_y
        y_yolov2[grid_y, grid_x, grid_idx_box, 2] = w_yolov2
        y_yolov2[grid_y, grid_x, grid_idx_box, 3] = h_yolov2
        return x, y_yolov2, y_voc


@dataclass
class VocConfig:
    img_h: int = 416
    img_w: int = 416
    multiscale_min_sizes: Tuple[int, ...] = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)  # square edge
    n_box_per_cell: int = 5
    perspective: float = 0.015
    crop_scale: float = 0.8
    ratio_min: float = 0.5
    ratio_max: float = 2.0
    degrees: float = 0.5  # unit: deg
    translate: float = 0.1
    scale: float = 0.25
    shear: float = 0.5  # unit: deg
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.7
    hue: float = 0.015
    flip_p: float = 0.5
    letterbox: bool = True
    imgs_mean: Tuple = (0.485, 0.456, 0.406)
    imgs_std: Tuple = (0.229, 0.224, 0.225)
    fill: Tuple = (123.0, 117.0, 104.0)


class VocCollateFn:
    def __init__(self, multiscale: bool, config: VocConfig):
        self.multiscale = multiscale
        self.config = config
        self.batched_multi_scale_transform = v2.RandomShortestSize(min_size=config.multiscale_min_sizes, antialias=True)
        self.voc2yolov2_transform = Voc2Yolov2(n_box_per_cell=config.n_box_per_cell)

    def __call__(self, batch):
        xs, ys, y_supps = [], [], []
        for x, y_supp in batch:
            xs.append(x)
            y_supps.append(y_supp)
        if self.multiscale:
            xs, y_supps = self.batched_multi_scale_transform(xs, y_supps)
        for idx_img, (x, y_supp) in enumerate(zip(xs, y_supps)):
            x, y, y_supp = self.voc2yolov2_transform(x, y_supp)
            xs[idx_img] = x
            ys.append(y)
            y_supps[idx_img] = y_supp
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs, ys, y_supps


class VocTrainDataLoader(DataLoader):
    def __init__(self, config: VocConfig, data_dir, batch_size, num_workers, collate_fn, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first two images as the entire dataset
        self.config = config
        self.fill = {tv_tensors.Image: config.fill, "others": 0}
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(brightness=config.brightness, contrast=config.contrast,
                           saturation=config.saturation, hue=config.hue),
            v2.RandomIoUCrop(min_scale=config.crop_scale, max_scale=1.0,
                             min_aspect_ratio=config.ratio_min, max_aspect_ratio=config.ratio_max),
            Resize(size=config.multiscale_min_sizes[-1], letterbox=config.letterbox, fill=self.fill, antialias=True),
            v2.RandomPerspective(distortion_scale=config.perspective, fill=self.fill),
            v2.RandomAffine(degrees=config.degrees, translate=(config.translate, config.translate),
                            scale=(1 - config.scale, 1 + config.scale),
                            shear=(-config.shear, config.shear, -config.shear, config.shear), fill=self.fill,
                            interpolation=InterpolationMode.BILINEAR),
            v2.RandomHorizontalFlip(p=config.flip_p),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
        ])
        dataset_2007_trainval = torchvision.datasets.VOCDetection(root=data_dir, year='2007', image_set='trainval',
                                                               download=False, transforms=transforms)
        dataset_2007_trainval_v2 = wrap_dataset_for_transforms_v2(dataset_2007_trainval, target_keys=['boxes', 'labels'])
        dataset_2012_trainval = torchvision.datasets.VOCDetection(root=data_dir, year='2012', image_set='trainval',
                                                                  download=False, transforms=transforms)
        dataset_2012_trainval_v2 = wrap_dataset_for_transforms_v2(dataset_2012_trainval, target_keys=['boxes', 'labels'])
        dataset = ConcatDataset([dataset_2007_trainval_v2, dataset_2012_trainval_v2])
        if nano:
            dataset = Subset(dataset, indices=range(2))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)


class VocValDataLoader(DataLoader):
    # Default shuffle=True since only eval partial data
    def __init__(self, config: VocConfig, data_dir, batch_size, num_workers, collate_fn, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first two images as the entire dataset
        self.config = config
        self.fill = {tv_tensors.Image: config.fill, "others": 0}
        transforms = v2.Compose([
            v2.ToImage(),
            Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=self.fill, antialias=True),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
        ])
        dataset_2007_test = torchvision.datasets.VOCDetection(root=data_dir, year='2007', image_set='test',
                                                              download=False, transforms=transforms)
        dataset = wrap_dataset_for_transforms_v2(dataset_2007_test, target_keys=['boxes', 'labels'])
        if nano:
            dataset = Subset(dataset, indices=range(2))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)


@dataclass
class BlankVocConfig:
    img_h: int = 224
    img_w: int = 224
    multiscale_min_sizes: Tuple[int, ...] = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)  # square edge
    n_box_per_cell: int = 5
    letterbox: bool = True
    imgs_mean: Tuple = (0.485, 0.456, 0.406)
    imgs_std: Tuple = (0.229, 0.224, 0.225)
    fill: Tuple = (123.0, 117.0, 104.0)


class BlankVocTrainDataLoader(DataLoader):
    """All images are set to zeros. Used for setting input-independent baseline."""
    def __init__(self, config: BlankVocConfig, data_dir, batch_size, num_workers, collate_fn, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first two images as the entire dataset
        self.config = config
        self.fill = {tv_tensors.Image: config.fill, "others": 0}
        transforms = v2.Compose([
            v2.ToImage(),
            v2.Lambda(
                lambda inp: tv_tensors.wrap(
                    torch.tensor(config.fill, dtype=inp.dtype, device=inp.device).view(3, 1, 1).expand(inp.shape),
                    like=inp)
                if isinstance(inp, tv_tensors.Image) else inp),
            Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=self.fill, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
        ])
        dataset_2007_trainval = torchvision.datasets.VOCDetection(root=data_dir, year='2007', image_set='trainval',
                                                                  download=False, transforms=transforms)
        dataset_2007_trainval_v2 = wrap_dataset_for_transforms_v2(dataset_2007_trainval, target_keys=['boxes', 'labels'])
        dataset_2012_trainval = torchvision.datasets.VOCDetection(root=data_dir, year='2012', image_set='trainval',
                                                                  download=False, transforms=transforms)
        dataset_2012_trainval_v2 = wrap_dataset_for_transforms_v2(dataset_2012_trainval, target_keys=['boxes', 'labels'])
        dataset = ConcatDataset([dataset_2007_trainval_v2, dataset_2012_trainval_v2])
        if nano:
            dataset = Subset(dataset, indices=range(2))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)


class BlankVocValDataLoader(DataLoader):
    """All images are set to zeros. Used for setting input-independent baseline."""
    # Default shuffle=True since only eval partial data
    def __init__(self, config: VocConfig, data_dir, batch_size, num_workers, collate_fn, shuffle=True, pin_memory=True,
                 nano=False):  # if True: use only the first two images as the entire dataset
        self.config = config
        self.fill = {tv_tensors.Image: config.fill, "others": 0}
        transforms = v2.Compose([
            v2.ToImage(),
            v2.Lambda(
                lambda inp: tv_tensors.wrap(
                    torch.tensor(config.fill, dtype=inp.dtype, device=inp.device).view(3, 1, 1).expand(inp.shape),
                    like=inp)
                if isinstance(inp, tv_tensors.Image) else inp),
            Resize(size=(config.img_h, config.img_w), letterbox=config.letterbox, fill=self.fill, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.imgs_mean, std=config.imgs_std),
        ])
        dataset_2007_test = torchvision.datasets.VOCDetection(root=data_dir, year='2007', image_set='test',
                                                              download=False, transforms=transforms)
        dataset = wrap_dataset_for_transforms_v2(dataset_2007_test, target_keys=['boxes', 'labels'])
        if nano:
            dataset = Subset(dataset, indices=range(2))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                         num_workers=num_workers, pin_memory=pin_memory)



if __name__ == '__main__':
    # Test the dataloaders by `python -m dataloaders.voc` from the workspace directory
    import matplotlib.pyplot as plt
    data_dir = 'data/voc'
    config = VocConfig()
    collate_fn_train = VocCollateFn(multiscale=True, config=config)
    collate_fn_val = VocCollateFn(multiscale=False, config=config)
    dataloader_train = VocTrainDataLoader(config, data_dir, batch_size=32, num_workers=4, collate_fn=collate_fn_train)
    dataloader_val = VocValDataLoader(config, data_dir, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    print(f"{len(dataloader_train)=}")
    print(f"{len(dataloader_val)=}")
    example_x, example_y, example_y_supp = next(iter(dataloader_train))
    print(f"{example_x.shape=}; {example_y.shape=}")
    # Unnormalize the image for plotting
    example_img = example_x[0]
    example_img = example_img * torch.tensor(config.imgs_std).reshape(3, 1, 1) + torch.tensor(config.imgs_mean).reshape(3, 1, 1)
    print(f"{example_img.shape=}")
    print(f"{example_img.mean(dim=(1,2))=}, {example_img.std(dim=(1,2))=}")
    print(f"{example_img.min()=}, {example_img.max()=}")
    # plt.imshow(example_img.permute(1, 2, 0))
    # plt.show()
