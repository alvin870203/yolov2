"""
Full definition of a Darknet19 model, all of it in this single file.
Ref:
1) the official Darknet implementation:
https://github.com/pjreddie/darknet/blob/master/examples/classifier.c
https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
https://github.com/pjreddie/darknet/blob/master/cfg/darknet19_448.cfg
"""

import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Darknet19Config:
    img_h: int = 224
    img_w: int = 224
    n_class: int = 1000


class Darknet19Conv2d(nn.Module):
    """
    A Conv2d layer with a BarchNorm2d and a LeakyReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        # Darknet implementation uses bias=False when batch norm is used.
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-06, momentum=0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, 0.1, inplace=True)


class Darknet19Backbone(nn.Module):
    """
    Backbone of the Darknet19 model.
    """
    def __init__(self, config: Darknet19Config) -> None:
        super().__init__()
        self.conv1 = Darknet19Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = Darknet19Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = Darknet19Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Darknet19Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv5 = Darknet19Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.conv6 = Darknet19Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = Darknet19Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv8 = Darknet19Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.conv9 = Darknet19Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = Darknet19Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv11 = Darknet19Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = Darknet19Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv13 = Darknet19Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(2, stride=2)

        self.conv14 = Darknet19Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv15 = Darknet19Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv16 = Darknet19Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv17 = Darknet19Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv18 = Darknet19Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): size(N, 3, img_h, img_w)
        Returns:
            x (Tensor): (N, 1024, img_h / 224 * 7, img_w / 224 * 7)
            feat (Tensor): (N, 512, img_h / 224 * 14, img_w / 224 * 14)
        """
        # N x 3 x 224 (or 256 or 448) x 224 (or 256 or 448)
        x = self.conv1(x)
        # N x 32 x 224 (or 256 or 448) x 224 (or 256 or 448)
        x = self.maxpool1(x)
        # N x 32 x 112 (or 128 or 224) x 112 (or 128 or 224)

        x = self.conv2(x)
        # N x 64 x 112 (or 128 or 224) x 112 (or 128 or 224)
        x = self.maxpool2(x)
        # N x 64 x 56 (or 64 or 112) x 56 (or 64 or 112)

        x = self.conv3(x)
        # N x 128 x 56 (or 64 or 112) x 56 (or 64 or 112)
        x = self.conv4(x)
        # N x 64 x 56 (or 64 or 112) x 56 (or 64 or 112)
        x = self.conv5(x)
        # N x 128 x 56 (or 64 or 112) x 56 (or 64 or 112)
        x = self.maxpool3(x)
        # N x 128 x 28 (or 32 or 56) x 28 (or 32 or 56)

        x = self.conv6(x)
        # N x 256 x 28 (or 32 or 56) x 28 (or 32 or 56)
        x = self.conv7(x)
        # N x 128 x 28 (or 32 or 56) x 28 (or 32 or 56)
        x = self.conv8(x)
        # N x 256 x 28 (or 32 or 56) x 28 (or 32 or 56)
        x = self.maxpool4(x)
        # N x 256 x 14 (or 16 or 28) x 14 (or 16 or 28)

        x = self.conv9(x)
        # N x 512 x 14 (or 16 or 28) x 14 (or 16 or 28)
        x = self.conv10(x)
        # N x 256 x 14 (or 16 or 28) x 14 (or 16 or 28)
        x = self.conv11(x)
        # N x 512 x 14 (or 16 or 28) x 14 (or 16 or 28)
        x = self.conv12(x)
        # N x 256 x 14 (or 16 or 28) x 14 (or 16 or 28)
        x = self.conv13(x)
        feat = x
        # N x 512 x 14 (or 16 or 28) x 14 (or 16 or 28)
        x = self.maxpool5(x)
        # N x 512 x 7 (or 8 or 14) x 7 (or 8 or 14)

        x = self.conv14(x)
        # N x 1024 x 7 (or 8 or 14) x 7 (or 8 or 14)
        x = self.conv15(x)
        # N x 512 x 7 (or 8 or 14) x 7 (or 8 or 14)
        x = self.conv16(x)
        # N x 1024 x 7 (or 8 or 14) x 7 (or 8 or 14)
        x = self.conv17(x)
        # N x 512 x 7 (or 8 or 14) x 7 (or 8 or 14)
        x = self.conv18(x)
        # N x 1024 x 7 (or 8 or 14) x 7 (or 8 or 14)

        return x, feat


class Darknet19(nn.Module):
    def __init__(self, config: Darknet19Config) -> None:
        super().__init__()
        self.config = config

        self.backbone = Darknet19Backbone(config)
        self.head = nn.Sequential(
            nn.Conv2d(1024, config.n_class, bias=True, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Init all weights
        self.apply(self._init_weights)

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # TODO: zero_init_last / trunc_normal_ / head_init_scale in timm?


    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the cross entropy loss.
        Args:
            logits (Tensor): size(N, n_class)
            targets (Tensor): size(N,)
        Returns:
            loss (Tensor): size(,)
        """
        return F.cross_entropy(logits, targets)


    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            targets (Tensor): size(N, n_class)
        Returns:
            logits (Tensor): size(N,)
            loss (Tensor): size(,)
        """
        device = imgs.device

        # Forward the Darknet19 model itself
        # N x 3 x 224 (or 256 or 448) x 224 (or 256 or 448)
        x, feat = self.backbone(imgs)
        # N x 1024 x 7 (or 8 or 14) x 7 (or 8 or 14)
        logits = self.head(x)
        # N x n_class

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            loss = self._compute_loss(logits, targets)
        else:
            loss = None

        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("FUTURE: init from pretrained model")


    def configure_optimizers(self, optimizer_type, learning_rate, betas, weight_decay, device_type, use_fused):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls decay, all biases and norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        if optimizer_type == 'adamw':
            # Create AdamW optimizer and use the fused version if it is available
            if use_fused:
                fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
                use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
        elif optimizer_type == 'adam':
            # Create Adam optimizer and use the fused version if it is available
            if use_fused:
                fused_available = 'fused' in inspect.signature(torch.optim.Adam).parameters
                use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused Adam: {use_fused}")
        elif optimizer_type == 'sgd':
            # Create SGD optimizer
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=betas[0])
            print(f"using SGD")
        else:
            raise ValueError(f"unrecognized optimizer_type: {optimizer_type}")

        return optimizer


    def estimate_tops(self):
        """
        Estimate the number of TOPS and parameters in the model.
        """
        raise NotImplementedError("FUTURE: estimate TOPS for Darknet19 model")


    @torch.inference_mode()
    def generate(self, imgs, top_k=None):
        """
        Predict on test imgs and return the top_k predictions.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()
        raise NotImplementedError("FUTURE: generate for Darknet19 model")
        self.train()



if __name__ == '__main__':
    # Test the model by `python -m models.darknet19` from the workspace directory
    config = Darknet19Config()
    # config = Darknet19Config(img_h=256, img_w=256)
    # config = Darknet19Config(img_h=448, img_w=448)
    model = Darknet19(config)
    print(model)
    print(f"num params: {model.get_num_params():,}")

    imgs = torch.randn(2, 3, config.img_h, config.img_w)
    targets = torch.randint(0, config.n_class, (2,))
    logits, loss = model(imgs, targets)
    print(f"logits shape: {logits.shape}")
    if loss is not None:
        print(f"loss shape: {loss.shape}")
