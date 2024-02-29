"""
VGG16 w/ batchnorm from torchvision, for debugging and testing purposes.
"""

import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torchvision


@dataclass
class Vgg16Config:
    img_h: int = 224
    img_w: int = 224
    n_class: int = 1000


class Vgg16(nn.Module):
    def __init__(self, config: Vgg16Config) -> None:
        super().__init__()
        self.config = config

        # Batchnorm version since input img might not be normalized
        self.model = torchvision.models.vgg16_bn(weights=None, num_classes=config.n_class)

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

        # Forward the Vgg16 model itself
        # N x 3 x 224 (or 256 or 448) x 224 (or 256 or 448)
        logits = self.model(imgs)
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
        raise NotImplementedError("FUTURE: estimate TOPS for Extraction model")


    @torch.no_grad()
    def generate(self, imgs, top_k=None):
        """
        Predict on test imgs and return the top_k predictions.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()
        raise NotImplementedError("FUTURE: generate for Extraction model")
        self.train()



if __name__ == '__main__':
    # Test the model by `python -m models.vgg16` from the workspace directory
    config = Vgg16Config()
    # config = Vgg16Config(img_h=256, img_w=256)
    # config = Vgg16Config(img_h=448, img_w=448)
    model = Vgg16(config)
    print(model)
    print(f"num params: {model.get_num_params():,}")

    imgs = torch.randn(2, 3, config.img_h, config.img_w)
    targets = torch.randint(0, config.n_class, (2,))
    logits, loss = model(imgs, targets)
    print(f"logits shape: {logits.shape}")
    if loss is not None:
        print(f"loss shape: {loss.shape}")
