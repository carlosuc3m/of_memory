import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTModel(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 1,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck

    def forward(self, sample: torch.Tensor, enc0: torch.Tensor, enc1: torch.Tensor, enc2: torch.Tensor):
        # Forward through backbone
        return self.neck(self.trunk(sample, enc0, enc1, enc2))