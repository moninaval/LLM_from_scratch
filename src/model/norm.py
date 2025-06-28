# src/model/norm.py

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed

class Norm(nn.Module):
    def __init__(self, hidden_size, norm_type="layernorm"):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_size)
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(hidden_size)
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")

    def forward(self, x):
        return self.norm(x)
