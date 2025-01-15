import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MLPLayer
from utils import weight_init


class KGPTHead(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_modes: int,
                 patch_size: int,
                 acc_dim: int,
                 delta_dim: int,
                 height_dim: int = 1) -> None:
        super(KGPTHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.acc_dim = acc_dim
        self.delta_dim = delta_dim
        self.height_dim = height_dim
        self.patch_size = patch_size

        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_modes)
        self.to_control_action = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=(acc_dim + delta_dim + height_dim) * num_modes)
        self.to_scale = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                 output_dim=(acc_dim + delta_dim + height_dim) * num_modes)
        self.apply(weight_init)

    def forward(self, x_a: torch.Tensor) -> Dict[str, torch.Tensor]:
        # [agents, steps, patch, dim]
        x_a = x_a.transpose(-1, -2)

        # [agents, steps, patch, modes]
        pi = self.to_pi(x_a)
        control_action = self.to_control_action(x_a)
        scale = self.to_scale(x_a)

        acc, delta, height = control_action.split([self.acc_dim * self.num_modes,
                                                   self.delta_dim * self.num_modes,
                                                   self.height_dim * self.num_modes], dim=-1)
        acc_scale, delta_scale, height_scale = scale.split([self.acc_dim * self.num_modes,
                                                            self.delta_dim * self.num_modes,
                                                            self.height_dim * self.num_modes], dim=-1)
        # constrain to [-pi,pi] same to target
        delta = torch.tanh(delta) * math.pi
        acc_scale = F.elu(acc_scale, alpha=1.0) + 1.0
        delta_scale = 1.0 / (F.elu(delta_scale, alpha=1.0) + 1.0 + 1e-4)
        height_scale = F.elu(height_scale, alpha=1.0) + 1.0

        acc = acc.reshape(*x_a.shape[:2], -1, self.num_modes, self.acc_dim).transpose(-3, -2)
        delta = delta.reshape(*x_a.shape[:2], -1, self.num_modes, self.delta_dim).transpose(-3, -2)
        height = height.reshape(*x_a.shape[:2], -1, self.num_modes, self.height_dim).transpose(-3, -2)
        acc_scale = acc_scale.reshape(*x_a.shape[:2], -1, self.num_modes, self.acc_dim).transpose(-3, -2)
        delta_scale = delta_scale.reshape(*x_a.shape[:2], -1, self.num_modes, self.delta_dim).transpose(-3, -2)
        height_scale = height_scale.reshape(*x_a.shape[:2], -1, self.num_modes, self.height_dim).transpose(-3, -2)

        return {
            "pi": pi,
            "acc": acc,
            "delta": delta,
            "height": height,
            "acc_scale": acc_scale,
            "delta_scale": delta_scale,
            "height_scale": height_scale
        }
