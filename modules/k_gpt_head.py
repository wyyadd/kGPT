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
        # [agents, steps, patch, modes]
        pi = self.to_pi(x_a.transpose(-2, -1))

        acc = torch.tensor([], device=x_a.device)
        delta = torch.tensor([], device=x_a.device)
        acc_scale = torch.tensor([], device=x_a.device)
        delta_scale = torch.tensor([], device=x_a.device)
        height = torch.tensor([], device=x_a.device)
        height_scale = torch.tensor([], device=x_a.device)

        for i in range(self.patch_size):
            h = x_a[..., i]
            control_action = self.to_control_action(h).unsqueeze(-2)
            scale = self.to_scale(h).unsqueeze(-2)

            new_acc, new_delta, new_height = control_action.split([self.acc_dim * self.num_modes,
                                                                   self.delta_dim * self.num_modes,
                                                                   self.height_dim * self.num_modes], dim=-1)
            new_acc_scale, new_delta_scale, new_height_scale = scale.split([self.acc_dim * self.num_modes,
                                                                            self.delta_dim * self.num_modes,
                                                                            self.height_dim * self.num_modes], dim=-1)

            # constrain to [-pi,pi] same to target
            new_delta = torch.tanh(new_delta) * math.pi
            new_acc_scale = F.elu(new_acc_scale, alpha=1.0) + 1.0
            new_delta_scale = 1.0 / (F.elu(new_delta_scale, alpha=1.0) + 1.0 + 1e-4)
            new_height_scale = F.elu(new_height_scale, alpha=1.0) + 1.0

            acc = torch.cat([acc, new_acc], dim=-2)
            delta = torch.cat([delta, new_delta], dim=-2)
            acc_scale = torch.cat([acc_scale, new_acc_scale], dim=-2)
            delta_scale = torch.cat([delta_scale, new_delta_scale], dim=-2)
            height = torch.cat([height, new_height], dim=-2)
            height_scale = torch.cat([height_scale, new_height_scale], dim=-2)

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
