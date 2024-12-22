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
                 vel_dim: int,
                 yaw_rate_dim: int,
                 patch_size: int) -> None:
        super(KGPTHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.vel_dim = vel_dim
        self.yaw_dim = yaw_rate_dim
        self.patch_size = patch_size

        self.to_next_patch = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim * patch_size)
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_modes)
        self.to_control_action = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=(vel_dim + yaw_rate_dim) * num_modes)
        self.to_scale = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                 output_dim=(vel_dim + yaw_rate_dim) * num_modes)
        self.apply(weight_init)

    def forward(self, x_a: torch.Tensor) -> Dict[str, torch.Tensor]:
        pi = self.to_pi(x_a)
        next_patch = self.to_next_patch(x_a).reshape(*x_a.shape, self.patch_size)

        vel = torch.tensor([], device=x_a.device)
        yaw_rate = torch.tensor([], device=x_a.device)
        vel_scale = torch.tensor([], device=x_a.device)
        yaw_scale = torch.tensor([], device=x_a.device)

        for i in range(self.patch_size):
            h = next_patch[..., i]
            control_action = self.to_control_action(h).unsqueeze(-2)
            scale = self.to_scale(h).unsqueeze(-2)

            new_vel, new_yaw_rate = control_action.split([self.vel_dim * self.num_modes,
                                                          self.yaw_dim * self.num_modes], dim=-1)
            new_vel_scale, new_yaw_scale = scale.split([self.vel_dim * self.num_modes,
                                                        self.yaw_dim * self.num_modes], dim=-1)

            # constrain to [-pi,pi] same to target
            new_yaw_rate = torch.tanh(new_yaw_rate) * math.pi
            new_vel_scale = F.elu(new_vel_scale, alpha=1.0) + 1.0
            new_yaw_scale = 1.0 / (F.elu(new_yaw_scale, alpha=1.0) + 1.0 + 1e-4)

            vel = torch.cat([vel, new_vel], dim=-2)
            yaw_rate = torch.cat([yaw_rate, new_yaw_rate], dim=-2)
            vel_scale = torch.cat([vel_scale, new_vel_scale], dim=-2)
            yaw_scale = torch.cat([yaw_scale, new_yaw_scale], dim=-2)

        vel = vel.reshape(*x_a.shape[:-1], -1, self.num_modes, self.vel_dim).transpose(-3, -2)
        yaw_rate = yaw_rate.reshape(*x_a.shape[:-1], -1, self.num_modes, self.yaw_dim).transpose(-3, -2)
        vel_scale = vel_scale.reshape(*x_a.shape[:-1], -1, self.num_modes, self.vel_dim).transpose(-3, -2)
        yaw_scale = yaw_scale.reshape(*x_a.shape[:-1], -1, self.num_modes, self.yaw_dim).transpose(-3, -2)

        return {
            "pi": pi,
            "vel": vel,
            "yaw_rate": yaw_rate,
            "vel_scale": vel_scale,
            "yaw_scale": yaw_scale,
        }
