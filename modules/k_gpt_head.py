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
                 delta_v_dim: int,
                 delta_yaw_dim: int,
                 delta_h_dim: int = 1) -> None:
        super(KGPTHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.delta_v_dim = delta_v_dim
        self.delta_h_dim = delta_h_dim
        self.delta_yaw_dim = delta_yaw_dim
        self.patch_size = patch_size

        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_modes)
        self.to_control_action = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=(delta_v_dim + delta_h_dim + delta_yaw_dim) * num_modes)
        self.to_scale = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                 output_dim=(delta_v_dim + delta_h_dim + delta_yaw_dim) * num_modes)
        self.apply(weight_init)

    def forward(self, x_a: torch.Tensor) -> Dict[str, torch.Tensor]:
        # [agents, steps, patch, dim]
        h = x_a[..., 0]
        pi = self.to_pi(h)
        x_a = x_a[..., 1:]
        x_a = x_a.transpose(-1, -2)

        # [agents, steps, patch, modes]
        control_action = self.to_control_action(x_a)
        scale = self.to_scale(x_a)

        delta_v, delta_h, delta_yaw = control_action.split([self.delta_v_dim * self.num_modes,
                                                            self.delta_h_dim * self.num_modes,
                                                            self.delta_yaw_dim * self.num_modes], dim=-1)
        v_scale, h_scale, yaw_scale = scale.split([self.delta_v_dim * self.num_modes,
                                                   self.delta_h_dim * self.num_modes,
                                                   self.delta_yaw_dim * self.num_modes], dim=-1)
        # constrain to [-pi,pi] same to target
        delta_yaw = torch.tanh(delta_yaw) * torch.pi
        v_scale = F.elu(v_scale, alpha=1.0) + 1.0
        h_scale = F.elu(h_scale, alpha=1.0) + 1.0
        yaw_scale = 1.0 / (F.elu(yaw_scale, alpha=1.0) + 1.0 + 1e-4)

        delta_v = delta_v.reshape(*x_a.shape[:2], -1, self.num_modes, self.delta_v_dim).transpose(-3, -2)
        delta_h = delta_h.reshape(*x_a.shape[:2], -1, self.num_modes, self.delta_h_dim).transpose(-3, -2)
        delta_yaw = delta_yaw.reshape(*x_a.shape[:2], -1, self.num_modes, self.delta_yaw_dim).transpose(-3, -2)
        v_scale = v_scale.reshape(*x_a.shape[:2], -1, self.num_modes, self.delta_v_dim).transpose(-3, -2)
        h_scale = h_scale.reshape(*x_a.shape[:2], -1, self.num_modes, self.delta_h_dim).transpose(-3, -2)
        yaw_scale = yaw_scale.reshape(*x_a.shape[:2], -1, self.num_modes, self.delta_yaw_dim).transpose(-3, -2)

        return {
            "pi": pi,
            "delta_v": delta_v,
            "delta_h": delta_h,
            "delta_yaw": delta_yaw,
            "v_scale": v_scale,
            "h_scale": h_scale,
            "yaw_scale": yaw_scale,
        }
