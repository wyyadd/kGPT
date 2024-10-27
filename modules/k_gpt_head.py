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
                 acceleration_dim: int,
                 yaw_rate_dim: int) -> None:
        super(KGPTHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.acc_dim = acceleration_dim
        self.yaw_dim = yaw_rate_dim
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_modes)
        self.to_control_action = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=(acceleration_dim + yaw_rate_dim) * num_modes)
        self.to_scale = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                 output_dim=(acceleration_dim + yaw_rate_dim) * num_modes)
        self.apply(weight_init)

    def forward(self, x_a: torch.Tensor) -> Dict[str, torch.Tensor]:
        pi = self.to_pi(x_a)

        control_action = self.to_control_action(x_a)
        scale = self.to_scale(x_a)

        acceleration, yaw_rate = control_action.split([self.acc_dim * self.num_modes,
                                                       self.yaw_dim * self.num_modes], dim=-1)
        acc_scale, yaw_scale = scale.split([self.acc_dim * self.num_modes,
                                            self.yaw_dim * self.num_modes], dim=-1)

        # constrain to [-10pi,10pi] same to target
        yaw_rate = torch.tanh(yaw_rate) * math.pi * 10
        acc_scale = F.elu(acc_scale, alpha=1.0) + 1.0
        yaw_scale = 1.0 / (F.elu(yaw_scale, alpha=1.0) + 1.0 + 1e-4)

        acceleration = acceleration.reshape(*x_a.shape[:-1], self.num_modes, self.acc_dim)
        yaw_rate = yaw_rate.reshape(*x_a.shape[:-1], self.num_modes, self.yaw_dim)
        acc_scale = acc_scale.reshape(*x_a.shape[:-1], self.num_modes, self.acc_dim)
        yaw_scale = yaw_scale.reshape(*x_a.shape[:-1], self.num_modes, self.yaw_dim)

        return {
            "pi": pi,
            "acceleration": acceleration,
            "yaw_rate": yaw_rate,
            "acc_scale": acc_scale,
            "yaw_scale": yaw_scale,
        }
