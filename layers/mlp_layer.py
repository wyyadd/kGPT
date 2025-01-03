# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
import torch.nn as nn

from utils import weight_init


class MLPLayer(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        super(MLPLayer, self).__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self,
                x: torch.Tensor,
                valid_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if valid_index is None:
            return self.mlp(x)
        else:
            return x.new_zeros((x.numel() // x.size(-1), self.output_dim)).scatter_(
                dim=0,
                index=valid_index.unsqueeze(-1).repeat(1, self.output_dim),
                src=self.mlp(x.reshape(-1, x.size(-1))[valid_index]),
            ).view(*x.size()[:-1], self.output_dim)
