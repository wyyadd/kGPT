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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


class SimTargetBuilder(BaseTransform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: HeteroData) -> HeteroData:
        pos = data['agent']['position']
        head = data['agent']['heading']
        vel = data['agent']['velocity']
        cos, sin = head.cos(), head.sin()
        rot_mat = torch.stack([torch.stack([cos, -sin], dim=-1),
                               torch.stack([sin, cos], dim=-1)],
                              dim=-2)

        #  target: 0-3: velocity, 3: yaw_rate, scalar, no need rotate
        data['agent']['target'] = pos.new_zeros(data['agent']['num_nodes'], pos.size(-2), 4)
        data['agent']['target'][:, :-1, :2] = (vel[:, 1:, :2].unsqueeze(-2) @ rot_mat[:, :-1]).squeeze(-2)
        data['agent']['target'][:, :-1, 2] = vel[:, 1:, 2]
        data['agent']['target'][:, :-1, 3] = wrap_angle(head[:, 1:] - head[:, :-1])
        return data
