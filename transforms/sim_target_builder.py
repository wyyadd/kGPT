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

    def __init__(self, patch) -> None:
        super().__init__()
        self.patch = patch

    def __call__(self, data: HeteroData) -> HeteroData:
        target_idx = data['agent']['target_idx']
        pos = data['agent']['position'][target_idx]
        head = data['agent']['heading'][target_idx]
        vel = data['agent']['velocity'][target_idx]
        cos, sin = head.cos(), head.sin()
        rot_mat = torch.stack([torch.stack([cos, -sin], dim=-1),
                               torch.stack([sin, cos], dim=-1)], dim=-2)

        # num_agent, steps, patch_size, 4
        data['agent']['target'] = pos.new_zeros(target_idx.numel(), pos.size(-2), self.patch, 4)
        delta_v = ((vel[:, 1:, :2] - vel[:, :- 1, :2]).unsqueeze(-2) @ rot_mat[:, :- 1]).squeeze(-2)
        delta_h = pos[:, 1:, 2] - pos[:, :- 1, 2]
        delta_yaw = wrap_angle(head[:, 1:] - head[:, : - 1])

        for t in range(self.patch):
            #  target: 0-2: delta_v (acc), 2: delta_h, 3: delta_yaw
            data['agent']['target'][:, :-t - 1, t, :2] = delta_v[:, t:]
            data['agent']['target'][:, :-t - 1, t, 2] = delta_h[:, t:]
            data['agent']['target'][:, :-t - 1, t, 3] = delta_yaw[:, t:]
        return data
