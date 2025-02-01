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


class SimAgentFilter(BaseTransform):

    def __init__(self,
                 max_num_agents: int) -> None:
        self.max_num_agents = max_num_agents

    def __call__(self, data: HeteroData) -> HeteroData:
        pos = data['agent']['position'][..., :2]
        av_index = data['agent']['av_index']
        target_mask = data['agent']['target_mask']
        # add sdv and track_to_predict
        agent_inds = pos.new_tensor([av_index], dtype=torch.long)
        agent_inds = torch.cat([agent_inds, torch.where(target_mask)[0]], dim=0)
        agent_inds = torch.unique(agent_inds)
        data['agent']['target_idx'] = agent_inds
        return data
