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
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class SimAgentFilter(BaseTransform):

    def __init__(self,
                 max_num_agents: int) -> None:
        self.max_num_agents = max_num_agents

    def __call__(self, data: HeteroData) -> HeteroData:
        if data['agent']['num_nodes'] <= self.max_num_agents:
            data['agent']['target_idx'] = torch.arange(data['agent']['num_nodes'], dtype=torch.long,
                                                       device=data['agent']['position'].device)
            return data

        pos = data['agent']['position'][..., :2]
        av_index = data['agent']['av_index']
        valid_mask = data['agent']['valid_mask']
        target_mask = data['agent']['target_mask'].clone()

        # add sdv and track_to_predict
        agent_inds = pos.new_tensor([av_index], dtype=torch.long)
        target_mask[av_index] = False
        agent_inds = torch.cat([agent_inds, torch.where(target_mask)[0]], dim=0)
        max_num_context_agents = self.max_num_agents - agent_inds.numel()

        if max_num_context_agents > 0:
            # add context agents
            valid_nums = torch.sum(valid_mask, dim=-1)
            valid_nums[agent_inds] = 0
            for i in range(valid_mask.size(0)):
                if valid_nums[i] > 1:
                    max_d = torch.max(torch.norm(pos[i, valid_mask[i]][1:] - pos[i, valid_mask[i]][:-1], p=2, dim=-1))
                    if max_d.item() < 0.03:
                        valid_nums[i] = 0
                else:
                    valid_nums[i] = 0

            context_agent_inds = torch.multinomial(F.softmax(valid_nums.to(torch.float32), dim=-1),
                                                   num_samples=max_num_context_agents, replacement=False)
            agent_inds = torch.cat([agent_inds, context_agent_inds], dim=0)
        agent_inds = torch.unique(agent_inds)
        data['agent']['target_idx'] = agent_inds

        return data
