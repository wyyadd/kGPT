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
import torch.nn as nn
from torch_cluster import knn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle


class KGPTDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_steps: int,
                 num_m2a_nbrs: int,
                 num_a2a_nbrs: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 patch_size: int) -> None:
        super(KGPTDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.num_m2a_nbrs = num_m2a_nbrs
        self.num_a2a_nbrs = num_a2a_nbrs
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        num_agent_types = 6
        num_map_types = 17
        input_dim_x_a = 5
        input_dim_x_m = 2
        input_dim_t = 5
        input_dim_r = 4
        self.patch_size = patch_size

        self.type_a_emb = nn.Embedding(num_agent_types, hidden_dim)
        self.type_m_emb = nn.Embedding(num_map_types, hidden_dim)
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_m_emb = FourierEmbedding(input_dim=input_dim_x_m, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.r_t_emb = FourierEmbedding(input_dim=input_dim_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_m2a_emb = FourierEmbedding(input_dim=input_dim_r, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)

        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers + 1)]
        )
        self.m2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.h_norm = nn.RMSNorm(hidden_dim)
        self.out_norm = nn.RMSNorm(hidden_dim)
        self.to_input = nn.Linear(hidden_dim * 2, hidden_dim)
        self.apply(weight_init)

    def forward(self,
                data: HeteroData) -> torch.Tensor:
        mask = data['agent']['valid_mask'][:, :self.num_steps].contiguous()
        pos_a = data['agent']['position'][:, :self.num_steps, :self.input_dim].contiguous()
        head_a = data['agent']['heading'][:, :self.num_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_m = data['map_point']['position'][:, :self.input_dim].contiguous()
        orient_m = data['map_point']['orientation'].contiguous()
        target = data['agent']['target_idx'].contiguous()

        vel = data['agent']['velocity'][:, :self.num_steps, :self.input_dim].contiguous()
        length = data['agent']['length'][:, :self.num_steps].contiguous()
        width = data['agent']['width'][:, :self.num_steps].contiguous()
        height = data['agent']['height'][:, :self.num_steps].contiguous()

        x_a = torch.stack(
            [torch.norm(vel[:, :, :2], p=2, dim=-1),
             head_a,
             length,
             width,
             height], dim=-1)
        type_a_emb = [self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_steps, dim=0)]
        valid_index_t = torch.where(mask.view(-1))[0]
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=type_a_emb,
                           valid_index=valid_index_t)
        x_a = x_a.view(-1, self.num_steps, self.hidden_dim)

        if self.input_dim == 2:
            x_m = data['map_point']['magnitude'].unsqueeze(-1)
        elif self.input_dim == 3:
            x_m = torch.stack([data['map_point']['magnitude'], data['map_point']['height']], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        type_m_emb = [self.type_m_emb(data['map_point']['type'].long())]
        x_m = self.x_m_emb(continuous_inputs=x_m, categorical_embs=type_m_emb)

        pos_t = pos_a[target].reshape(-1, self.input_dim)
        head_t = head_a[target].reshape(-1)
        head_vector_t = head_vector_a[target].reshape(-1, 2)
        mask_t = mask[target].unsqueeze(2) & mask[target].unsqueeze(1)
        valid_index_t = torch.where(mask[target].view(-1))[0]
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] >= edge_index_t[0]]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_pos_t[:, -1],
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        mask_t = mask[target].reshape(-1)
        if isinstance(data, Batch):
            batch_t = data['agent']['batch'][target].repeat_interleave(self.num_steps)
            batch_m = data['map_point']['batch']
        else:
            batch_t = pos_t.new_zeros(target.numel() * self.num_steps, dtype=torch.long)
            batch_m = pos_m.new_zeros(data['map_point']['num_nodes'], dtype=torch.long)
        edge_index_m2a = knn(x=pos_m[:, :2], y=pos_t[:, :2], k=self.num_m2a_nbrs, batch_x=batch_m, batch_y=batch_t)
        edge_index_m2a = edge_index_m2a[[1, 0]]
        edge_index_m2a = edge_index_m2a[:, mask_t[edge_index_m2a[1]]]
        valid_index_m = edge_index_m2a[0].unique()
        rel_pos_m2a = pos_m[edge_index_m2a[0]] - pos_t[edge_index_m2a[1]]
        rel_orient_m2a = wrap_angle(orient_m[edge_index_m2a[0]] - head_t[edge_index_m2a[1]])
        r_m2a = torch.stack(
            [torch.norm(rel_pos_m2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_m2a[1]], nbr_vector=rel_pos_m2a[:, :2]),
             rel_pos_m2a[:, -1],
             rel_orient_m2a], dim=-1)
        r_m2a = self.r_m2a_emb(continuous_inputs=r_m2a, categorical_embs=None)

        pos_s_i = pos_a[target].transpose(0, 1).reshape(-1, self.input_dim)
        head_s_i = head_a[target].transpose(0, 1).reshape(-1)
        head_vector_s_i = head_vector_a[target].transpose(0, 1).reshape(-1, 2)
        mask_s_i = mask[target].transpose(0, 1).reshape(-1)
        valid_index_s_i = torch.where(mask_s_i)[0]
        pos_s_j = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s_j = head_a.transpose(0, 1).reshape(-1)
        mask_s_j = mask.transpose(0, 1).reshape(-1)
        valid_index_s_j = torch.where(mask_s_j)[0]
        if isinstance(data, Batch):
            batch_s_i = torch.cat([data['agent']['batch'][target] + data.num_graphs * t for t in range(self.num_steps)], dim=0)
            batch_s_j = torch.cat([data['agent']['batch'] + data.num_graphs * t for t in range(self.num_steps)], dim=0)
        else:
            batch_s_i = torch.arange(self.num_steps, device=pos_a.device).repeat_interleave(target.numel())
            batch_s_j = torch.arange(self.num_steps, device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
        edge_index_a2a = knn(x=pos_s_j[:, :2], y=pos_s_i[:, :2], k=self.num_a2a_nbrs + 1, batch_x=batch_s_j, batch_y=batch_s_i)
        row_s, col_s = edge_index_a2a[1], edge_index_a2a[0]
        mask_s = row_s != col_s
        edge_index_a2a = torch.stack([row_s[mask_s], col_s[mask_s]], dim=0)
        edge_index_a2a = subgraph(subset=mask_s_j, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s_j[edge_index_a2a[0]] - pos_s_i[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s_j[edge_index_a2a[0]] - head_s_i[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s_i[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_pos_a2a[:, -1],
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        x_a_full = x_a
        x_a = x_a[target].reshape(-1, self.hidden_dim)
        for i in range(self.num_layers):
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t, valid_index=valid_index_t)
            x_a = self.m2a_attn_layers[i]((x_m, x_a), r_m2a, edge_index_m2a, valid_index=(valid_index_m, valid_index_t))
            x_a = x_a.reshape(-1, self.num_steps, self.hidden_dim)
            x_a_full[target] = x_a
            x_a_full = x_a_full.transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = x_a.transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = self.a2a_attn_layers[i]((x_a_full, x_a), r_a2a, edge_index_a2a, valid_index=(valid_index_s_j,valid_index_s_i))
            x_a = x_a.reshape(self.num_steps, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a_full = x_a_full.reshape(self.num_steps, -1, self.hidden_dim).transpose(0, 1)
        # [steps*agents, dim, patch]
        h = x_a
        x_a = x_a.new_zeros(*x_a.shape, self.patch_size)
        for i in range(self.patch_size):
            out = self.t_attn_layers[self.num_layers](h, r_t, edge_index_t, valid_index=valid_index_t)
            h = self.to_input(torch.cat([self.h_norm(h), self.out_norm(out)], dim=-1))
            x_a[..., i] = out
        return x_a.reshape(-1, self.num_steps, self.hidden_dim, self.patch_size)
