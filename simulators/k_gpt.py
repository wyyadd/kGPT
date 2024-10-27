import os
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from modules import KGPTDecoder
from modules.k_gpt_head import KGPTHead
from utils import unbatch
from utils import wrap_angle


class KGPT(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 pos_dim: int,
                 vel_dim: int,
                 theta_dim: int,
                 num_steps: int,
                 num_init_steps: int,
                 num_rollout_steps: int,
                 num_m2a_nbrs: int,
                 num_a2a_nbrs: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 bin_nums: int,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 **kwargs) -> None:
        super(KGPT, self).__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.vel_dim = vel_dim
        self.theta_dim = theta_dim
        self.acceleration_dim = 2
        self.yaw_rate_dim = 1
        self.num_modes = 16
        self.num_steps = num_steps
        self.num_init_steps = num_init_steps
        self.num_rollout_steps = num_rollout_steps
        self.num_m2a_nbrs = num_m2a_nbrs
        self.num_a2a_nbrs = num_a2a_nbrs
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.bin_nums = bin_nums
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir

        self.decoder = KGPTDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            num_m2a_nbrs=num_m2a_nbrs,
            num_a2a_nbrs=num_a2a_nbrs,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.head = KGPTHead(
            hidden_dim=hidden_dim,
            acceleration_dim=self.acceleration_dim,
            yaw_rate_dim=self.yaw_rate_dim,
            num_modes=self.num_modes,
        )

        self.loss = MixtureNLLLoss(component_distribution=['laplace'] * self.acceleration_dim +
                                                          ['von_mises'] * self.yaw_rate_dim,
                                   reduction='none')
        self.test_predictions = dict()

    def forward(self, data: HeteroData):
        x_a = self.decoder(data)
        return self.head(x_a)

    def training_step(self,
                      data,
                      batch_idx):
        valid_mask = data['agent']['valid_mask'][:, :self.num_steps]
        predict_mask = torch.zeros_like(valid_mask)
        predict_mask[:, :-1] = valid_mask[:, :-1] & valid_mask[:, 1:]
        # Due to the limitation of the Waymo Open Sim Agents Challenge, we fix the bbox size
        data['agent']['length'] = data['agent']['length'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)
        data['agent']['width'] = data['agent']['width'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)
        data['agent']['height'] = data['agent']['height'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)
        pred = self(data)
        pi = pred['pi']
        pred = torch.cat(
            [pred['acceleration'],
             pred['yaw_rate'],
             pred['acc_scale'],
             pred['yaw_scale']], dim=-1)
        target = data['agent']['target'][:, :self.num_steps]

        loss = self.loss(pred=pred.reshape(-1, self.num_modes, pred.size(-1)),
                         target=target.reshape(-1, target.size(-1)),
                         prob=pi.reshape(-1, pi.size(-1)),
                         mask=predict_mask.reshape(-1)).reshape(-1, self.num_steps)

        # loss = loss / predict_mask.sum(dim=-1).clamp(min=1)
        loss = loss[predict_mask].mean()
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        valid_mask = data['agent']['valid_mask'][:, :self.num_steps]
        predict_mask = torch.zeros_like(valid_mask)
        predict_mask[:, :-1] = valid_mask[:, :-1] & valid_mask[:, 1:]
        # Due to the limitation of the Waymo Open Sim Agents Challenge, we fix the bbox size
        data['agent']['length'] = data['agent']['length'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)
        data['agent']['width'] = data['agent']['width'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)
        data['agent']['height'] = data['agent']['height'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)
        pred = self(data)
        pi = pred['pi']
        pred = torch.cat(
            [pred['acceleration'],
             pred['yaw_rate'],
             pred['acc_scale'],
             pred['yaw_scale']], dim=-1)
        target = data['agent']['target'][:, :self.num_steps]

        loss = self.loss(pred=pred.reshape(-1, self.num_modes, pred.size(-1)),
                         target=target.reshape(-1, target.size(-1)),
                         prob=pi.reshape(-1, pi.size(-1)),
                         mask=predict_mask.reshape(-1)).reshape(-1, self.num_steps)
        loss = loss[predict_mask].mean()
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

    def test_step(self,
                  data,
                  batch_idx):
        # We only simulate the agents that are valid at the init step
        eval_mask = data['agent']['valid_mask'][:, self.num_init_steps - 1]
        data['agent']['valid_mask'][:, self.num_init_steps:] = False
        data['agent']['valid_mask'][eval_mask, self.num_init_steps:] = True
        # Due to the limitation of the Waymo Open Sim Agents Challenge, we fix the bbox size
        data['agent']['length'] = data['agent']['length'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)
        data['agent']['width'] = data['agent']['width'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)
        data['agent']['height'] = data['agent']['height'][:, [self.num_init_steps - 1]].repeat(1, self.num_steps)

        traj_eval = []
        num_action_steps = 10
        for k in range(32):
            for t in range(self.num_rollout_steps // num_action_steps):
                pred = self(data)
                pi = pred['pi'][:, self.num_init_steps + t * num_action_steps - 1]  # [A, K]
                sample_inds = torch.multinomial(F.softmax(pi, dim=-1), num_samples=1, replacement=True).squeeze(-1)
                # sample_inds = top_p_sampling(pi, 0.95)
                pos = pred['pos_loc'][torch.arange(pi.size(0)), self.num_init_steps + t * num_action_steps - 1,
                      sample_inds, :num_action_steps, :self.pos_dim]
                vel = pred['vel_loc'][torch.arange(pi.size(0)), self.num_init_steps + t * num_action_steps - 1,
                      sample_inds, :num_action_steps, :self.vel_dim]
                theta = pred['theta_loc'][torch.arange(pi.size(0)), self.num_init_steps + t * num_action_steps - 1,
                        sample_inds, :num_action_steps, :self.theta_dim]
                # Transform to global coordinates
                current_theta = data['agent']['heading'][:, self.num_init_steps + t * num_action_steps - 1]
                cos, sin = current_theta.cos(), current_theta.sin()
                rot_mat = torch.stack([torch.stack([cos, sin], dim=-1),
                                       torch.stack([-sin, cos], dim=-1)],
                                      dim=-2)
                pos[..., :2] = pos[..., :2] @ rot_mat
                pos[..., :2] += data['agent']['position'][:, [self.num_init_steps + t * num_action_steps - 1], :2]
                if self.pos_dim == 3:
                    pos[..., 2] += data['agent']['position'][:, [self.num_init_steps + t * num_action_steps - 1], 2]
                else:
                    pos[..., 2] = data['agent']['position'][:, [self.num_init_steps - 1], 2]
                data['agent']['position'][:,
                self.num_init_steps + t * num_action_steps: self.num_init_steps + (t + 1) * num_action_steps,
                :self.pos_dim] = pos[..., :self.pos_dim]
                vel[..., :2] = vel[..., :2] @ rot_mat
                data['agent']['velocity'][:,
                self.num_init_steps + t * num_action_steps: self.num_init_steps + (t + 1) * num_action_steps,
                :self.vel_dim] = vel[..., :self.vel_dim]
                data['agent']['heading'][:, self.num_init_steps + t * num_action_steps: self.num_init_steps + (
                        t + 1) * num_action_steps] = wrap_angle(current_theta.unsqueeze(-1) + theta[..., 0])
            traj_eval.append(torch.cat([data['agent']['position'][eval_mask, self.num_init_steps:, :3],
                                        data['agent']['heading'][eval_mask, self.num_init_steps:].unsqueeze(-1)],
                                       dim=-1))
        traj_eval = torch.stack(traj_eval, dim=1)
        if isinstance(data, Batch):
            eval_id_unbatch = unbatch(src=data['agent']['id'][eval_mask], batch=data['agent']['batch'][eval_mask],
                                      dim=0)
            traj_eval_unbatch = unbatch(src=traj_eval, batch=data['agent']['batch'][eval_mask], dim=0)
            for i in range(data.num_graphs):
                track_predictions = dict()
                for j in range(len(eval_id_unbatch[i])):
                    track_predictions[eval_id_unbatch[i][j].item()] = traj_eval_unbatch[i][j].cpu().numpy()
                self.test_predictions[data['scenario_id'][i]] = track_predictions
        else:
            eval_id = data['agent']['id'][eval_mask]
            track_predictions = dict()
            for i in range(len(eval_id)):
                track_predictions[eval_id[i].item()] = traj_eval[i].cpu().numpy()
            self.test_predictions[data['scenario_id']] = track_predictions

    def on_test_end(self):
        try:
            with open(os.path.join(self.submission_dir, f'{self.global_rank}.pkl'), 'wb') as handle:
                pickle.dump(self.test_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'Error saving predictions: {e}')

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('KGPT')
        parser.add_argument('--input_dim', type=int, default=3)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--pos_dim', type=int, default=3)
        parser.add_argument('--vel_dim', type=int, default=2)
        parser.add_argument('--theta_dim', type=int, default=1)
        parser.add_argument('--num_steps', type=int, default=91)
        parser.add_argument('--num_init_steps', type=int, default=11)
        parser.add_argument('--num_rollout_steps', type=int, default=80)
        parser.add_argument('--num_m2a_nbrs', type=int, default=32)
        parser.add_argument('--num_a2a_nbrs', type=int, default=32)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--bin_nums', type=int, default=64)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=0.1)
        parser.add_argument('--T_max', type=int, default=30)
        parser.add_argument('--submission_dir', type=str, default='./')
        return parent_parser
