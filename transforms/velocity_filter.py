import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class VelocityFilter(BaseTransform):

    def __init__(self,
                 window_size: int = 5) -> None:
        self.window_size = window_size

    def mean_filter(self, vel: torch.Tensor, yaw: torch.Tensor, valid_mask: torch.Tensor):
        # [N, L, D, window_size] L = steps - window_size + 1
        windows = vel.unfold(dimension=1, size=self.window_size, step=1)
        # [N, L, 1, window_size]
        valid_windows = valid_mask.unfold(dimension=1, size=self.window_size, step=1).unsqueeze(-2)

        sum_vel = (windows * valid_windows).sum(dim=-1)  # shape: [N, L, D]
        count_valid = valid_windows.sum(dim=-1)  # shape: [N, L, 1]
        avg_vel = sum_vel / torch.clamp(count_valid, min=1)

        smoothed_vel = vel
        offset = self.window_size // 2
        # index = offset ~ (steps - window_size + offset)
        smoothed_vel[:, offset:vel.size(1) - self.window_size + offset + 1] = avg_vel

        vel_mask = torch.abs(smoothed_vel) < 0.001
        smoothed_vel[vel_mask] = 0
        new_yaw = torch.atan2(smoothed_vel[..., 1], smoothed_vel[..., 0])
        new_yaw = torch.where(vel_mask.all(dim=-1), yaw, new_yaw)
        return smoothed_vel, new_yaw

    def __call__(self, data: HeteroData) -> HeteroData:
        # [agents, steps, xy-dim]
        vel = data['agent']['velocity'][..., :2]
        yaw = data['agent']['heading']
        valid_mask = data['agent']['valid_mask']
        vel, yaw = self.mean_filter(vel, yaw, valid_mask)
        data['agent']['velocity'][..., :2] = vel
        data['agent']['heading'] = yaw
        return data
