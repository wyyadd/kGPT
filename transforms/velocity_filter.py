import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class VelocityFilter(BaseTransform):

    def __init__(self,
                 window_size: int = 5) -> None:
        self.window_size = window_size

    def mean_filter(self, vel: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # [N, L, window_size, D] L = steps - window_size + 1
        windows = vel.unfold(dimension=1, size=self.window_size, step=1)
        # [N, L, window_size, 1]
        valid_windows = valid_mask.unfold(dimension=1, size=self.window_size, step=1).unsqueeze(-1)

        sum_vel = (windows * valid_windows).sum(dim=2)  # shape: [N, L, D]
        count_valid = valid_windows.sum(dim=2)  # shape: [N, L, 1]
        avg_vel = sum_vel / torch.clamp(count_valid, min=1)

        smoothed_vel = vel
        offset = self.window_size // 2
        # index = offset ~ (steps - window_size + offset)
        smoothed_vel[:, offset:vel.size(0) - self.window_size + offset + 1] = avg_vel
        return smoothed_vel

    def __call__(self, data: HeteroData) -> HeteroData:
        # [agents, steps, xy-dim]
        vel = data['agent']['velocity'][..., :2]
        valid_mask = data['agent']['valid_mask']
        data['agent']['velocity'][..., :2] = self.mean_filter(vel, valid_mask)
        return data
