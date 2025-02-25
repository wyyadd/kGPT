import torch
import torch.nn as nn

from utils import weight_init


class Shrinkage(nn.Module):
    def __init__(self, hidden_dim):
        super(Shrinkage, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(self, x):
        x_raw = x
        x_abs = torch.abs(x)
        # avg
        avg = self.avg(x_abs)
        # alpha
        x = torch.sigmoid(self.fc(x_raw))
        # threshold
        x = torch.mul(avg, x)
        # sub
        x = x_abs - x
        x = torch.mul(torch.sign(x_raw), torch.max(x, torch.zeros_like(x)))
        return x
