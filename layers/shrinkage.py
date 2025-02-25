import torch
import torch.nn as nn

from utils import weight_init


class Shrinkage(nn.Module):
    def __init__(self, hidden_dim):
        super(Shrinkage, self).__init__()
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
        alpha = torch.sigmoid(self.fc(x_abs))
        threshold = torch.mul(x_abs, alpha)
        sub = x_abs - threshold
        x = torch.mul(torch.sign(x_raw), torch.max(sub, torch.zeros_like(x)))
        return x
