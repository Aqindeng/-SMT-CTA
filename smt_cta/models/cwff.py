import torch
import torch.nn as nn
import torch.nn.functional as F


class CWFF(nn.Module):
    """
    Change-Weighted Feature Fusion:
    - GAP over sum of streams
    - shared bottleneck
    - stream-specific logits
    - per-channel softmax across the two streams
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.shared = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_a = nn.Linear(hidden, dim)
        self.fc_b = nn.Linear(hidden, dim)

    def forward(self, fa: torch.Tensor, fb: torch.Tensor):
        # fa, fb: (B,C,H,W)
        B, C, _, _ = fa.shape
        g = (fa + fb).mean(dim=(2, 3))  # (B,C)
        z = self.shared(g)              # (B,hidden)
        ua = self.fc_a(z)               # (B,C)
        ub = self.fc_b(z)               # (B,C)

        # per-channel softmax across dates
        u = torch.stack([ua, ub], dim=1)         # (B,2,C)
        w = F.softmax(u, dim=1)                  # (B,2,C)
        wa = w[:, 0].view(B, C, 1, 1)
        wb = w[:, 1].view(B, C, 1, 1)

        e = wa * fa + wb * fb
        return e, wa, wb
