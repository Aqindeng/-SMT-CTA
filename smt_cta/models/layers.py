import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth (per sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for 2D feature maps (B,C,H,W)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B,C,H,W)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DWSeparableConv2d(nn.Module):
    """Depthwise-separable conv: DW conv + PW conv."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


class MLP2d(nn.Module):
    """Pointwise MLP implemented via 1x1 convolutions."""
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def unfold_neighbors(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Unfold local neighborhoods for each spatial location.
    Input:  x (B,C,H,W)
    Output: (B, H*W, k*k, C)
    """
    B, C, H, W = x.shape
    pad = k // 2
    patches = F.unfold(x, kernel_size=k, padding=pad)  # (B, C*k*k, H*W)
    patches = patches.view(B, C, k * k, H * W).permute(0, 3, 2, 1).contiguous()
    return patches  # (B, HW, kk, C)


def local_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Local MHA for unfolded neighborhood attention.
    q: (B, HW, Cq)
    k: (B, HW, KK, Ck)
    v: (B, HW, KK, Cv)
    Assumes Cq == Ck == Cv and divisible by num_heads.
    Returns: (B, HW, Cq)
    """
    B, HW, C = q.shape
    KK = k.shape[2]
    assert C % num_heads == 0
    d = C // num_heads

    q = q.view(B, HW, num_heads, d)                      # (B,HW,h,d)
    k = k.view(B, HW, KK, num_heads, d)                  # (B,HW,KK,h,d)
    v = v.view(B, HW, KK, num_heads, d)                  # (B,HW,KK,h,d)

    attn = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(d)  # (B,HW,KK,h)
    attn = attn.softmax(dim=2)

    out = (attn.unsqueeze(-1) * v).sum(dim=2)            # (B,HW,h,d)
    out = out.view(B, HW, C)
    return out
