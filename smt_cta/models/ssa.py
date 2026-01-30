import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LayerNorm2d, MLP2d, DropPath, DWSeparableConv2d, unfold_neighbors, local_mha


class SSA(nn.Module):
    """
    Shuffled Sparse Attention (SSA):
    - Predict offsets Δ via a lightweight offset head
    - Sample rho^2 disjoint sub-lattices using strided grids + offsets (grid_sample)
    - Apply local attention within each sparse group
    - Shuffle/scatter outputs back to the dense lattice
    """
    def __init__(self, dim: int, num_heads: int, window_size: int, rho: int, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.rho = int(rho)

        # offset head (predict per-location 2D offsets)
        self.off = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, 2, 1)
        )
        self.offset_scale = 2.0  # clamp range (pixels) via tanh scaling

        self.q = DWSeparableConv2d(dim, dim, 3, 1, 1)
        self.k = DWSeparableConv2d(dim, dim, 3, 1, 1)
        self.v = DWSeparableConv2d(dim, dim, 3, 1, 1)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MLP2d(dim, mlp_ratio=4.0)
        self.drop_path = DropPath(drop_path)

    def _sample_group(self, x, offsets, k, l):
        """
        Sample one sparse group: positions (k::rho, l::rho) with offsets at those anchor points.
        """
        B, C, H, W = x.shape
        rho = self.rho
        assert H % rho == 0 and W % rho == 0, "H,W must be divisible by rho for SSA scattering."

        # offsets at anchor points
        off = offsets[:, :, k::rho, l::rho]  # (B,2,H/rho,W/rho)
        off = torch.tanh(off) * self.offset_scale

        Hg, Wg = H // rho, W // rho
        y_base = (torch.arange(Hg, device=x.device).float() * rho + k).view(1, Hg, 1).expand(B, Hg, Wg)
        x_base = (torch.arange(Wg, device=x.device).float() * rho + l).view(1, 1, Wg).expand(B, Hg, Wg)

        y = (y_base + off[:, 0]).clamp(0, H - 1)
        xg = (x_base + off[:, 1]).clamp(0, W - 1)

        # normalize for grid_sample (x, y) in [-1,1]
        gx = 2.0 * xg / (W - 1) - 1.0
        gy = 2.0 * y / (H - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)  # (B,Hg,Wg,2)

        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled  # (B,C,Hg,Wg)

    def forward(self, x):
        B, C, H, W = x.shape
        rho = self.rho

        xn = self.norm1(x)
        offsets = self.off(xn)  # (B,2,H,W)

        out = torch.zeros_like(x)

        # per-group attention then scatter
        for k in range(rho):
            for l in range(rho):
                g = self._sample_group(xn, offsets, k, l)  # (B,C,H/rho,W/rho)

                q = self.q(g).flatten(2).transpose(1, 2)  # (B,HWg,C)
                ku = unfold_neighbors(self.k(g), self.window_size)
                vu = unfold_neighbors(self.v(g), self.window_size)

                # local attention (full heads) in the sparse lattice
                g_out = local_mha(q, ku, vu, self.num_heads)
                Hg, Wg = g.shape[2], g.shape[3]
                g_out = g_out.transpose(1, 2).view(B, C, Hg, Wg)

                # scatter back to dense lattice anchors
                out[:, :, k::rho, l::rho] = g_out

        x = x + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
