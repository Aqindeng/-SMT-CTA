import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LayerNorm2d
from .attention import CAAttnBlock
from .ssa import SSA


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = LayerNorm2d(out_ch)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Downsample(nn.Module):
    """Simple pyramid downsampling (2x)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.norm = LayerNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.conv(x))


class EncoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, head_groups, window_size, ssa_rho, drop_path_rates):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(
                CAAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_groups=head_groups,
                    window_size=window_size,
                    mlp_ratio=4.0,
                    drop_path=drop_path_rates[i],
                )
            )
            # SSA for each stream (token mixing / sparsified attention)
            blocks.append(
                nn.ModuleDict({
                    "ssa_a": SSA(dim, num_heads, window_size=window_size, rho=ssa_rho, drop_path=drop_path_rates[i]),
                    "ssa_b": SSA(dim, num_heads, window_size=window_size, rho=ssa_rho, drop_path=drop_path_rates[i]),
                })
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, xa, xb):
        for m in self.blocks:
            if isinstance(m, CAAttnBlock):
                xa, xb = m(xa, xb)
            else:
                xa = m["ssa_a"](xa)
                xb = m["ssa_b"](xb)
        return xa, xb


class SMTCTAEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=(64, 128, 256, 512),
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 8, 16),
        head_groups=(1, 1, 1),
        window_sizes=(7, 7, 9, 11),
        ssa_rhos=(1, 2, 2, 4),
        drop_path=0.2,
    ):
        super().__init__()
        self.patch = PatchEmbed(in_channels, base_channels[0], patch_size=4)

        self.down1 = Downsample(base_channels[0], base_channels[1])
        self.down2 = Downsample(base_channels[1], base_channels[2])
        self.down3 = Downsample(base_channels[2], base_channels[3])

        # stochastic depth schedule
        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path, total_blocks).tolist()

        idx = 0
        self.stage1 = EncoderStage(base_channels[0], depths[0], num_heads[0], head_groups, window_sizes[0], ssa_rhos[0], dpr[idx:idx+depths[0]])
        idx += depths[0]
        self.stage2 = EncoderStage(base_channels[1], depths[1], num_heads[1], head_groups, window_sizes[1], ssa_rhos[1], dpr[idx:idx+depths[1]])
        idx += depths[1]
        self.stage3 = EncoderStage(base_channels[2], depths[2], num_heads[2], head_groups, window_sizes[2], ssa_rhos[2], dpr[idx:idx+depths[2]])
        idx += depths[2]
        self.stage4 = EncoderStage(base_channels[3], depths[3], num_heads[3], head_groups, window_sizes[3], ssa_rhos[3], dpr[idx:idx+depths[3]])

    def forward(self, xa, xb):
        # patch embed
        xa = self.patch(xa)
        xb = self.patch(xb)

        xa1, xb1 = self.stage1(xa, xb)

        xa2, xb2 = self.stage2(self.down1(xa1), self.down1(xb1))
        xa3, xb3 = self.stage3(self.down2(xa2), self.down2(xb2))
        xa4, xb4 = self.stage4(self.down3(xa3), self.down3(xb3))

        feats_a = [xa1, xa2, xa3, xa4]
        feats_b = [xb1, xb2, xb3, xb4]
        return feats_a, feats_b
