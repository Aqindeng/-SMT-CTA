import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # refinement convs from coarse to fine
        self.ref3 = nn.Sequential(nn.Conv2d(channels[2], channels[2], 3, padding=1), nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True))
        self.ref2 = nn.Sequential(nn.Conv2d(channels[1], channels[1], 3, padding=1), nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True))
        self.ref1 = nn.Sequential(nn.Conv2d(channels[0], channels[0], 3, padding=1), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))

        self.proj3 = nn.Conv2d(channels[3], channels[2], 1)
        self.proj2 = nn.Conv2d(channels[2], channels[1], 1)
        self.proj1 = nn.Conv2d(channels[1], channels[0], 1)

    def forward(self, e1, e2, e3, e4):
        # bottom-up aggregation
        a4 = e4
        a3 = self.ref3(e3 + F.interpolate(self.proj3(a4), size=e3.shape[-2:], mode="bilinear", align_corners=False))
        a2 = self.ref2(e2 + F.interpolate(self.proj2(a3), size=e2.shape[-2:], mode="bilinear", align_corners=False))
        a1 = self.ref1(e1 + F.interpolate(self.proj1(a2), size=e1.shape[-2:], mode="bilinear", align_corners=False))
        return a1, a2, a3, a4


class LightweightDecoder(nn.Module):
    def __init__(self, ch1: int, ch2: int):
        super().__init__()
        # fuse stage2 into stage1 resolution
        self.fuse12 = nn.Sequential(
            nn.Conv2d(ch1 + ch2, ch1, 1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        # upsample to input resolution (patch stride=4 => stage1 is 1/4 resolution)
        self.up = nn.Sequential(
            nn.Conv2d(ch1, ch1, 3, padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(ch1, 1, 1)

    def forward(self, a1, e2, out_size):
        # upsample e2 to stage1 size and fuse
        e2u = F.interpolate(e2, size=a1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse12(torch.cat([a1, e2u], dim=1))

        # upsample to full resolution
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        x = self.up(x)
        logits = self.head(x)
        return logits
