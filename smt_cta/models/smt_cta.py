import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SMTCTAEncoder
from .cwff import CWFF
from .decoder import PyramidAggregation, LightweightDecoder


class SMTCTA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.aux_outputs = bool(m.get("aux_outputs", True))

        self.encoder = SMTCTAEncoder(
            in_channels=m["in_channels"],
            base_channels=m["base_channels"],
            depths=m["depths"],
            num_heads=m["num_heads"],
            head_groups=tuple(m["head_groups"]),
            window_sizes=m["window_sizes"],
            ssa_rhos=m["ssa_rhos"],
            drop_path=m.get("drop_path", 0.2),
        )

        ch = m["base_channels"]
        self.cwff1 = CWFF(ch[0])
        self.cwff2 = CWFF(ch[1])
        self.cwff3 = CWFF(ch[2])
        self.cwff4 = CWFF(ch[3])

        self.pyr = PyramidAggregation(channels=ch)
        self.dec = LightweightDecoder(ch1=ch[0], ch2=ch[1])

        if self.aux_outputs:
            self.aux1 = nn.Conv2d(ch[0], 1, 1)
            self.aux2 = nn.Conv2d(ch[1], 1, 1)
            self.aux3 = nn.Conv2d(ch[2], 1, 1)
            self.aux4 = nn.Conv2d(ch[3], 1, 1)

    def forward(self, xa, xb):
        """
        xa, xb: (B,3,H,W)
        Returns:
          logits: (B,1,H,W)
          aux_logits: list of 4 tensors (optional), each upsampled to (H,W)
        """
        B, _, H, W = xa.shape
        feats_a, feats_b = self.encoder(xa, xb)

        e1, _, _ = self.cwff1(feats_a[0], feats_b[0])
        e2, _, _ = self.cwff2(feats_a[1], feats_b[1])
        e3, _, _ = self.cwff3(feats_a[2], feats_b[2])
        e4, _, _ = self.cwff4(feats_a[3], feats_b[3])

        a1, a2, a3, a4 = self.pyr(e1, e2, e3, e4)
        logits = self.dec(a1, e2, out_size=(H, W))

        aux = None
        if self.aux_outputs:
            aux = [
                F.interpolate(self.aux1(a1), size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(self.aux2(a2), size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(self.aux3(a3), size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(self.aux4(a4), size=(H, W), mode="bilinear", align_corners=False),
            ]

        return logits, aux
