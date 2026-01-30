import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LayerNorm2d, MLP2d, DropPath, DWSeparableConv2d, unfold_neighbors, local_mha


class CAAttnBlock(nn.Module):
    """
    Change-Aware Attention (CAAttn):
    - Self attention on alpha
    - Self attention on beta
    - Cross-temporal attention using difference-aware value stream
    Local key/value expansion is realized via unfolding kxk neighborhoods.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_groups=(1, 1, 1),  # (self_alpha, self_beta, cross) ratios
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        gsum = sum(head_groups)
        assert num_heads % gsum == 0, "num_heads must be divisible by sum(head_groups)."
        self.h_self_a = num_heads * head_groups[0] // gsum
        self.h_self_b = num_heads * head_groups[1] // gsum
        self.h_cross = num_heads * head_groups[2] // gsum

        # projections for alpha/beta
        self.q_a = DWSeparableConv2d(dim, dim, k=3, s=1, p=1, bias=True)
        self.k_a = DWSeparableConv2d(dim, dim, k=3, s=1, p=1, bias=True)
        self.v_a = DWSeparableConv2d(dim, dim, k=3, s=1, p=1, bias=True)

        self.q_b = DWSeparableConv2d(dim, dim, k=3, s=1, p=1, bias=True)
        self.k_b = DWSeparableConv2d(dim, dim, k=3, s=1, p=1, bias=True)
        self.v_b = DWSeparableConv2d(dim, dim, k=3, s=1, p=1, bias=True)

        # cross-query from appearance difference (practical instantiation)
        self.q_c = DWSeparableConv2d(dim, dim, k=3, s=1, p=1, bias=True)

        # project [self_out, cross_out] back to dim for each stream
        self.proj_a = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.proj_b = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.norm_a1 = LayerNorm2d(dim)
        self.norm_b1 = LayerNorm2d(dim)
        self.norm_a2 = LayerNorm2d(dim)
        self.norm_b2 = LayerNorm2d(dim)

        self.mlp_a = MLP2d(dim, mlp_ratio=mlp_ratio)
        self.mlp_b = MLP2d(dim, mlp_ratio=mlp_ratio)

        self.drop_path = DropPath(drop_path)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        """
        xa, xb: (B,C,H,W)
        returns: xa', xb' (B,C,H,W)
        """
        B, C, H, W = xa.shape
        assert xb.shape == xa.shape

        # Pre-norm
        na = self.norm_a1(xa)
        nb = self.norm_b1(xb)

        qa = self.q_a(na)
        ka = self.k_a(na)
        va = self.v_a(na)

        qb = self.q_b(nb)
        kb = self.k_b(nb)
        vb = self.v_b(nb)

        # difference-aware value stream
        d = torch.abs(va - vb)

        # cross query from absolute appearance difference
        qc = self.q_c(torch.abs(na - nb))

        # flatten queries
        qa = qa.flatten(2).transpose(1, 2)  # (B,HW,C)
        qb = qb.flatten(2).transpose(1, 2)
        qc = qc.flatten(2).transpose(1, 2)

        # local expansion (keys and values)
        ka_u = unfold_neighbors(ka, self.window_size)  # (B,HW,KK,C)
        va_u = unfold_neighbors(va, self.window_size)
        kb_u = unfold_neighbors(kb, self.window_size)
        vb_u = unfold_neighbors(vb, self.window_size)

        d_u = unfold_neighbors(d, self.window_size)

        # allocate channels per group using head partitioning
        # For simplicity: use contiguous channel blocks. Each group uses (heads_group * head_dim) channels.
        head_dim = C // self.num_heads
        ca = self.h_self_a * head_dim
        cb = self.h_self_b * head_dim
        cc = self.h_cross * head_dim

        qa_s = qa[:, :, :ca]
        qb_s = qb[:, :, :cb]

        # cross uses first cc channels of qc / and first cc channels of keys/values
        qc_s = qc[:, :, :cc]

        ka_c = ka_u[:, :, :, :cc]
        kb_c = kb_u[:, :, :, :cc]

        # cross keys: concat expanded keys from both dates
        k_cross = torch.cat([ka_c, kb_c], dim=2)  # (B,HW,2*KK,cc)

        # cross values: duplicate local diff-values for both key sets (alignment by spatial neighborhood)
        v_cross = torch.cat([d_u[:, :, :, :cc], d_u[:, :, :, :cc]], dim=2)

        # self (alpha) uses its own local neighbors
        out_a_self = local_mha(qa_s, ka_u[:, :, :, :ca], va_u[:, :, :, :ca], self.h_self_a)  # (B,HW,ca)
        out_b_self = local_mha(qb_s, kb_u[:, :, :, :cb], vb_u[:, :, :, :cb], self.h_self_b)  # (B,HW,cb)

        # cross attention
        out_cross = local_mha(qc_s, k_cross, v_cross, self.h_cross)  # (B,HW,cc)

        # reshape and fuse for each stream
        out_a_self = out_a_self.transpose(1, 2).view(B, ca, H, W)
        out_b_self = out_b_self.transpose(1, 2).view(B, cb, H, W)
        out_cross  = out_cross.transpose(1, 2).view(B, cc, H, W)

        # combine [self, cross] then project to full dim
        # (we pad self outputs to dim via concatenation with zeros if needed)
        pad_a = torch.zeros(B, C - ca, H, W, device=xa.device, dtype=xa.dtype)
        pad_b = torch.zeros(B, C - cb, H, W, device=xb.device, dtype=xb.dtype)
        a_full = torch.cat([out_a_self, pad_a], dim=1)
        b_full = torch.cat([out_b_self, pad_b], dim=1)

        # concatenate with cross (broadcast to dim via padding)
        pad_c = torch.zeros(B, C - cc, H, W, device=xa.device, dtype=xa.dtype)
        c_full = torch.cat([out_cross, pad_c], dim=1)

        attn_a = self.proj_a(torch.cat([a_full, c_full], dim=1))
        attn_b = self.proj_b(torch.cat([b_full, c_full], dim=1))

        xa = xa + self.drop_path(attn_a)
        xb = xb + self.drop_path(attn_b)

        # FFN
        xa = xa + self.drop_path(self.mlp_a(self.norm_a2(xa)))
        xb = xb + self.drop_path(self.mlp_b(self.norm_b2(xb)))
        return xa, xb
