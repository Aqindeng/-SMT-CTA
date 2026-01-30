import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(2, 3))
    union = (probs + targets - probs * targets).sum(dim=(2, 3))
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()


def gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    return kernel2d


def ssim_loss(logits, targets, window_size=11, sigma=1.5, eps=1e-6):
    """
    Differentiable SSIM loss on sigmoid probabilities.
    """
    x = torch.sigmoid(logits)
    y = targets.float()

    B, C, H, W = x.shape
    device = x.device
    k = gaussian_kernel(window_size, sigma, device=device).view(1, 1, window_size, window_size)
    k = k.repeat(C, 1, 1, 1)  # depthwise

    mu_x = F.conv2d(x, k, padding=window_size//2, groups=C)
    mu_y = F.conv2d(y, k, padding=window_size//2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(x * x, k, padding=window_size//2, groups=C) - mu_x2
    sig_y2 = F.conv2d(y * y, k, padding=window_size//2, groups=C) - mu_y2
    sig_xy = F.conv2d(x * y, k, padding=window_size//2, groups=C) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2) + eps)
    return 1.0 - ssim.mean()


def weighted_bce_loss(logits, targets, max_pos_weight=10.0):
    """
    Weighted BCE with adaptive positive class weight computed per mini-batch:
      pos_weight = min(max_pos_weight, Nneg / (Npos + eps))
    """
    targets = targets.float()
    with torch.no_grad():
        npos = targets.sum()
        nneg = targets.numel() - npos
        pos_weight = (nneg / (npos + 1e-6)).clamp(max=max_pos_weight)
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    return loss


class CompositeLoss(nn.Module):
    """
    L = CE(final) + sum_s λ_s [ WBCE_s + w_ssim*SSIM_s + w_siou*SoftIoU_s ]
    """
    def __init__(self, cfg_loss):
        super().__init__()
        self.ce_weight = float(cfg_loss["ce_weight"])
        self.stage_weights = cfg_loss["stage_weights"]
        self.wbce_weight = float(cfg_loss["wbce_weight"])
        self.ssim_weight = float(cfg_loss["ssim_weight"])
        self.siou_weight = float(cfg_loss["siou_weight"])
        self.max_pos_weight = float(cfg_loss.get("max_pos_weight", 10.0))

    def forward(self, logits, aux_logits, targets):
        # targets: (B,1,H,W) float or uint8
        targets = targets.float()

        # final CE on logits as BCE (binary)
        loss_final = F.binary_cross_entropy_with_logits(logits, targets)

        loss_aux = 0.0
        if aux_logits is not None:
            for s, l_s in enumerate(aux_logits):
                lam = float(self.stage_weights[s])
                wbce = weighted_bce_loss(l_s, targets, max_pos_weight=self.max_pos_weight)
                ssim = ssim_loss(l_s, targets)
                siou = soft_iou_loss(l_s, targets)
                loss_aux = loss_aux + lam * (self.wbce_weight * wbce + self.ssim_weight * ssim + self.siou_weight * siou)

        return self.ce_weight * loss_final + loss_aux
