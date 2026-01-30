import numpy as np
import cv2


def confusion_counts(pred, gt):
    # pred, gt: uint8 {0,1}
    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    return tp, tn, fp, fn


def compute_metrics_from_counts(tp, tn, fp, fn, eps=1e-6):
    pre = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * pre * rec / (pre + rec + eps)
    oa = (tp + tn) / (tp + tn + fp + fn + eps)
    iou_c = tp / (tp + fp + fn + eps)
    iou_u = tn / (tn + fp + fn + eps)
    miou = 0.5 * (iou_c + iou_u)
    return oa, pre, rec, f1, miou


def compute_metrics_batch(preds, gts):
    # preds, gts: list of numpy uint8 masks {0,1}
    TP = TN = FP = FN = 0
    for p, g in zip(preds, gts):
        tp, tn, fp, fn = confusion_counts(p, g)
        TP += tp; TN += tn; FP += fp; FN += fn
    return compute_metrics_from_counts(TP, TN, FP, FN)


def mask_boundary(mask):
    # boundary = mask - erode(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    bd = (mask.astype(np.uint8) - er).clip(0, 1)
    return bd


def boundary_f1(pred_mask, gt_mask, tol=2, eps=1e-6):
    """
    Boundary F1 with tolerance dilation:
      Pre_b = |Bd_pred ∩ dilate(Bd_gt)| / |Bd_pred|
      Rec_b = |Bd_gt ∩ dilate(Bd_pred)| / |Bd_gt|
    """
    pred_bd = mask_boundary(pred_mask)
    gt_bd = mask_boundary(gt_mask)

    if pred_bd.sum() == 0 and gt_bd.sum() == 0:
        return 1.0
    if pred_bd.sum() == 0 or gt_bd.sum() == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gt_d = cv2.dilate(gt_bd, k, iterations=1)
    pr_d = cv2.dilate(pred_bd, k, iterations=1)

    pre_b = (pred_bd & gt_d).sum() / (pred_bd.sum() + eps)
    rec_b = (gt_bd & pr_d).sum() / (gt_bd.sum() + eps)
    bf1 = 2 * pre_b * rec_b / (pre_b + rec_b + eps)
    return float(bf1)
