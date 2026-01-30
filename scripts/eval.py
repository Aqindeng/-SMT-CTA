import argparse
import torch
from torch.utils.data import DataLoader

from smt_cta.utils import load_config
from smt_cta.data import GenericChangeDetectionDataset, build_eval_transforms
from smt_cta.models import SMTCTA
from smt_cta.utils.checkpoint import load_checkpoint
from smt_cta.metrics import compute_metrics_batch, boundary_f1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_cfg = cfg["dataset"]
    root = ds_cfg["root"]
    split = ds_cfg["split_folders"]["test"]
    tf = build_eval_transforms()

    ds = GenericChangeDetectionDataset(root, split, ds_cfg["img_dir_a"], ds_cfg["img_dir_b"],
                                       ds_cfg["mask_dir"], ds_cfg["img_ext"], ds_cfg["mask_ext"], tf)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = SMTCTA(cfg).to(device)
    load_checkpoint(args.ckpt, model, optimizer=None, map_location=device)
    model.eval()

    thr = float(cfg["eval"]["threshold"])
    tol = int(cfg["eval"]["boundary_tol"])

    preds, gts, bf1s = [], [], []

    for a, b, m, _ in dl:
        a = a.to(device)
        b = b.to(device)
        m = m.to(device)

        logits, _ = model(a, b)
        prob = torch.sigmoid(logits)
        pred = (prob > thr).float()

        pred_np = pred[0, 0].cpu().numpy().astype("uint8")
        gt_np = (m[0, 0].cpu().numpy() > 0.5).astype("uint8")

        preds.append(pred_np)
        gts.append(gt_np)
        bf1s.append(boundary_f1(pred_np, gt_np, tol=tol))

    oa, pre, rec, f1, miou = compute_metrics_batch(preds, gts)
    bf1 = float(sum(bf1s) / max(len(bf1s), 1))

    print(f"OA={oa:.4f} Pre={pre:.4f} Rec={rec:.4f} F1={f1:.4f} mIoU={miou:.4f} BF1={bf1:.4f}")


if __name__ == "__main__":
    main()
