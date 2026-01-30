import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from smt_cta.utils import load_config, set_seed, get_logger, save_checkpoint
from smt_cta.data import GenericChangeDetectionDataset, build_train_transforms, build_eval_transforms
from smt_cta.models import SMTCTA
from smt_cta.losses import CompositeLoss
from smt_cta.metrics import compute_metrics_batch, boundary_f1


def build_loaders(cfg):
    ds_cfg = cfg["dataset"]
    root = ds_cfg["root"]
    splits = ds_cfg["split_folders"]

    train_tf = build_train_transforms()
    eval_tf = build_eval_transforms()

    train_ds = GenericChangeDetectionDataset(root, splits["train"], ds_cfg["img_dir_a"], ds_cfg["img_dir_b"],
                                            ds_cfg["mask_dir"], ds_cfg["img_ext"], ds_cfg["mask_ext"], train_tf)
    val_ds = GenericChangeDetectionDataset(root, splits["val"], ds_cfg["img_dir_a"], ds_cfg["img_dir_b"],
                                          ds_cfg["mask_dir"], ds_cfg["img_ext"], ds_cfg["mask_ext"], eval_tf)

    tr = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    va = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return tr, va


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5, tol=2):
    model.eval()
    preds, gts = [], []
    bf1s = []

    for a, b, m, _ in loader:
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
    return {"OA": oa, "Pre": pre, "Rec": rec, "F1": f1, "mIoU": miou, "BF1": bf1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["experiment"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg["train"]["seed"])
    logger = get_logger(out_dir, name=cfg["experiment"]["name"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_loader, val_loader = build_loaders(cfg)

    model = SMTCTA(cfg).to(device)
    loss_fn = CompositeLoss(cfg["loss"]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))

    best_f1 = -1.0
    bad = 0
    patience = int(cfg["train"]["early_stop_patience"])

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        running = 0.0

        for a, b, m, _ in pbar:
            a = a.to(device)
            b = b.to(device)
            m = m.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"])):
                logits, aux = model(a, b)
                loss = loss_fn(logits, aux, m)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            pbar.set_postfix(loss=running / max(1, pbar.n))

        metrics = evaluate(model, val_loader, device,
                           thr=float(cfg["eval"]["threshold"]),
                           tol=int(cfg["eval"]["boundary_tol"]))

        logger.info(f"[Val] Epoch={epoch} OA={metrics['OA']:.4f} Pre={metrics['Pre']:.4f} "
                    f"Rec={metrics['Rec']:.4f} F1={metrics['F1']:.4f} mIoU={metrics['mIoU']:.4f} BF1={metrics['BF1']:.4f}")

        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            bad = 0
            save_checkpoint(out_dir, "best.pt", model, opt, epoch, best_f1)
            logger.info(f"Saved best checkpoint (F1={best_f1:.4f}).")
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"Early stopping triggered (patience={patience}).")
                break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
