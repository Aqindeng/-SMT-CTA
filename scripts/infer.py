import argparse
import cv2
import numpy as np
import torch

from smt_cta.utils import load_config
from smt_cta.models import SMTCTA
from smt_cta.utils.checkpoint import load_checkpoint
from smt_cta.data.transforms import build_eval_transforms


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_a", required=True)
    ap.add_argument("--img_b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SMTCTA(cfg).to(device)
    load_checkpoint(args.ckpt, model, optimizer=None, map_location=device)
    model.eval()

    a = cv2.cvtColor(cv2.imread(args.img_a), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.imread(args.img_b), cv2.COLOR_BGR2RGB)

    tf = build_eval_transforms()
    a, b, _ = tf(a, b, np.zeros((a.shape[0], a.shape[1]), dtype=np.uint8))
    a = a.unsqueeze(0).to(device)
    b = b.unsqueeze(0).to(device)

    logits, _ = model(a, b)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > float(cfg["eval"]["threshold"])).astype(np.uint8) * 255

    cv2.imwrite(args.out, mask)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
