# SMT-CTA: Spatiotemporal Multi-granular Transformer with Cross-temporal Aggregation

This repository provides a PyTorch implementation of **SMT-CTA**, a hybrid CNN–Transformer model for bi-temporal remote sensing change detection, incorporating:
- Change-Aware Attention (CAAttn) with grouped heads and difference-aware cross-temporal interaction
- Shuffled Sparse Attention (SSA) with learnable offsets for content-adaptive sparse token mixing
- Change-Weighted Feature Fusion (CWFF) with per-channel softmax competition
- Bottom-up pyramid aggregation and a lightweight decoder
- Composite supervision: CE + stage-wise WBCE + SSIM + SoftIoU
- Evaluation metrics: OA, Precision, Recall, F1, mIoU, and Boundary F1 (BF1)

## Installation

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Download datasets
See `download_data.md`. This repository does **not** redistribute any dataset content.

## Quick Start

### Train
```bash
python scripts/train.py --config configs/levir_cd.yaml
```

### Evaluate
```bash
python scripts/eval.py --config configs/levir_cd.yaml --ckpt path/to/best.pt
```

### Inference (single pair)
```bash
python scripts/infer.py \
  --config configs/levir_cd.yaml \
  --ckpt path/to/best.pt \
  --img_a path/to/pre.png \
  --img_b path/to/post.png \
  --out out_mask.png
```

## Reproducibility
- Seeds are fixed via `smt_cta.utils.seed.set_seed`.
- All hyperparameters are stored in YAML configs.
