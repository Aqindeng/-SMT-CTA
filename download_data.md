# Dataset download instructions (no redistribution)

This repository does not redistribute datasets. You must download them from the official sources and comply with their licenses.

## Expected directory convention

Set `dataset.root` in each YAML config to the dataset root folder. The default loaders assume:

```
<root>/
  train/
    A/   (pre-change images)
    B/   (post-change images)
    label/ (binary masks)
  val/
    A/
    B/
    label/
  test/
    A/
    B/
    label/
```

If your dataset uses different folder names, modify the config fields:
- `img_dir_a`, `img_dir_b`, `mask_dir`, and `split_folders`.

## LEVIR-CD
Source: Kaggle:https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd (LEVIR-CD).
Download and arrange into the structure above.

## EGY-BCD
Source: GitHub:https://github.com/oshholail/EGY-BCD (EGY-BCD).
Download and arrange into the structure above.

## SYSU-CD
Source: GitHub:https://github.com/liumency/SYSU-CD (SYSU-CD).
Download and arrange into the structure above.
