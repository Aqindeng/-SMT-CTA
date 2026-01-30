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
Source: Kaggle (LEVIR-CD).
Download and arrange into the structure above.

## EGY-BCD
Source: GitHub (EGY-BCD).
Download and arrange into the structure above.

## SYSU-CD
Source: GitHub (SYSU-CD).
Download and arrange into the structure above.

## DSIFN-CD
Download from the dataset authors’ repository and arrange into train/val/test folders.
