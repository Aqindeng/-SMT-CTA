import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset


class GenericChangeDetectionDataset(Dataset):
    """
    Generic paired change detection dataset:
      <root>/<split>/A/*.png
      <root>/<split>/B/*.png
      <root>/<split>/label/*.png
    """
    def __init__(self, root, split_folder, img_dir_a="A", img_dir_b="B", mask_dir="label",
                 img_ext=".png", mask_ext=".png", transform=None):
        self.root = root
        self.split = split_folder
        self.dir_a = os.path.join(root, split_folder, img_dir_a)
        self.dir_b = os.path.join(root, split_folder, img_dir_b)
        self.dir_m = os.path.join(root, split_folder, mask_dir)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

        # index by mask filenames
        self.masks = sorted(glob.glob(os.path.join(self.dir_m, f"*{mask_ext}")))
        if len(self.masks) == 0:
            raise RuntimeError(f"No masks found in: {self.dir_m}")

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        m_path = self.masks[idx]
        name = os.path.splitext(os.path.basename(m_path))[0]
        a_path = os.path.join(self.dir_a, name + self.img_ext)
        b_path = os.path.join(self.dir_b, name + self.img_ext)

        img_a = cv2.imread(a_path, cv2.IMREAD_COLOR)
        img_b = cv2.imread(b_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

        if img_a is None or img_b is None or mask is None:
            raise RuntimeError(f"Failed to read sample: {a_path}, {b_path}, {m_path}")

        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.uint8) * 255  # binary

        if self.transform is not None:
            img_a, img_b, mask = self.transform(img_a, img_b, mask)

        return img_a, img_b, mask, name
