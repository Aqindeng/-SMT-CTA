import random
import numpy as np
import torch
import cv2


class Compose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, a, b, m):
        for op in self.ops:
            a, b, m = op(a, b, m)
        return a, b, m


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, a, b, m):
        if random.random() < self.p:
            a = np.flip(a, axis=1).copy()
            b = np.flip(b, axis=1).copy()
            m = np.flip(m, axis=1).copy()
        if random.random() < self.p:
            a = np.flip(a, axis=0).copy()
            b = np.flip(b, axis=0).copy()
            m = np.flip(m, axis=0).copy()
        return a, b, m


class RandomRotate90:
    def __call__(self, a, b, m):
        k = random.randint(0, 3)
        if k > 0:
            a = np.rot90(a, k).copy()
            b = np.rot90(b, k).copy()
            m = np.rot90(m, k).copy()
        return a, b, m


class ColorJitter:
    def __init__(self, p=0.5, brightness=0.15, contrast=0.15):
        self.p = p
        self.b = brightness
        self.c = contrast

    def __call__(self, a, b, m):
        if random.random() >= self.p:
            return a, b, m

        def jitter(x):
            x = x.astype(np.float32) / 255.0
            br = 1.0 + random.uniform(-self.b, self.b)
            ct = 1.0 + random.uniform(-self.c, self.c)
            x = x * br
            mean = x.mean(axis=(0, 1), keepdims=True)
            x = (x - mean) * ct + mean
            x = np.clip(x, 0, 1)
            return (x * 255.0).astype(np.uint8)

        return jitter(a), jitter(b), m


class GaussianBlur:
    def __init__(self, p=0.2, sigma_max=1.0):
        self.p = p
        self.sigma_max = sigma_max

    def __call__(self, a, b, m):
        if random.random() >= self.p:
            return a, b, m
        sigma = random.uniform(0.0, self.sigma_max)
        k = 3 if sigma < 0.5 else 5
        a = cv2.GaussianBlur(a, (k, k), sigmaX=sigma)
        b = cv2.GaussianBlur(b, (k, k), sigmaX=sigma)
        return a, b, m


class ToTensorNormalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean if mean is not None else [0.0, 0.0, 0.0]
        self.std = std if std is not None else [1.0, 1.0, 1.0]

    def __call__(self, a, b, m):
        a = torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
        b = torch.from_numpy(b).permute(2, 0, 1).float() / 255.0
        m = torch.from_numpy(m).unsqueeze(0).float() / 255.0

        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        a = (a - mean) / std
        b = (b - mean) / std
        return a, b, m


def build_train_transforms():
    return Compose([
        RandomFlip(p=0.5),
        RandomRotate90(),
        ColorJitter(p=0.5, brightness=0.15, contrast=0.15),
        GaussianBlur(p=0.2, sigma_max=1.0),
        ToTensorNormalize(),
    ])


def build_eval_transforms():
    return Compose([
        ToTensorNormalize(),
    ])
