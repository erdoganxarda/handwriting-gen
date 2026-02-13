from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FakeData
from torchvision.transforms import Compose, Lambda, ToTensor


def make_fake_emnist_loaders(
    batch_size: int,
    seed: int = 42,
    train_size: int = 512,
    val_size: int = 128,
    test_size: int = 128,
    num_workers: int = 0,
):
    transform = Compose([ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    train_ds = FakeData(
        size=train_size,
        image_size=(1, 28, 28),
        num_classes=26,
        transform=transform,
        random_offset=seed,
    )
    val_ds = FakeData(
        size=val_size,
        image_size=(1, 28, 28),
        num_classes=26,
        transform=transform,
        random_offset=seed + 1,
    )
    test_ds = FakeData(
        size=test_size,
        image_size=(1, 28, 28),
        num_classes=26,
        transform=transform,
        random_offset=seed + 2,
    )

    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def _random_walk_sequences(num_samples: int, max_len: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    seq = np.zeros((num_samples, max_len, 3), dtype=np.float32)
    mask = np.zeros((num_samples, max_len), dtype=np.float32)
    labels = rng.integers(0, 26, size=(num_samples,), dtype=np.int64)

    for i in range(num_samples):
        length = int(rng.integers(max_len // 3, max_len))
        dx = rng.normal(0.0, 0.05, size=(length,)).astype(np.float32)
        dy = rng.normal(0.0, 0.05, size=(length,)).astype(np.float32)
        pen = np.zeros((length,), dtype=np.float32)
        pen[rng.choice(length, size=max(1, length // 12), replace=False)] = 1.0
        pen[-1] = 1.0

        seq[i, :length, 0] = np.clip(dx, -1.0, 1.0)
        seq[i, :length, 1] = np.clip(dy, -1.0, 1.0)
        seq[i, :length, 2] = pen
        mask[i, :length] = 1.0

    return seq, mask, labels


def make_fake_stroke_loaders(
    batch_size: int,
    max_len: int,
    seed: int = 42,
    train_size: int = 512,
    val_size: int = 128,
    test_size: int = 128,
    num_workers: int = 0,
):
    train_seq, train_mask, train_labels = _random_walk_sequences(train_size, max_len=max_len, seed=seed)
    val_seq, val_mask, val_labels = _random_walk_sequences(val_size, max_len=max_len, seed=seed + 1)
    test_seq, test_mask, test_labels = _random_walk_sequences(test_size, max_len=max_len, seed=seed + 2)

    def _loader(seq: np.ndarray, mask: np.ndarray, labels: np.ndarray, shuffle: bool):
        ds = TensorDataset(
            torch.from_numpy(seq),
            torch.from_numpy(mask),
            torch.from_numpy(labels),
        )
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=gen)

    return {
        "train": _loader(train_seq, train_mask, train_labels, True),
        "val": _loader(val_seq, val_mask, val_labels, False),
        "test": _loader(test_seq, test_mask, test_labels, False),
    }
