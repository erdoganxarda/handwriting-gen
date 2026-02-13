from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor


SPLIT_NAMES = ("train", "val", "test")


class EMNISTCombinedDataset(Dataset):
    """Combined EMNIST letters train+test with consistent 0-25 labels."""

    def __init__(self, root: str | Path, normalize: bool = True, download: bool = False) -> None:
        self.root = Path(root)
        self.normalize = normalize
        transform = ToTensor()
        self.train_ds = EMNIST(
            root=self.root.as_posix(),
            split="letters",
            train=True,
            download=download,
            transform=transform,
            target_transform=lambda y: int(y) - 1,
        )
        self.test_ds = EMNIST(
            root=self.root.as_posix(),
            split="letters",
            train=False,
            download=download,
            transform=transform,
            target_transform=lambda y: int(y) - 1,
        )
        self.train_len = len(self.train_ds)

    def __len__(self) -> int:
        return self.train_len + len(self.test_ds)

    def __getitem__(self, index: int):
        if index < self.train_len:
            image, label = self.train_ds[index]
        else:
            image, label = self.test_ds[index - self.train_len]
        if self.normalize:
            image = image * 2.0 - 1.0
        return image, int(label)


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _split_path(data_dir: str | Path, seed: int) -> Path:
    return Path(data_dir) / "processed" / f"splits_letters_seed{seed}.npz"


def get_or_create_splits(
    data_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    download: bool = True,
) -> SplitIndices:
    path = _split_path(data_dir, seed)
    if path.exists():
        payload = np.load(path)
        return SplitIndices(
            train=payload["train_idx"],
            val=payload["val_idx"],
            test=payload["test_idx"],
        )

    dataset = EMNISTCombinedDataset(data_dir, normalize=False, download=download)
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val :])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def get_split_subset(
    dataset: Dataset,
    split_indices: SplitIndices,
    split: Literal["train", "val", "test"],
) -> Subset:
    indices = split_indices.as_dict()[split]
    return Subset(dataset, indices.tolist())


def make_emnist_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 2,
    download: bool = True,
) -> Dict[str, DataLoader]:
    split_indices = get_or_create_splits(data_dir=data_dir, seed=seed, download=download)
    dataset = EMNISTCombinedDataset(data_dir, normalize=True, download=download)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders: Dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:
        subset = get_split_subset(dataset, split_indices, split=split)  # type: ignore[arg-type]
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            generator=generator,
        )
    return loaders
