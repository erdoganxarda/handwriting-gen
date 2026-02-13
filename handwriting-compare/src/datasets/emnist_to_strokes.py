from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from skimage.morphology import skeletonize
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.datasets.emnist import EMNISTCombinedDataset, SplitIndices, get_or_create_splits

Coord = Tuple[int, int]


@dataclass
class StrokeCache:
    sequences: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _neighbors8(r: int, c: int) -> List[Coord]:
    out: List[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _build_adjacency(skel: np.ndarray) -> Dict[Coord, List[Coord]]:
    coords = np.argwhere(skel > 0)
    pixels = {tuple(x) for x in coords.tolist()}
    adjacency: Dict[Coord, List[Coord]] = {p: [] for p in pixels}

    for r, c in pixels:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in pixels:
                adjacency[(r, c)].append((nr, nc))
    return adjacency


def _connected_components(adjacency: Dict[Coord, List[Coord]]) -> List[List[Coord]]:
    comps: List[List[Coord]] = []
    seen: set[Coord] = set()

    for node in adjacency:
        if node in seen:
            continue
        stack = [node]
        comp: List[Coord] = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _is_critical(node: Coord, adjacency: Dict[Coord, List[Coord]]) -> bool:
    deg = len(adjacency[node])
    return deg != 2


def _edge_key(a: Coord, b: Coord) -> Tuple[Coord, Coord]:
    return (a, b) if a <= b else (b, a)


def _walk_path(
    start: Coord,
    nxt: Coord,
    critical: set[Coord],
    adjacency: Dict[Coord, List[Coord]],
    visited_edges: set[Tuple[Coord, Coord]],
) -> List[Coord]:
    path = [start, nxt]
    visited_edges.add(_edge_key(start, nxt))
    prev = start
    cur = nxt

    while True:
        if cur in critical and cur != start:
            break

        candidates = [v for v in adjacency[cur] if v != prev]
        unvisited = [v for v in candidates if _edge_key(cur, v) not in visited_edges]
        if not unvisited:
            break

        nxt = unvisited[0]
        visited_edges.add(_edge_key(cur, nxt))
        path.append(nxt)
        prev, cur = cur, nxt

    return path


def _extract_component_paths(component: Sequence[Coord], adjacency: Dict[Coord, List[Coord]]) -> List[List[Coord]]:
    critical = {n for n in component if _is_critical(n, adjacency)}
    if not critical:
        critical = {min(component)}

    paths: List[List[Coord]] = []
    visited_edges: set[Tuple[Coord, Coord]] = set()

    for node in sorted(critical):
        for nb in adjacency[node]:
            key = _edge_key(node, nb)
            if key in visited_edges:
                continue
            path = _walk_path(node, nb, critical, adjacency, visited_edges)
            paths.append(path)

    # Capture leftover cycle edges that may be disconnected from selected critical traversals.
    for node in component:
        for nb in adjacency[node]:
            key = _edge_key(node, nb)
            if key in visited_edges:
                continue
            path = _walk_path(node, nb, critical, adjacency, visited_edges)
            paths.append(path)

    return paths


def _to_xy(path: Sequence[Coord], height: int, width: int) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float32)
    ys = arr[:, 0] / max(1, height - 1)
    xs = arr[:, 1] / max(1, width - 1)
    return np.stack([xs, ys], axis=1)


def _left_top_endpoint(paths: List[np.ndarray]) -> Tuple[int, bool]:
    best_idx = -1
    best_flip = False
    best_key = (float("inf"), float("inf"))
    for i, p in enumerate(paths):
        for flip in (False, True):
            start = p[-1] if flip else p[0]
            key = (float(start[0]), float(start[1]))
            if key < best_key:
                best_idx = i
                best_flip = flip
                best_key = key
    return best_idx, best_flip


def _order_paths(paths: List[np.ndarray]) -> List[np.ndarray]:
    if not paths:
        return []

    remaining = set(range(len(paths)))
    ordered: List[np.ndarray] = []

    idx, flip = _left_top_endpoint(paths)
    current = paths[idx][::-1] if flip else paths[idx]
    ordered.append(current)
    remaining.remove(idx)
    current_end = current[-1]

    while remaining:
        best_idx = -1
        best_flip = False
        best_dist = float("inf")

        for i in remaining:
            p = paths[i]
            d_start = float(np.linalg.norm(current_end - p[0]))
            d_end = float(np.linalg.norm(current_end - p[-1]))
            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                best_flip = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx = i
                best_flip = True

        nxt = paths[best_idx][::-1] if best_flip else paths[best_idx]
        ordered.append(nxt)
        remaining.remove(best_idx)
        current_end = nxt[-1]

    return ordered


def _fallback_path_from_ink(ink: np.ndarray) -> np.ndarray:
    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        return np.asarray([[0.5, 0.5]], dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    pts[:, 0] /= max(1, ink.shape[1] - 1)
    pts[:, 1] /= max(1, ink.shape[0] - 1)
    return pts


def image_to_stroke_sequence(image: np.ndarray, max_len: int = 160, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2D EMNIST image in [0,1] to a fixed-length (dx,dy,pen) sequence."""
    if image.ndim != 2:
        raise ValueError("image must be [H, W]")

    ink = (1.0 - image) > threshold
    skel = skeletonize(ink)

    adjacency = _build_adjacency(skel)
    all_paths_xy: List[np.ndarray] = []

    if adjacency:
        components = _connected_components(adjacency)
        for comp in components:
            comp_paths = _extract_component_paths(comp, adjacency)
            for p in comp_paths:
                if len(p) < 2:
                    continue
                all_paths_xy.append(_to_xy(p, image.shape[0], image.shape[1]))

    if not all_paths_xy:
        all_paths_xy = [_fallback_path_from_ink(ink)]

    ordered_paths = _order_paths(all_paths_xy)

    points: List[np.ndarray] = []
    pen: List[float] = []
    for p in ordered_paths:
        for xy in p:
            points.append(xy)
            pen.append(0.0)
        pen[-1] = 1.0

    pts = np.asarray(points, dtype=np.float32)
    pts = np.clip(pts, 0.0, 1.0)

    dxdy = np.zeros((len(pts), 2), dtype=np.float32)
    if len(pts) > 1:
        dxdy[1:] = np.diff(pts, axis=0)
    dxdy = np.clip(dxdy, -1.0, 1.0)

    seq = np.zeros((max_len, 3), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)

    usable = min(max_len, len(pts))
    if usable == 0:
        seq[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        mask[0] = 1.0
        return seq, mask

    seq[:usable, :2] = dxdy[:usable]
    seq[:usable, 2] = np.asarray(pen[:usable], dtype=np.float32)
    seq[usable - 1, 2] = 1.0
    mask[:usable] = 1.0
    return seq, mask


def convert_images_to_stroke_cache(
    images: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    max_len: int = 160,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> StrokeCache:
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if images.ndim != 3:
        raise ValueError("images must be [N, H, W]")

    n = images.shape[0]
    sequences = np.zeros((n, max_len, 3), dtype=np.float32)
    masks = np.zeros((n, max_len), dtype=np.float32)

    for i in range(n):
        seq, mask = image_to_stroke_sequence(images[i], max_len=max_len)
        sequences[i] = seq
        masks[i] = mask

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val :])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sequences=sequences,
        masks=masks,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        max_len=np.array([max_len], dtype=np.int64),
    )

    return StrokeCache(
        sequences=sequences,
        masks=masks,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def build_or_load_stroke_cache(
    data_dir: str | Path,
    out_path: str | Path,
    max_len: int = 160,
    seed: int = 42,
    download: bool = True,
) -> StrokeCache:
    out_path = Path(out_path)
    if out_path.exists():
        return load_stroke_cache(out_path)

    split_indices: SplitIndices = get_or_create_splits(data_dir=data_dir, seed=seed, download=download)
    dataset = EMNISTCombinedDataset(data_dir, normalize=False, download=download)

    n = len(dataset)
    sequences = np.zeros((n, max_len, 3), dtype=np.float32)
    masks = np.zeros((n, max_len), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)

    for idx in tqdm(range(n), desc="Converting EMNIST -> strokes"):
        image, label = dataset[idx]
        img_np = image.squeeze(0).numpy().astype(np.float32)
        seq, mask = image_to_stroke_sequence(img_np, max_len=max_len)
        sequences[idx] = seq
        masks[idx] = mask
        labels[idx] = int(label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sequences=sequences,
        masks=masks,
        labels=labels,
        train_idx=split_indices.train,
        val_idx=split_indices.val,
        test_idx=split_indices.test,
        max_len=np.array([max_len], dtype=np.int64),
    )
    return StrokeCache(
        sequences=sequences,
        masks=masks,
        labels=labels,
        train_idx=split_indices.train,
        val_idx=split_indices.val,
        test_idx=split_indices.test,
    )


def load_stroke_cache(path: str | Path) -> StrokeCache:
    payload = np.load(path)
    return StrokeCache(
        sequences=payload["sequences"],
        masks=payload["masks"],
        labels=payload["labels"],
        train_idx=payload["train_idx"],
        val_idx=payload["val_idx"],
        test_idx=payload["test_idx"],
    )


class StrokeSequenceDataset(Dataset):
    def __init__(self, cache_path: str | Path, split: str = "train") -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train/val/test")
        self.payload = np.load(cache_path, mmap_mode="r")
        self.sequences = self.payload["sequences"]
        self.masks = self.payload["masks"]
        self.labels = self.payload["labels"]
        self.indices = self.payload[f"{split}_idx"]

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        seq = torch.from_numpy(np.asarray(self.sequences[idx], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(self.masks[idx], dtype=np.float32))
        label = int(self.labels[idx])
        return seq, mask, label
