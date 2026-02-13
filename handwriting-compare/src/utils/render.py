from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image


def sequence_to_absolute_points(sequence: np.ndarray, start_xy: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Convert (dx, dy, pen) to absolute (x, y, pen) points in [0, 1]."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 3:
        raise ValueError("sequence must have shape [T, 3]")

    xy = np.zeros((seq.shape[0], 2), dtype=np.float32)
    xy[0] = np.asarray(start_xy, dtype=np.float32)
    if seq.shape[0] > 1:
        deltas = seq[1:, :2]
        xy[1:] = xy[0] + np.cumsum(deltas, axis=0)
    xy = np.clip(xy, 0.0, 1.0)
    pen = seq[:, 2:3]
    return np.concatenate([xy, pen], axis=1)


def render_sequence_to_image(sequence: np.ndarray, size: int = 28, line_width: int = 1) -> np.ndarray:
    """Render one sequence as a grayscale image in [0, 1]."""
    pts = sequence_to_absolute_points(sequence)
    canvas = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(canvas)

    for i in range(len(pts) - 1):
        x0, y0, pen = pts[i]
        x1, y1, _ = pts[i + 1]
        if pen >= 0.5:
            continue
        p0 = (int(round(x0 * (size - 1))), int(round(y0 * (size - 1))))
        p1 = (int(round(x1 * (size - 1))), int(round(y1 * (size - 1))))
        draw.line([p0, p1], fill=255, width=line_width)

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr


def render_sequences_to_tensor(sequences: np.ndarray | torch.Tensor, size: int = 28) -> torch.Tensor:
    """Render a batch of (dx, dy, pen) sequences to [N,1,H,W] in [-1,1]."""
    if isinstance(sequences, torch.Tensor):
        seq_np = sequences.detach().cpu().numpy()
    else:
        seq_np = np.asarray(sequences)

    images: List[np.ndarray] = []
    for seq in seq_np:
        images.append(render_sequence_to_image(seq, size=size, line_width=1))
    stacked = np.stack(images, axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(1).float()
    return tensor * 2.0 - 1.0


def save_tensor_grid(images: torch.Tensor, out_path: str | Path, nrow: int = 8) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, out_path.as_posix(), nrow=nrow, normalize=True, value_range=(-1, 1))


def plot_stroke_sequences(sequences: Iterable[np.ndarray], out_path: str | Path, cols: int = 4) -> None:
    seqs = list(sequences)
    if not seqs:
        return
    rows = int(np.ceil(len(seqs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if idx >= len(seqs):
            continue
        pts = sequence_to_absolute_points(np.asarray(seqs[idx]))
        current_x = [pts[0, 0]]
        current_y = [pts[0, 1]]
        for i in range(len(pts) - 1):
            x0, y0, pen = pts[i]
            x1, y1, _ = pts[i + 1]
            current_x.append(x1)
            current_y.append(y1)
            if pen >= 0.5:
                ax.plot(current_x, current_y, color="black", linewidth=1)
                current_x = [x1]
                current_y = [y1]
        if len(current_x) > 1:
            ax.plot(current_x, current_y, color="black", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
