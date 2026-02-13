from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.emnist_to_strokes import (
    StrokeSequenceDataset,
    build_or_load_stroke_cache,
)
from src.models.rnn_mdn import RNNMDN, generate_unconditional, mdn_nll, pen_bce_loss
from src.utils.io import append_metrics_csv, ensure_dir, save_json
from src.utils.render import plot_stroke_sequences, render_sequences_to_tensor, save_tensor_grid
from src.utils.seed import set_global_seed
from src.utils.synthetic import make_fake_stroke_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM+MDN on EMNIST-derived stroke sequences.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--strokes-path",
        type=str,
        default="data/processed/emnist_letters_strokes_len160_seed42.npz",
    )
    parser.add_argument("--out-dir", type=str, default="runs/rnn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mixtures", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def make_stroke_dataloaders(
    cache_path: str | Path,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    train_ds = StrokeSequenceDataset(cache_path, split="train")
    val_ds = StrokeSequenceDataset(cache_path, split="val")
    test_ds = StrokeSequenceDataset(cache_path, split="test")

    gen = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            generator=gen,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


def _run_epoch(
    model: RNNMDN,
    loader: DataLoader,
    device: torch.device,
    optimizer: Adam | None = None,
    grad_clip: float = 1.0,
    max_batches: int = 0,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_nll = 0.0
    total_pen = 0.0
    steps = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (seq, mask, _) in enumerate(loader):
            seq = seq.to(device)
            mask = mask.to(device)

            input_seq = seq[:, :-1, :]
            target_seq = seq[:, 1:, :]
            target_mask = mask[:, 1:]

            params, _ = model(input_seq)
            nll = mdn_nll(params, target_seq[..., :2], target_mask)
            pen_loss = pen_bce_loss(params["pen_logit"], target_seq[..., 2], target_mask)
            loss = nll + pen_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.item())
            total_nll += float(nll.item())
            total_pen += float(pen_loss.item())
            steps += 1

            if max_batches > 0 and (batch_idx + 1) >= max_batches:
                break

    denom = max(steps, 1)
    return total_loss / denom, total_nll / denom, total_pen / denom


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.out_dir)
    save_json(out_dir / "config.json", vars(args))

    if args.synthetic_smoke:
        loaders = make_fake_stroke_loaders(
            batch_size=args.batch_size,
            max_len=args.max_len,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    else:
        stroke_path = Path(args.strokes_path)
        if not stroke_path.exists():
            build_or_load_stroke_cache(
                data_dir=args.data_dir,
                out_path=stroke_path,
                max_len=args.max_len,
                seed=args.seed,
                download=True,
            )
        loaders = make_stroke_dataloaders(
            cache_path=stroke_path,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )

    model = RNNMDN(
        input_dim=3,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_mixtures=args.mixtures,
        dropout=args.dropout,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_nll, train_pen = _run_epoch(
            model,
            loaders["train"],
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
        )
        val_loss, val_nll, val_pen = _run_epoch(
            model,
            loaders["val"],
            device=device,
            optimizer=None,
            grad_clip=args.grad_clip,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_nll": round(train_nll, 6),
            "train_pen": round(train_pen, 6),
            "val_loss": round(val_loss, 6),
            "val_nll": round(val_nll, 6),
            "val_pen": round(val_pen, 6),
        }
        append_metrics_csv(out_dir / "metrics.csv", row)
        tqdm.write(f"[RNN] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": vars(args),
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, out_dir / "best.pt")

        if epoch % args.sample_interval == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            samples = generate_unconditional(
                model=model,
                num_samples=64,
                max_len=args.max_len,
                device=device,
            ).cpu()
            render = render_sequences_to_tensor(samples.numpy())
            save_tensor_grid(render, out_dir / f"samples_epoch{epoch:03d}.png", nrow=8)
            plot_stroke_sequences(
                [samples[i].numpy() for i in range(16)],
                out_dir / f"strokes_epoch{epoch:03d}.png",
                cols=4,
            )


if __name__ == "__main__":
    main()
