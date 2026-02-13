from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm

from src.datasets.emnist import make_emnist_dataloaders
from src.models.classifier_cnn import ClassifierCNN
from src.utils.io import append_metrics_csv, ensure_dir, save_json
from src.utils.seed import set_global_seed
from src.utils.synthetic import make_fake_emnist_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EMNIST letters classifier.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="runs/classifier")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def _run_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int = 0,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            pred = logits.argmax(dim=1)
            total_correct += int((pred == labels).sum().item())
            total_count += int(labels.size(0))
            total_loss += float(loss.item()) * labels.size(0)

            if max_batches > 0 and (batch_idx + 1) >= max_batches:
                break

    mean_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return mean_loss, acc


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(args.out_dir)
    save_json(out_dir / "config.json", vars(args))

    if args.synthetic_smoke:
        loaders = make_fake_emnist_loaders(batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers)
    else:
        loaders = make_emnist_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            download=True,
        )

    model = ClassifierCNN(num_classes=26).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(
            model,
            loaders["train"],
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = _run_epoch(
            model,
            loaders["val"],
            device=device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        append_metrics_csv(out_dir / "metrics.csv", row)
        tqdm.write(f"[Classifier] epoch={epoch} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "config": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, out_dir / "best.pt")


if __name__ == "__main__":
    main()
