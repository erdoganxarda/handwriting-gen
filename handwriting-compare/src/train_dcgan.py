from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from src.datasets.emnist import make_emnist_dataloaders
from src.models.dcgan import Discriminator, Generator, weights_init
from src.utils.io import append_metrics_csv, ensure_dir, save_json
from src.utils.render import save_tensor_grid
from src.utils.seed import set_global_seed
from src.utils.synthetic import make_fake_emnist_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCGAN on EMNIST letters.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="runs/dcgan")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


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

    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator(dropout=0.2).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    opt_g = Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = nn.BCELoss()

    fixed_z = torch.randn(64, args.latent_dim, device=device)
    best_g_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        g_loss_running = 0.0
        d_loss_running = 0.0
        steps = 0

        progress = tqdm(loaders["train"], desc=f"DCGAN epoch {epoch}")
        for batch_idx, (real_images, _) in enumerate(progress):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # Train discriminator.
            opt_d.zero_grad(set_to_none=True)
            z = torch.randn((batch_size, args.latent_dim), device=device)
            fake_images = generator(z)
            real_loss = criterion(discriminator(real_images), valid)
            fake_loss = criterion(discriminator(fake_images.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            opt_d.step()

            # Train generator.
            opt_g.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake_images), valid)
            g_loss.backward()
            opt_g.step()

            steps += 1
            g_loss_running += float(g_loss.item())
            d_loss_running += float(d_loss.item())
            progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

            if args.max_train_batches > 0 and (batch_idx + 1) >= args.max_train_batches:
                break

        mean_g = g_loss_running / max(steps, 1)
        mean_d = d_loss_running / max(steps, 1)
        append_metrics_csv(
            out_dir / "metrics.csv",
            {
                "epoch": epoch,
                "g_loss": round(mean_g, 6),
                "d_loss": round(mean_d, 6),
            },
        )

        checkpoint = {
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "opt_g_state": opt_g.state_dict(),
            "opt_d_state": opt_d.state_dict(),
            "g_loss": mean_g,
            "d_loss": mean_d,
            "config": vars(args),
        }
        torch.save(checkpoint, out_dir / "last.pt")

        if mean_g < best_g_loss:
            best_g_loss = mean_g
            torch.save(checkpoint, out_dir / "best.pt")

        if epoch % args.sample_interval == 0 or epoch == 1 or epoch == args.epochs:
            generator.eval()
            with torch.no_grad():
                sample_images = generator(fixed_z)
            save_tensor_grid(sample_images, out_dir / f"samples_epoch{epoch:03d}.png", nrow=8)


if __name__ == "__main__":
    main()
