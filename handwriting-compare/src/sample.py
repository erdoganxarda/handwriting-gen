from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.dcgan import Generator
from src.models.rnn_mdn import RNNMDN, generate_unconditional
from src.utils.io import ensure_dir
from src.utils.render import plot_stroke_sequences, render_sequences_to_tensor, save_tensor_grid
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples from trained handwriting models.")
    parser.add_argument("--model", choices=["dcgan", "rnn"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


def _sample_dcgan(args: argparse.Namespace, device: torch.device) -> None:
    ckpt = _load_checkpoint(args.ckpt, device)
    latent_dim = int(ckpt.get("config", {}).get("latent_dim", args.latent_dim))

    model = Generator(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["generator_state"])
    model.eval()

    with torch.no_grad():
        z = torch.randn(args.num_samples, latent_dim, device=device)
        images = model(z)

    save_tensor_grid(images.cpu(), args.out, nrow=int(np.sqrt(args.num_samples)) or 8)

    # Save a simple latent interpolation strip.
    steps = 10
    z0 = torch.randn(1, latent_dim, device=device)
    z1 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps=steps, device=device).view(-1, 1)
    z_interp = (1.0 - alphas) * z0 + alphas * z1
    with torch.no_grad():
        interp_images = model(z_interp)

    out_path = Path(args.out)
    interp_path = out_path.with_name(f"{out_path.stem}_interp{out_path.suffix}")
    save_tensor_grid(interp_images.cpu(), interp_path, nrow=steps)


def _sample_rnn(args: argparse.Namespace, device: torch.device) -> None:
    ckpt = _load_checkpoint(args.ckpt, device)
    cfg = ckpt.get("config", {})

    model = RNNMDN(
        input_dim=3,
        hidden_size=int(cfg.get("hidden_size", 256)),
        num_layers=int(cfg.get("layers", 2)),
        num_mixtures=int(cfg.get("mixtures", 20)),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    max_len = int(cfg.get("max_len", args.max_len))
    with torch.no_grad():
        sequences = generate_unconditional(
            model=model,
            num_samples=args.num_samples,
            max_len=max_len,
            device=device,
            temperature=args.temperature,
        )

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(npz_path, sequences=sequences.cpu().numpy())

    if args.render:
        images = render_sequences_to_tensor(sequences.cpu().numpy())
        save_tensor_grid(images, out_path, nrow=int(np.sqrt(args.num_samples)) or 8)

    stroke_path = out_path.with_name(f"{out_path.stem}_strokes{out_path.suffix}")
    plot_stroke_sequences([sequences[i].cpu().numpy() for i in range(min(16, args.num_samples))], stroke_path, cols=4)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "dcgan":
        _sample_dcgan(args, device)
    else:
        _sample_rnn(args, device)


if __name__ == "__main__":
    main()
