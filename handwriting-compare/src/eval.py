from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.classifier_cnn import ClassifierCNN
from src.models.dcgan import Generator
from src.models.rnn_mdn import RNNMDN, generate_unconditional
from src.utils.io import ensure_dir, save_metrics_rows
from src.utils.render import render_sequences_to_tensor
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAN vs RNN handwriting generators.")
    parser.add_argument("--dcgan-ckpt", type=str, required=True)
    parser.add_argument("--rnn-ckpt", type=str, required=True)
    parser.add_argument("--classifier-ckpt", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_classifier(path: str | Path, device: torch.device) -> ClassifierCNN:
    ckpt = torch.load(path, map_location=device)
    model = ClassifierCNN(num_classes=26).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_dcgan(path: str | Path, device: torch.device) -> tuple[Generator, int]:
    ckpt = torch.load(path, map_location=device)
    latent_dim = int(ckpt.get("config", {}).get("latent_dim", 100))
    model = Generator(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["generator_state"])
    model.eval()
    return model, latent_dim


def _load_rnn(path: str | Path, device: torch.device, max_len_default: int) -> tuple[RNNMDN, int]:
    ckpt = torch.load(path, map_location=device)
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
    max_len = int(cfg.get("max_len", max_len_default))
    return model, max_len


def _class_entropy(pred_counts: np.ndarray) -> float:
    probs = pred_counts.astype(np.float64)
    probs = probs / max(probs.sum(), 1.0)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs + 1e-12)).sum())


def _confidence_metrics(confidences: list[float]) -> tuple[float, float]:
    if not confidences:
        return 0.0, 0.0
    arr = np.asarray(confidences, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 80))


def _sequence_smoothness(seq: np.ndarray) -> float:
    v = seq[:, :2]
    if len(v) < 3:
        return 0.0
    v1 = v[:-1]
    v2 = v[1:]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-5) & (n2 > 1e-5)
    if not np.any(valid):
        return 0.0
    dot = np.sum(v1[valid] * v2[valid], axis=1)
    cosang = np.clip(dot / (n1[valid] * n2[valid] + 1e-8), -1.0, 1.0)
    turns = np.abs(np.arccos(cosang))
    return float(turns.mean()) if turns.size else 0.0


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    classifier = _load_classifier(args.classifier_ckpt, device)
    dcgan, latent_dim = _load_dcgan(args.dcgan_ckpt, device)
    rnn, rnn_max_len = _load_rnn(args.rnn_ckpt, device, args.max_len)

    # GAN metrics.
    gan_conf: list[float] = []
    gan_pred_counts = np.zeros((26,), dtype=np.int64)

    with torch.no_grad():
        remaining = args.num_samples
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            z = torch.randn((batch, latent_dim), device=device)
            images = dcgan(z)
            logits = classifier(images)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            gan_conf.extend(conf.detach().cpu().numpy().tolist())
            bincount = torch.bincount(pred, minlength=26).detach().cpu().numpy()
            gan_pred_counts += bincount
            remaining -= batch

    gan_conf_mean, gan_conf_p80 = _confidence_metrics(gan_conf)
    gan_entropy = _class_entropy(gan_pred_counts)

    # RNN metrics.
    rnn_conf: list[float] = []
    rnn_pred_counts = np.zeros((26,), dtype=np.int64)
    stroke_lengths: list[float] = []
    pen_lifts: list[float] = []
    smoothness_vals: list[float] = []

    with torch.no_grad():
        remaining = args.num_samples
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            seq = generate_unconditional(
                model=rnn,
                num_samples=batch,
                max_len=rnn_max_len,
                device=device,
                temperature=args.temperature,
            )
            seq_np = seq.detach().cpu().numpy()

            lengths = np.linalg.norm(seq_np[..., :2], axis=-1).sum(axis=1)
            lifts = (seq_np[..., 2] > 0.5).sum(axis=1)
            smooth = [_sequence_smoothness(s) for s in seq_np]

            stroke_lengths.extend(lengths.tolist())
            pen_lifts.extend(lifts.astype(np.float64).tolist())
            smoothness_vals.extend(smooth)

            rendered = render_sequences_to_tensor(seq_np).to(device)
            logits = classifier(rendered)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            rnn_conf.extend(conf.detach().cpu().numpy().tolist())
            bincount = torch.bincount(pred, minlength=26).detach().cpu().numpy()
            rnn_pred_counts += bincount
            remaining -= batch

    rnn_conf_mean, _ = _confidence_metrics(rnn_conf)
    rnn_entropy = _class_entropy(rnn_pred_counts)

    metrics = {
        "gan_classifier_confidence_mean": float(gan_conf_mean),
        "gan_classifier_confidence_p80": float(gan_conf_p80),
        "gan_class_entropy": float(gan_entropy),
        "rnn_stroke_length_mean": float(np.mean(stroke_lengths) if stroke_lengths else 0.0),
        "rnn_pen_lifts_mean": float(np.mean(pen_lifts) if pen_lifts else 0.0),
        "rnn_smoothness_mean_abs_turn": float(np.mean(smoothness_vals) if smoothness_vals else 0.0),
        "rnn_render_classifier_confidence_mean": float(rnn_conf_mean),
        "rnn_render_class_entropy": float(rnn_entropy),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    csv_path = out_path.with_suffix(".csv")
    save_metrics_rows(csv_path, [metrics])

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
