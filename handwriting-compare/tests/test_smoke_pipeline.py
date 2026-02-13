from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REQUIRED_METRIC_KEYS = {
    "gan_classifier_confidence_mean",
    "gan_classifier_confidence_p80",
    "gan_class_entropy",
    "rnn_stroke_length_mean",
    "rnn_pen_lifts_mean",
    "rnn_smoothness_mean_abs_turn",
    "rnn_render_classifier_confidence_mean",
    "rnn_render_class_entropy",
}


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def test_smoke_end_to_end(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"

    classifier_dir = runs / "classifier"
    dcgan_dir = runs / "dcgan"
    rnn_dir = runs / "rnn"

    _run(
        [
            sys.executable,
            "-m",
            "src.train_classifier",
            "--out-dir",
            str(classifier_dir),
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--synthetic-smoke",
            "--num-workers",
            "0",
            "--max-train-batches",
            "2",
            "--max-val-batches",
            "1",
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "-m",
            "src.train_dcgan",
            "--out-dir",
            str(dcgan_dir),
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--synthetic-smoke",
            "--num-workers",
            "0",
            "--max-train-batches",
            "2",
            "--sample-interval",
            "1",
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "-m",
            "src.train_rnn",
            "--out-dir",
            str(rnn_dir),
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--synthetic-smoke",
            "--num-workers",
            "0",
            "--max-train-batches",
            "2",
            "--max-val-batches",
            "1",
            "--sample-interval",
            "1",
            "--max-len",
            "64",
        ],
        cwd=root,
    )

    gan_samples = reports / "samples_dcgan.png"
    rnn_samples = reports / "samples_rnn.png"

    _run(
        [
            sys.executable,
            "-m",
            "src.sample",
            "--model",
            "dcgan",
            "--ckpt",
            str(dcgan_dir / "best.pt"),
            "--num-samples",
            "16",
            "--out",
            str(gan_samples),
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "-m",
            "src.sample",
            "--model",
            "rnn",
            "--ckpt",
            str(rnn_dir / "best.pt"),
            "--num-samples",
            "16",
            "--out",
            str(rnn_samples),
            "--render",
            "--max-len",
            "64",
        ],
        cwd=root,
    )

    metrics_path = reports / "metrics.json"
    _run(
        [
            sys.executable,
            "-m",
            "src.eval",
            "--dcgan-ckpt",
            str(dcgan_dir / "best.pt"),
            "--rnn-ckpt",
            str(rnn_dir / "best.pt"),
            "--classifier-ckpt",
            str(classifier_dir / "best.pt"),
            "--num-samples",
            "64",
            "--batch-size",
            "16",
            "--out",
            str(metrics_path),
            "--max-len",
            "64",
        ],
        cwd=root,
    )

    assert (classifier_dir / "best.pt").exists()
    assert (dcgan_dir / "best.pt").exists()
    assert (rnn_dir / "best.pt").exists()
    assert gan_samples.exists()
    assert rnn_samples.exists()
    assert metrics_path.exists()

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert REQUIRED_METRIC_KEYS.issubset(metrics.keys())
