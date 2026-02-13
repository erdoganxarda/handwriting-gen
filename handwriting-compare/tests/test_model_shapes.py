from __future__ import annotations

import torch

from src.models.dcgan import Discriminator, Generator
from src.models.rnn_mdn import RNNMDN, generate_unconditional, mdn_nll


def test_dcgan_shapes() -> None:
    gen = Generator(latent_dim=100)
    dis = Discriminator(dropout=0.2)

    z = torch.randn(4, 100)
    fake = gen(z)
    score = dis(fake)

    assert fake.shape == (4, 1, 28, 28)
    assert score.shape == (4, 1)


def test_rnn_mdn_shapes_and_nll() -> None:
    model = RNNMDN(input_dim=3, hidden_size=128, num_layers=2, num_mixtures=10)
    x = torch.randn(3, 12, 3)
    params, _ = model(x)

    assert params["pi_logits"].shape == (3, 12, 10)
    assert params["mu_x"].shape == (3, 12, 10)
    assert params["mu_y"].shape == (3, 12, 10)
    assert params["pen_logit"].shape == (3, 12)

    target_xy = torch.randn(3, 12, 2)
    mask = torch.ones(3, 12)
    loss = mdn_nll(params, target_xy, mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rnn_sampling_shape() -> None:
    model = RNNMDN(input_dim=3, hidden_size=64, num_layers=1, num_mixtures=5)
    seq = generate_unconditional(
        model=model,
        num_samples=5,
        max_len=40,
        device=torch.device("cpu"),
    )

    assert seq.shape == (5, 40, 3)
    assert torch.all(seq[..., :2] <= 1.0)
    assert torch.all(seq[..., :2] >= -1.0)
