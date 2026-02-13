from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNMDN(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_mixtures: int = 20,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_mixtures * 6 + 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        rnn_out, hidden = self.rnn(x, hidden)
        raw = self.head(rnn_out)

        k = self.num_mixtures
        pi_logits = raw[..., :k]
        mu_x = raw[..., k : 2 * k]
        mu_y = raw[..., 2 * k : 3 * k]
        log_sigma_x = raw[..., 3 * k : 4 * k].clamp(-7.0, 2.0)
        log_sigma_y = raw[..., 4 * k : 5 * k].clamp(-7.0, 2.0)
        rho = torch.tanh(raw[..., 5 * k : 6 * k]) * 0.95
        pen_logit = raw[..., -1]

        params = {
            "pi_logits": pi_logits,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "log_sigma_x": log_sigma_x,
            "log_sigma_y": log_sigma_y,
            "rho": rho,
            "pen_logit": pen_logit,
        }
        return params, hidden


def mdn_nll(
    params: Dict[str, torch.Tensor],
    target_xy: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    x = target_xy[..., 0].unsqueeze(-1)
    y = target_xy[..., 1].unsqueeze(-1)

    log_pi = F.log_softmax(params["pi_logits"], dim=-1)
    sigma_x = torch.exp(params["log_sigma_x"]).clamp_min(eps)
    sigma_y = torch.exp(params["log_sigma_y"]).clamp_min(eps)
    rho = params["rho"].clamp(-0.999, 0.999)

    norm_x = (x - params["mu_x"]) / sigma_x
    norm_y = (y - params["mu_y"]) / sigma_y

    z = norm_x * norm_x + norm_y * norm_y - 2.0 * rho * norm_x * norm_y
    one_minus_rho2 = (1.0 - rho * rho).clamp_min(eps)

    log_normalizer = (
        -math.log(2.0 * math.pi)
        - params["log_sigma_x"]
        - params["log_sigma_y"]
        - 0.5 * torch.log(one_minus_rho2)
    )
    log_component = log_normalizer - z / (2.0 * one_minus_rho2)

    log_prob = torch.logsumexp(log_pi + log_component, dim=-1)
    nll = -log_prob

    if mask is not None:
        denom = mask.sum().clamp_min(1.0)
        return (nll * mask).sum() / denom
    return nll.mean()


def pen_bce_loss(pen_logit: torch.Tensor, target_pen: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(pen_logit, target_pen, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (bce * mask).sum() / denom


def sample_mdn_step(
    params: Dict[str, torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample one time-step from MDN params, returns [B, 3] => (dx, dy, pen)."""
    pi_logits = params["pi_logits"][:, -1, :] / max(temperature, 1e-4)
    pi = F.softmax(pi_logits, dim=-1)
    comp = torch.multinomial(pi, num_samples=1).squeeze(-1)

    batch_idx = torch.arange(comp.shape[0], device=comp.device)
    mu_x = params["mu_x"][:, -1, :][batch_idx, comp]
    mu_y = params["mu_y"][:, -1, :][batch_idx, comp]
    sigma_x = torch.exp(params["log_sigma_x"][:, -1, :][batch_idx, comp])
    sigma_y = torch.exp(params["log_sigma_y"][:, -1, :][batch_idx, comp])
    rho = params["rho"][:, -1, :][batch_idx, comp].clamp(-0.999, 0.999)

    eps1 = torch.randn_like(mu_x) * temperature
    eps2 = torch.randn_like(mu_y) * temperature

    dx = mu_x + sigma_x * eps1
    dy = mu_y + sigma_y * (rho * eps1 + torch.sqrt(1.0 - rho * rho) * eps2)

    pen_prob = torch.sigmoid(params["pen_logit"][:, -1] / max(temperature, 1e-4))
    pen = torch.bernoulli(pen_prob)

    out = torch.stack([dx, dy, pen], dim=-1)
    out[..., :2] = out[..., :2].clamp(-1.0, 1.0)
    return out


@torch.no_grad()
def generate_unconditional(
    model: RNNMDN,
    num_samples: int,
    max_len: int,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    model.eval()
    hidden = None
    current = torch.zeros((num_samples, 1, 3), device=device)
    seq = torch.zeros((num_samples, max_len, 3), device=device)

    for t in range(max_len):
        params, hidden = model(current, hidden)
        step = sample_mdn_step(params, temperature=temperature)
        seq[:, t, :] = step
        current = step.unsqueeze(1)

    seq[:, -1, 2] = 1.0
    return seq
