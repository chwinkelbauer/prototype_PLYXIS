"""Learned prices for the decomposition (colgen.py): the learning layer of the agreed plan, phase 5.

Column generation finds the prices of the shared limits (gas per hour, a mill, a fleet) by repeated pricing;
it needs fewer rounds the closer it starts. A small network, trained locally on the prices column generation
itself found (no external model, no external data), predicts them from the situation:

    per hour t:   x_t = (limit / what all assets could use at full rate, value per unit of the shared resource
                         at full rate, the same relative to the horizon's mean, hour of day as sin / cos)
    output:       pi_t / s_t >= 0       s_t = value per unit of the resource at full rate (scale-free)

Predictions only seed column generation (`colgen.solve(warm_pi=...)`): a poor prediction costs rounds, never
correctness or the bound. Learning reduces the solve time; the spread of plans between runs is already gone,
because column generation and its DP are deterministic.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .colgen import UnitFamily


def features(fam: UnitFamily, cap: np.ndarray) -> tuple:
    """Per hour features [T, F] and the scale s [R, T] of the prices."""
    R, T = cap.shape
    top_use = fam.use[:, :, -1, :].sum(0).T                         # [R, T] what all assets could use at full rate
    top_val = fam.value[:, :, -1].sum(0)                            # [T]
    s = np.maximum(top_val[None, :] / np.maximum(top_use, 1e-9), 1e-9)
    tight = cap / np.maximum(top_use, 1e-9)
    h = np.arange(T) % 24
    f = np.concatenate([tight.T, (s / s.mean(1, keepdims=True)).T, np.log(s).T - np.log(s).mean(),
                        np.sin(2 * math.pi * h / 24)[:, None], np.cos(2 * math.pi * h / 24)[:, None]], 1)
    return f.astype(np.float32), s


class PriceNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, n_out))
        # hours see their neighbours through a small temporal convolution of the features
        self.mix = nn.Conv1d(n_in, n_in, 5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [B, T, F] -> [B, T, R] (scaled prices >= 0)
        x = x + self.mix(x.transpose(1, 2)).transpose(1, 2)
        return nn.functional.softplus(self.net(x))


def fit(samples: list, epochs: int = 600, lr: float = 3e-3, seed: int = 0) -> PriceNet:
    """samples: (features [T, F], scaled prices [T, R]) from column generation's own solutions."""
    torch.manual_seed(seed)
    X = torch.tensor(np.stack([s[0] for s in samples]))
    Y = torch.tensor(np.stack([s[1] for s in samples]), dtype=torch.float32)
    net = PriceNet(X.shape[2], Y.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    net.train_loss = float(loss.detach())
    return net


def target(fam: UnitFamily, cap: np.ndarray, pi: np.ndarray) -> tuple:
    f, s = features(fam, cap)
    return f, (pi / s).T.astype(np.float32)


@torch.no_grad()
def predict(net: PriceNet, fam: UnitFamily, cap: np.ndarray) -> np.ndarray:
    f, s = features(fam, cap)
    y = net(torch.tensor(f)[None])[0].numpy().T                     # [R, T]
    return y * s
