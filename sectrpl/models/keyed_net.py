"""Face embedder conditioned on a cancelable key (FiLM fusion)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class KeyedFaceNet(nn.Module):
    def __init__(self, key_dim: int = 64, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.key_dim = key_dim
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        # FiLM: key → (scale, shift) on 512-d features
        self.key_mlp = nn.Sequential(
            nn.Linear(key_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512 * 2),
        )
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        f = torch.flatten(self.features(x), 1)
        film = self.key_mlp(key)
        scale, shift = film.chunk(2, dim=1)
        f = f * (1.0 + torch.tanh(scale)) + shift
        z = self.bn(self.fc(f))
        return F.normalize(z, p=2, dim=1)


def sample_keys(n: int, key_dim: int, device: torch.device) -> torch.Tensor:
    """Random bipolar keys in {-1, +1}^d (BioHash-style)."""
    return torch.randint(0, 2, (n, key_dim), device=device, dtype=torch.float32) * 2 - 1
