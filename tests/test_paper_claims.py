"""The paper's central claim, reproduced with this repo's loss on synthetic features.

Sec. III: templates should match only when both identity and key match.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sectrpl.losses.secure_triplet import secure_triplet_loss
from sectrpl.models.keyed_net import sample_keys

DIM, KEY, IDS = 32, 32, 40


class KeyedHead(nn.Module):
    """Sec. IV-B: features concatenated with the key, then two FC layers."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM + KEY, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x, key):
        return F.normalize(self.net(torch.cat([x, key], dim=1)), dim=1)


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    centers = torch.tensor(rng.normal(size=(IDS, DIM)), dtype=torch.float32)
    device = torch.device("cpu")
    n_args = sum(
        p.default is inspect.Parameter.empty for p in inspect.signature(secure_triplet_loss).parameters.values()
    )

    def batch():
        a = torch.randint(0, IDS, (64,))
        n = (a + torch.randint(1, IDS, (64,))) % IDS
        sample = lambda ids: centers[ids] + 0.2 * torch.randn(64, DIM)
        return sample(a), sample(a), sample(n)

    model = KeyedHead()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(400):
        xa, xp, xn = batch()
        k1, k2 = sample_keys(64, KEY, device), sample_keys(64, KEY, device)
        templates = (model(xa, k1), model(xp, k1), model(xp, k2), model(xn, k1), model(xn, k2))
        loss, _ = secure_triplet_loss(*templates[:n_args])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, batch, device


def distances(trained):
    model, batch, device = trained
    xa, xp, xn = batch()
    k1, k2 = sample_keys(64, KEY, device), sample_keys(64, KEY, device)
    with torch.no_grad():
        ya = model(xa, k1)
        return {
            "same_id_same_key": (ya - model(xp, k1)).norm(dim=1).mean().item(),
            "same_id_diff_key": (ya - model(xp, k2)).norm(dim=1).mean().item(),
            "diff_id_same_key": (ya - model(xn, k1)).norm(dim=1).mean().item(),
            "diff_id_diff_key": (ya - model(xn, k2)).norm(dim=1).mean().item(),
        }


def test_templates_match_only_with_identity_and_key(trained):
    d = distances(trained)
    assert d["same_id_same_key"] < min(d["same_id_diff_key"], d["diff_id_same_key"], d["diff_id_diff_key"])


def test_key_mismatch_hides_identity(trained):
    """Sec. III-C: dDP and dDN should assume similar values.

    The authors' loss, Eqs. (3), (4) and (6), reaches a gap of about 0.14 on this task.
    """
    d = distances(trained)
    assert abs(d["same_id_diff_key"] - d["diff_id_diff_key"]) <= 0.2
