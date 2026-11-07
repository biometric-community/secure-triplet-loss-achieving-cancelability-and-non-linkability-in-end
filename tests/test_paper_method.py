"""The loss, keys and metrics, checked against Sec. III-IV and the authors' losses.py."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sectrpl.eval import best_acc_eer, eval_linkability
from sectrpl.losses.secure_triplet import secure_triplet_loss
from sectrpl.models.keyed_net import KeyedFaceNet, sample_keys

ROOT = Path(__file__).resolve().parents[1]


def embeddings_taken() -> int:
    params = inspect.signature(secure_triplet_loss).parameters.values()
    return sum(p.default is inspect.Parameter.empty for p in params)


def repo_loss(templates, margin, gamma) -> float:
    return secure_triplet_loss(*templates[: embeddings_taken()], margin=margin, gamma=gamma)[0].item()


def paper_loss(templates, margin, gamma) -> float:
    """Eqs. (3), (4) and (6), as in SecureTripletLossSL of the authors' losses.py."""
    a, p1, p2, n1, n2 = templates
    d_sp, d_dp = (a - p1).norm(dim=1), (a - p2).norm(dim=1)
    d_sn, d_dn = (a - n1).norm(dim=1), (a - n2).norm(dim=1)
    negative = torch.min(torch.min(d_sn, d_dp), d_dn)
    stl = F.relu(margin + d_sp - negative).mean()
    linkability = (d_dp.mean() - d_dn.mean()).abs() + (d_dp.std() - d_dn.std()).abs()
    return (gamma * stl + (1 - gamma) * linkability).item()


@pytest.fixture
def templates():
    torch.manual_seed(0)
    return tuple(F.normalize(torch.randn(16, 32), dim=1) for _ in range(5))


def test_loss_takes_the_five_templates_of_sec_iii_b():
    """yA, yP1, yP2, yN1 and yN2: dDN needs a non-mated template under the second key."""
    assert embeddings_taken() == 5


def test_gamma_one_leaves_only_eq_3(templates):
    assert repo_loss(templates, 1.0, 1.0) == pytest.approx(paper_loss(templates, 1.0, 1.0), abs=1e-4)


def test_loss_matches_eqs_4_and_6(templates):
    assert repo_loss(templates, 1.0, 0.9) == pytest.approx(paper_loss(templates, 1.0, 0.9), abs=1e-4)


@pytest.mark.parametrize("key,section", [("gamma", "train"), ("margin", "train"), ("lr", "train"), ("batch_size", "train")])
def test_training_hyperparameters_match_the_paper(default_cfg, targets, key, section):
    assert default_cfg[section][key] == targets["hyperparameters"][key]


def test_key_length_matches_the_paper(default_cfg, targets):
    assert default_cfg["model"]["key_dim"] == targets["hyperparameters"]["key_dim"]


def test_training_uses_adam():
    assert "optim.Adam" in (ROOT / "sectrpl" / "train.py").read_text(encoding="utf-8")


def test_keys_have_unit_norm():
    """Sec. IV: each key is processed to verify unit l2 norm."""
    keys = sample_keys(8, 100, torch.device("cpu"))
    assert torch.allclose(keys.norm(dim=1), torch.ones(8), atol=1e-5)


def test_keys_are_binary():
    assert len(sample_keys(8, 100, torch.device("cpu")).unique()) == 2


def test_key_changes_the_template():
    torch.manual_seed(1)
    model = KeyedFaceNet(key_dim=16, emb_dim=8, pretrained=False).eval()
    x = torch.randn(2, 3, 64, 64)
    k1, k2 = sample_keys(2, 16, torch.device("cpu")), sample_keys(2, 16, torch.device("cpu"))
    with torch.no_grad():
        assert not torch.allclose(model(x, k1), model(x, k2))


class RecordingModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, x, key):
        self.calls.append((x.clone(), key.clone()))
        return F.normalize(x.flatten(1), dim=1)


@pytest.fixture
def linkability_calls():
    torch.manual_seed(2)
    images = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    pairs = [(images[0][0], images[1][0], 1), (images[0][1], images[1][1], 0)]
    model = RecordingModel()
    eval_linkability(model, pairs, torch.device("cpu"), 8, 2, 0)
    return model.calls


def test_linkability_non_mated_pairs_use_different_keys(linkability_calls):
    """Sec. IV-C: non-mated instances have different identities and different keys."""
    (_, key_a), (_, key_b) = linkability_calls[-2:]
    assert not torch.equal(key_a, key_b)


def test_linkability_mated_pairs_are_different_samples(linkability_calls):
    """Sec. IV-C: mated instances are samples from the same identity, not one image."""
    (image_a, _), (image_b, _) = linkability_calls[:2]
    assert not torch.equal(image_a, image_b)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_eer_is_the_point_where_fmr_equals_fnmr():
    lo, hi = 0.0, 1.0
    for _ in range(80):
        t = 0.5 * (lo + hi)
        fmr, fnmr = 1.0 - _norm_cdf(t), _norm_cdf((t - 1.0) / 0.1)
        lo, hi = (t, hi) if fmr > fnmr else (lo, t)
    expected = 1.0 - _norm_cdf(t)

    rng = np.random.RandomState(3)
    scores = np.concatenate([rng.normal(1.0, 0.1, 200000), rng.normal(0.0, 1.0, 200000)])
    labels = np.concatenate([np.ones(200000, dtype=int), np.zeros(200000, dtype=int)])
    _, eer = best_acc_eer(scores, labels)
    assert eer == pytest.approx(expected, abs=0.01)
