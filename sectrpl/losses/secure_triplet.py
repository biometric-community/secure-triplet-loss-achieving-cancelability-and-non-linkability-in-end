"""Secure Triplet Loss with cancelability + non-linkability terms (Eqs. 3, 4 and 6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between matching rows."""
    return (a - b).norm(dim=1)


def secure_triplet_loss(
    emb_a: torch.Tensor,
    emb_p1: torch.Tensor,
    emb_p2: torch.Tensor,
    emb_n1: torch.Tensor,
    emb_n2: torch.Tensor,
    margin: float = 1.0,
    gamma: float = 0.9,
) -> tuple[torch.Tensor, dict]:
    """
    emb_a  : anchor with key k1
    emb_p1 : positive (same ID) with key k1
    emb_p2 : positive (same ID) with key k2
    emb_n1 : negative ID with key k1
    emb_n2 : negative ID with key k2
    gamma  : balance between the triplet and linkability terms (SecureTL w/SL)
    """
    d_sp = pairwise_dist(emb_a, emb_p1)
    d_dp = pairwise_dist(emb_a, emb_p2)
    d_sn = pairwise_dist(emb_a, emb_n1)
    d_dn = pairwise_dist(emb_a, emb_n2)

    # Eq. (3): the closest of the three negatives sets the margin
    d_neg = torch.min(torch.min(d_sn, d_dp), d_dn)
    loss_stl = F.relu(margin + d_sp - d_neg).mean()

    # Eq. (6): key-mismatched distances should not reveal the identity
    loss_nl = (d_dp.mean() - d_dn.mean()).abs() + (d_dp.std() - d_dn.std()).abs()

    loss = gamma * loss_stl + (1.0 - gamma) * loss_nl

    stats = {
        "d_pos": float(d_sp.mean().detach()),
        "d_cancel": float(d_dp.mean().detach()),
        "d_imp": float(d_sn.mean().detach()),
        "d_dn": float(d_dn.mean().detach()),
        "loss_stl": float(loss_stl.detach()),
        "loss_nl": float(loss_nl.detach()),
    }
    return loss, stats
