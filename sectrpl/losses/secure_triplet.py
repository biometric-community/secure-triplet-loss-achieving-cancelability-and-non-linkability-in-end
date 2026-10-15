"""Secure Triplet Loss with cancelability + non-linkability terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean on L2-normalized vectors (= 2 - 2 cos)."""
    return ((a - b) ** 2).sum(dim=1)


def secure_triplet_loss(
    emb_ak: torch.Tensor,
    emb_pk: torch.Tensor,
    emb_ak2: torch.Tensor,
    emb_nk: torch.Tensor,
    margin: float = 0.5,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    emb_ak  : anchor with key k
    emb_pk  : positive (same ID) with same key k
    emb_ak2 : anchor with different key k'  (cancelability negative)
    emb_nk  : negative ID with key k
    gamma   : weight for non-linkability term
    """
    d_pos = pairwise_dist(emb_ak, emb_pk)
    d_cancel = pairwise_dist(emb_ak, emb_ak2)
    d_imp = pairwise_dist(emb_ak, emb_nk)

    loss_c = F.relu(d_pos - d_cancel + margin).mean()
    loss_i = F.relu(d_pos - d_imp + margin).mean()

    # Non-linkability: same-ID different-key distances should resemble impostors
    loss_nl = (d_cancel - d_imp.detach()).pow(2).mean()
    # Extra cancelability: explicitly maximize d_cancel (cap at 2 for unit sphere)
    loss_c_aux = F.relu(1.5 - d_cancel).mean()

    loss = loss_c + loss_i + gamma * loss_nl + 0.5 * loss_c_aux

    stats = {
        "d_pos": float(d_pos.mean().detach()),
        "d_cancel": float(d_cancel.mean().detach()),
        "d_imp": float(d_imp.mean().detach()),
        "loss_c": float(loss_c.detach()),
        "loss_i": float(loss_i.detach()),
        "loss_nl": float(loss_nl.detach()),
    }
    return loss, stats
