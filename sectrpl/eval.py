"""LFW utility + cancelability + non-linkability metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sectrpl.data.pairs import build_calfw, build_cplfw, build_lfw
from sectrpl.models.keyed_net import KeyedFaceNet, sample_keys


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def eval_lfw_same_key(model, ds, device, key_dim, batch_size, nw, tag) -> Dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc=tag, leave=False):
        a, b = a.to(device), b.to(device)
        k = sample_keys(a.size(0), key_dim, device)
        ea, eb = model(a, k), model(b, k)
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer = best_acc_eer(s, y)
    print(f"[{tag}] acc={acc:.4f} eer={eer:.4f} n={len(y)}")
    return {"accuracy": acc, "eer": eer, "n_pairs": int(len(y)), "mean_score": float(s.mean())}


@torch.no_grad()
def eval_cancelability(model, ds, device, key_dim, batch_size, nw) -> Dict:
    """Same image, different keys → should NOT match (low cosine)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    scores = []
    model.eval()
    for a, _b, _y in tqdm(loader, desc="cancel", leave=False):
        a = a.to(device)
        k1 = sample_keys(a.size(0), key_dim, device)
        k2 = sample_keys(a.size(0), key_dim, device)
        flip = torch.randint(0, key_dim, (a.size(0),), device=device)
        k2 = k2.clone()
        k2[torch.arange(a.size(0)), flip] = -k1[torch.arange(a.size(0)), flip]
        e1, e2 = model(a, k1), model(a, k2)
        scores.append((e1 * e2).sum(dim=1).cpu().numpy())
    s = np.concatenate(scores)
    # treat as impostors: accuracy if thresholded using same-key genuine threshold is tricky;
    # report mean cosine (lower = better cancelability) and fraction below 0.3
    out = {
        "mean_cosine": float(s.mean()),
        "frac_below_0.3": float((s < 0.3).mean()),
        "n": int(len(s)),
    }
    print(f"[cancel] mean_cos={out['mean_cosine']:.4f} frac<0.3={out['frac_below_0.3']:.4f}")
    return out


@torch.no_grad()
def eval_linkability(model, ds, device, key_dim, batch_size, nw) -> Dict:
    """
    Compare score distributions:
      - mated different-key (same image, k vs k')
      - impostor same-key (different images from LFW negatives)
    Lower D_sys-style gap ≈ better unlinkability. We report |mean_mated_diffkey - mean_imp|.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    mated_diff, impostor = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc="link", leave=False):
        a, b, y = a.to(device), b.to(device), y.to(device)
        k1 = sample_keys(a.size(0), key_dim, device)
        k2 = sample_keys(a.size(0), key_dim, device)
        flip = torch.randint(0, key_dim, (a.size(0),), device=device)
        k2 = k2.clone()
        k2[torch.arange(a.size(0)), flip] = -k1[torch.arange(a.size(0)), flip]
        # mated diff-key on image a
        ea1, ea2 = model(a, k1), model(a, k2)
        mated_diff.append((ea1 * ea2).sum(dim=1).cpu().numpy())
        # impostor: use negative pairs with same key
        mask = y == 0
        if mask.any():
            ea = model(a[mask], k1[mask])
            eb = model(b[mask], k1[mask])
            impostor.append((ea * eb).sum(dim=1).cpu().numpy())
    md = np.concatenate(mated_diff)
    imp = np.concatenate(impostor) if impostor else np.array([0.0])
    gap = float(abs(md.mean() - imp.mean()))
    out = {
        "mated_diffkey_mean": float(md.mean()),
        "impostor_mean": float(imp.mean()),
        "abs_mean_gap": gap,
        "n_mated": int(len(md)),
        "n_imp": int(len(imp)),
    }
    print(f"[link] mated_dk={out['mated_diffkey_mean']:.4f} imp={out['impostor_mean']:.4f} gap={gap:.4f}")
    return out


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    key_dim = int(blob.get("key_dim", cfg["model"].get("key_dim", 64)))
    emb_dim = int(blob.get("emb_dim", cfg["model"].get("emb_dim", 128)))
    model = KeyedFaceNet(key_dim=key_dim, emb_dim=emb_dim, pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None

    results: Dict = {"verification": {}, "security": {}}
    for name, builder in (("lfw", build_lfw), ("calfw", build_calfw), ("cplfw", build_cplfw)):
        try:
            ds = builder(cfg, image_size, max_p)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        results["verification"][name] = eval_lfw_same_key(model, ds, device, key_dim, bs, nw, f"{name}_samekey")

    ds_lfw = build_lfw(cfg, image_size, max_p)
    results["security"]["cancelability"] = eval_cancelability(model, ds_lfw, device, key_dim, bs, nw)
    results["security"]["linkability"] = eval_linkability(model, ds_lfw, device, key_dim, bs, nw)

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
