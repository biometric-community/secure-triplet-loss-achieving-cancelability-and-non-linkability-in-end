"""Train keyed FR with Secure Triplet Loss on CelebA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sectrpl.data.celeba import CelebAIdDataset, resolve_celeba_root
from sectrpl.losses.secure_triplet import secure_triplet_loss
from sectrpl.models.keyed_net import KeyedFaceNet, sample_keys


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


class PairIdDataset(Dataset):
    """Yields (img_a, img_b, label) with same-ID pairs when possible."""

    def __init__(self, base: CelebAIdDataset):
        self.base = base
        by_id: Dict[int, List[int]] = defaultdict(list)
        for i, (_, sid) in enumerate(base.samples):
            by_id[sid].append(i)
        self.ids = [sid for sid, idxs in by_id.items() if len(idxs) >= 2]
        self.by_id = by_id
        self.all_idx = list(range(len(base)))

    def __len__(self):
        return max(len(self.ids) * 2, 1)

    def __getitem__(self, i):
        rng = np.random.RandomState(None)
        sid = self.ids[i % len(self.ids)]
        idxs = self.by_id[sid]
        a, b = rng.choice(idxs, size=2, replace=False)
        # negative identity
        neg_sid = sid
        while neg_sid == sid:
            neg_sid = self.ids[rng.randint(0, len(self.ids))]
        n = rng.choice(self.by_id[neg_sid])
        xa, ya = self.base[a]
        xb, yb = self.base[b]
        xn, yn = self.base[n]
        assert ya == yb
        return xa, xb, xn, ya


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    key_dim = int(cfg["model"].get("key_dim", 64))
    emb_dim = int(cfg["model"].get("emb_dim", 128))

    base = CelebAIdDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    ds = PairIdDataset(base)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = KeyedFaceNet(key_dim=key_dim, emb_dim=emb_dim, pretrained=True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=float(cfg["train"].get("lr", 0.05)), momentum=0.9, weight_decay=5e-4)
    epochs = int(cfg["train"].get("epochs", 8))
    margin = float(cfg["train"].get("margin", 0.5))
    gamma = float(cfg["train"].get("gamma", 0.5))

    hist = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses, stats_acc = [], defaultdict(list)
        pbar = tqdm(loader, desc=f"ep{epoch}/{epochs}", leave=False)
        for xa, xb, xn, _ylab in pbar:
            xa, xb, xn = xa.to(device), xb.to(device), xn.to(device)
            bsz = xa.size(0)
            k = sample_keys(bsz, key_dim, device)
            k2 = sample_keys(bsz, key_dim, device)
            # ensure k2 != k for at least one bit
            flip = torch.randint(0, key_dim, (bsz,), device=device)
            k2 = k2.clone()
            k2[torch.arange(bsz), flip] = -k[torch.arange(bsz), flip]

            emb_ak = model(xa, k)
            emb_pk = model(xb, k)
            emb_ak2 = model(xa, k2)
            emb_nk = model(xn, k)

            loss, st = secure_triplet_loss(emb_ak, emb_pk, emb_ak2, emb_nk, margin=margin, gamma=gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            for kk, vv in st.items():
                stats_acc[kk].append(vv)
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")

        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        for kk, vv in stats_acc.items():
            row[kk] = float(np.mean(vv))
        hist.append(row)
        print(f"[train] epoch={epoch} loss={row['loss']:.4f} d_pos={row['d_pos']:.3f} d_cancel={row['d_cancel']:.3f} d_imp={row['d_imp']:.3f}")

    elapsed = time.time() - t0
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "key_dim": key_dim, "emb_dim": emb_dim}, ckpt_dir / "last.pt")
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "epochs": epochs,
        "n_train_pairs": len(ds),
        "n_base": len(base),
        "num_classes": base.num_classes,
        "elapsed_sec": elapsed,
        "history": hist,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
