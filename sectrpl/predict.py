"""Embed one image with a provided or random key."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sectrpl.models.keyed_net import KeyedFaceNet, sample_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device_name = args.device or cfg.get("device", "cpu")
    device = torch.device(
        device_name if not str(device_name).startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    blob = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "last.pt", map_location=device, weights_only=False)
    key_dim = int(blob.get("key_dim", 64))
    emb_dim = int(blob.get("emb_dim", 128))
    model = KeyedFaceNet(key_dim=key_dim, emb_dim=emb_dim, pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    k = sample_keys(1, key_dim, device)
    with torch.no_grad():
        emb = model(x, k).cpu().numpy()[0]
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "embedding.json"
    with open(out, "w") as f:
        json.dump({"key": k.cpu().numpy()[0].tolist(), "embedding": emb.tolist()}, f)
        f.write("\n")
    print(f"Wrote {out} dim={len(emb)}")


if __name__ == "__main__":
    main()
