"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sectrpl.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Secure triplet loss")
    ax.set_title("Secure Triplet training (CelebA)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_verify(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ver = eval_res.get("verification", {})
    keys = list(ver.keys())
    vals = [ver[k]["accuracy"] * 100 for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, vals, color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Same-key verification")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_security(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sec = eval_res.get("security", {})
    can = sec.get("cancelability", {})
    link = sec.get("linkability", {})
    names = ["cancel mean cos", "frac cos<0.3", "link |gap|"]
    vals = [
        can.get("mean_cosine", 0) * 100,
        can.get("frac_below_0.3", 0) * 100,
        link.get("abs_mean_gap", 0) * 100,
    ]
    x = np.arange(len(names))
    ax.bar(x, vals, color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Value (×100 where cosine)")
    ax.set_title("Cancelability / linkability proxies")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        "# REPORT — Secure Triplet Loss",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin of Pinto et al. (TBIOM): keyed face embedder trained with Secure Triplet Loss",
        "(cancelability + non-linkability) on CelebA; evaluated on LFW/CALFW/CPLFW.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Base images: {train_s.get('n_base')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Same-key verification",
        "",
        "| Dataset | Accuracy | EER | N |",
        "|---------|----------|-----|---|",
    ]
    for k, v in sorted(eval_res.get("verification", {}).items()):
        lines.append(f"| {k} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['n_pairs']} |")
    sec = eval_res.get("security", {})
    can = sec.get("cancelability", {})
    link = sec.get("linkability", {})
    lines += [
        "",
        f"![Verification]({fig_dir.as_posix()}/verification.svg)",
        "",
        "## Security proxies",
        "",
        f"- Cancelability mean cosine (same image, different keys): **{can.get('mean_cosine', 0):.4f}**",
        f"- Fraction cosine < 0.3: **{can.get('frac_below_0.3', 0)*100:.1f}%**",
        f"- Linkability |mated − non-mated| (different keys) mean gap: **{link.get('abs_mean_gap', 0):.4f}**",
        f"- Linkability Dsys: **{link.get('dsys', 0):.4f}**",
        "",
        f"![Security]({fig_dir.as_posix()}/security.svg)",
        "",
        "## Notes",
        "",
        "- Paper uses YouTube Faces / UofTDB ECG; this rebuild uses CelebA + LFW family.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_loss.svg")
    plot_verify(eval_res, fig_dir / "verification.svg")
    plot_security(eval_res, fig_dir / "security.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
