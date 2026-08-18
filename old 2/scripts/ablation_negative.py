#!/usr/bin/env python3
"""Ablation A2 (diagnostic): false-positive behaviour on absent references.

Pairs each Search image with a Reference cut from a DIFFERENT canvas, so the
target is genuinely absent. The model has no "absent" output, so this tests
the confidence signal rather than the coordinates.

Metric: AUROC of `confidence` separating present from absent. The classical
baseline fails this badly -- it averaged 0.902 on a set containing 35%
failures -- so this is the check that peak margin is genuinely informative.

Gates nothing: the deployed task guarantees the reference is present.

Example:
    python scripts/ablation_negative.py --checkpoint checkpoints/m3_hn_r24/best.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.data import generate_canvas_bundle, sample_pair
from src.localizer.model import DriftSenseLocalizer


def auroc(pos, neg) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties counted as half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--out-dir", default="./eval_results")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LocalizerConfig()
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model = DriftSenseLocalizer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.align_offset.fill_(ckpt["align_offset"])
    model.eval()

    present, absent = [], []
    lo = cfg.test_seed_lo
    for i in range(args.n_samples):
        a = generate_canvas_bundle(lo + i)
        b = generate_canvas_bundle(lo + i + 5000)      # unrelated canvas
        s_a = sample_pair(a, lo + i, 0)
        s_b = sample_pair(b, lo + i + 5000, 0)
        search = s_a["search_img"].unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            present.append(float(model.predict(
                s_a["reference_img"].unsqueeze(0).unsqueeze(0).to(device),
                search)["confidence"][0]))
            absent.append(float(model.predict(
                s_b["reference_img"].unsqueeze(0).unsqueeze(0).to(device),
                search)["confidence"][0]))

    score = auroc(present, absent)
    out = {"auroc": score,
           "present_mean": float(np.mean(present)),
           "absent_mean": float(np.mean(absent)),
           "n": args.n_samples}
    print(f"confidence AUROC (present vs absent): {score:.3f}")
    print(f"  present mean {out['present_mean']:.4f} | absent mean {out['absent_mean']:.4f}")
    print("  1.0 = perfect separation, 0.5 = confidence carries no information")
    with open(os.path.join(args.out_dir, "ablation_negative.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
