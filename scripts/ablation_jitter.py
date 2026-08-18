#!/usr/bin/env python3
"""Ablation A1: does the model depend on synthetic jitter fingerprints?

Trains on one jitter profile, evaluates on all three.

  normal  : POSITION_JITTER_NM=1.5, WIDTH_JITTER_FRACTION=0.10, gaussian
  zero    : no jitter -- an exactly periodic lattice (provably degenerate)
  shifted : 3.0 / 0.20, uniform position jitter -- different distribution
            family (uniform vs. normal's gaussian) AND magnitude for the
            position-jitter component, which is the aperiodicity signal.
            Width (CD) jitter stays gaussian in every profile; it is not
            the subject of this ablation.

Reading the result:
  * normal -> zero drop is EXPECTED. Zero jitter removes the only aperiodic
    signal, leaving mat boundaries as the sole disambiguator. A drop here
    measures how much the model leaned on fingerprints, and some drop is
    correct behaviour.
  * normal -> shifted drop is the SERIOUS failure: it means the model fitted
    one noise distribution rather than aperiodicity in general, and would not
    transfer to real SEM data.

Example:
    python scripts/ablation_jitter.py --checkpoint checkpoints/production_v3/best.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.localizer.config import LocalizerConfig
from src.localizer.data import LocalizerDataset
from src.localizer.metrics import summarize
from src.localizer.model import DriftSenseLocalizer

PROFILES = ["normal", "zero", "shifted"]


def evaluate_profile(model, cfg, profile, n, device):
    px, py, gx, gy, conf = [], [], [], [], []
    with torch.no_grad():
        for i, s in enumerate(LocalizerDataset("test", cfg, profile)):
            if i >= n:
                break
            o = model.predict(s["reference_img"].unsqueeze(0).to(device),
                              s["search_img"].unsqueeze(0).to(device))
            px.append(float(o["x"][0])); py.append(float(o["y"][0]))
            conf.append(float(o["confidence"][0]))
            gx.append(float(s["gt_x"])); gy.append(float(s["gt_y"]))
    return summarize(px, py, gx, gy, conf)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="a model trained on the 'normal' profile")
    p.add_argument("--checkpoint-zero", default=None,
                   help="optional model trained on the 'zero' profile")
    p.add_argument("--best-metric", type=int, default=5, choices=[1, 3, 5],
                   help="which acc@Tpx tolerance (in px) drives the "
                        "comparison table and the normal->{zero,shifted} "
                        "drop computation.")
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--out-dir", default="./eval_results")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LocalizerConfig()
    results = {}

    for train_profile, ckpt_path in (("normal", args.checkpoint),
                                     ("zero", args.checkpoint_zero)):
        if ckpt_path is None:
            continue
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model = DriftSenseLocalizer(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.align_offset.fill_(ckpt["align_offset"])
        model.eval()
        results[train_profile] = {
            tp: evaluate_profile(model, cfg, tp, args.n_samples, device)
            for tp in PROFILES
        }

    best_key = f"acc@{args.best_metric}px"
    print(f"\n{'train \\ test':<14} " + " ".join(f"{p:>12}" for p in PROFILES))
    print("-" * 56)
    for tr, row in results.items():
        print(f"{tr:<14} " + " ".join(f"{row[tp][best_key]:>12.3f}" for tp in PROFILES))

    if "normal" in results:
        base = results["normal"]["normal"][best_key]
        for tp in ("zero", "shifted"):
            drop = base - results["normal"][tp][best_key]
            note = ("expected -- exactly periodic, degenerate by construction"
                    if tp == "zero" else
                    "SERIOUS if large -- overfit to one noise distribution")
            print(f"  normal -> {tp:<8} drop {drop:+.3f}   ({note})")

    with open(os.path.join(args.out_dir, "ablation_jitter.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_dir}/ablation_jitter.json")


if __name__ == "__main__":
    main()
