#!/usr/bin/env python3
"""Compare B0 (ZNCC), B1 (10x10 grid), B2 (dense corr), B3 (corr + context).

All methods see identical samples from the canvas-disjoint test split.

Example:
    python scripts/evaluate_localizer.py --checkpoint checkpoints/m2_context/best.pt \
        --n-samples 500
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from baseline_solution.grid_baseline import analytic_ceiling, grid_predict
from baseline_solution.zncc import zncc_match
from src.localizer.config import LocalizerConfig
from src.localizer.data import LocalizerDataset
from src.localizer.metrics import summarize
from src.localizer.model import DriftSenseLocalizer


def _to_u8(t: torch.Tensor) -> np.ndarray:
    """Undo standardisation for the classical baselines, which expect uint8."""
    a = t.squeeze().cpu().numpy()
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
    return (a * 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None,
                   help="omit to evaluate only the classical baselines")
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--out-dir", default="./eval_results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LocalizerConfig()

    model = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
        model = DriftSenseLocalizer(cfg, use_context=not args.no_context).to(device)
        model.load_state_dict(ckpt["model"])
        model.align_offset.fill_(ckpt["align_offset"])
        model.eval()

    rec = {k: {"x": [], "y": [], "s": []} for k in ("B0", "B1", "B3")}
    gx, gy = [], []

    for i, s in enumerate(LocalizerDataset("test", cfg)):
        if i >= args.n_samples:
            break
        ref_u8, srch_u8 = _to_u8(s["reference_img"]), _to_u8(s["search_img"])
        gx.append(float(s["gt_x"])); gy.append(float(s["gt_y"]))

        # scales=(1.0,): LocalizerDataset's reference_img is already downsampled
        # to search-image scale (10 nm/px), unlike zncc_match's original CLI
        # contract (native 1 nm/px PNG needing ~10x internal downsampling) --
        # the default multi-scale sweep would downsample it a second time.
        m = zncc_match(ref_u8, srch_u8, scales=(1.0,))
        rec["B0"]["x"].append(m["x"]); rec["B0"]["y"].append(m["y"])
        rec["B0"]["s"].append(m["score"])

        g = grid_predict(ref_u8, srch_u8)
        rec["B1"]["x"].append(g["x"]); rec["B1"]["y"].append(g["y"])
        rec["B1"]["s"].append(g["score"])

        if model is not None:
            with torch.no_grad():
                o = model.predict(s["reference_img"].unsqueeze(0).to(device),
                                  s["search_img"].unsqueeze(0).to(device))
            rec["B3"]["x"].append(float(o["x"][0]))
            rec["B3"]["y"].append(float(o["y"][0]))
            rec["B3"]["s"].append(float(o["confidence"][0]))

    label = {"B0": "ZNCC (classical)", "B1": "10x10 grid",
             "B3": "dense corr + context" if not args.no_context else "dense corr (B2)"}
    results = {}
    print(f"\n{'method':<26} {'acc@5px':>9} {'acc@10px':>9} {'acc@50px':>9} "
          f"{'median':>8} {'mean':>8} {'AP':>6}")
    print("-" * 80)
    for key in ("B0", "B1", "B3"):
        if not rec[key]["x"]:
            continue
        r = summarize(rec[key]["x"], rec[key]["y"], gx, gy, rec[key]["s"])
        results[key] = r
        print(f"{label[key]:<26} {r['acc@5px']:>9.3f} {r['acc@10px']:>9.3f} "
              f"{r['acc@50px']:>9.3f} {r['median_error_px']:>8.1f} "
              f"{r['mean_error_px']:>8.1f} {r['ap']:>6.3f}")

    print("\nB1 analytic ceiling (perfect cell classification):")
    ceiling_violations = []
    for tol in (5, 10, 50):
        c = analytic_ceiling(tol)
        got = results.get("B1", {}).get(f"acc@{tol}px")
        flag = ""
        if got is not None and got > c + 1e-6:
            flag = "  <-- EXCEEDS CEILING (bug!)"
            ceiling_violations.append((tol, got, c))
        print(f"  acc@{tol}px <= {c:.4f}" + (f"   measured {got:.4f}{flag}" if got is not None else ""))
    results["B1_analytic_ceiling"] = {f"acc@{t}px": analytic_ceiling(t) for t in (5, 10, 50)}

    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_dir}/comparison.json")

    if ceiling_violations:
        raise AssertionError(
            f"B1 measured accuracy exceeded its analytic ceiling at "
            f"{len(ceiling_violations)} tolerance(s): {ceiling_violations} -- "
            f"this indicates a bug in grid_predict, not a good result."
        )


if __name__ == "__main__":
    main()
