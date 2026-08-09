#!/usr/bin/env python3
"""Calibrate decode.peak_tie_ratio on the validation split.

Sweeps tau in [0.80, 1.00] and selects the value maximising acc@50px, then
reports the full sensitivity curve so the choice is auditable rather than
arbitrary.

Example:
    python scripts/calibrate_tie_ratio.py --checkpoint checkpoints/m3_hn_r24/best.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.data import LocalizerDataset
from src.localizer.decode import decode
from src.localizer.metrics import summarize
from src.localizer.model import DriftSenseLocalizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--out-dir", default="./eval_results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LocalizerConfig()

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model = DriftSenseLocalizer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    align = float(ckpt["align_offset"])
    model.eval()

    # Cache heatmaps once; the sweep is pure post-processing.
    heatmaps, offsets, gx, gy = [], [], [], []
    with torch.no_grad():
        for i, s in enumerate(LocalizerDataset("val", cfg)):
            if i >= args.n_samples:
                break
            logits, off = model(s["reference_img"].unsqueeze(0).to(device),
                                s["search_img"].unsqueeze(0).to(device))
            heatmaps.append(torch.sigmoid(logits).cpu())
            offsets.append(off.cpu())
            gx.append(float(s["gt_x"])); gy.append(float(s["gt_y"]))

    hm = torch.cat(heatmaps); off = torch.cat(offsets)
    curve = {}
    print(f"{'tau':>6} {'acc@50px':>9} {'acc@5px':>9}")
    for tau in np.arange(0.80, 1.001, 0.01):
        out = decode(hm, off, peak_tie_ratio=float(tau),
                     nms_kernel=cfg.nms_kernel, align_offset=align)
        m = summarize(out["x"].tolist(), out["y"].tolist(), gx, gy,
                      out["confidence"].tolist())
        curve[round(float(tau), 2)] = m
        print(f"{tau:>6.2f} {m['acc@50px']:>9.3f} {m['acc@5px']:>9.3f}")

    best = max(curve, key=lambda t: curve[t]["acc@50px"])
    print(f"\nselected peak_tie_ratio = {best} "
          f"(acc@50px {curve[best]['acc@50px']:.3f})")
    with open(os.path.join(args.out_dir, "tie_ratio.json"), "w") as f:
        json.dump({"selected": best, "curve": curve}, f, indent=2)


if __name__ == "__main__":
    main()
