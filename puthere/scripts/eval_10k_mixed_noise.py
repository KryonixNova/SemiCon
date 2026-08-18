#!/usr/bin/env python3
"""Test a checkpoint on 10000 held-out samples, each canvas randomly
imaged under either the baseline ("normal") or amplified ("harsh")
imaging-noise profile.

Held-out means the test split's own seed range (config.test_seed_lo/hi),
disjoint from every training/val seed this checkpoint ever saw.

Example:
    python scripts/eval_10k_mixed_noise.py \
        --checkpoint checkpoints/production_v3/best.pt --n-samples 10000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.data import generate_canvas_bundle, sample_pair
from src.localizer.metrics import summarize
from src.localizer.model import DriftSenseLocalizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the per-canvas noise-profile coin flip only "
                        "-- which canvas seeds are used comes from the "
                        "config's test split range, not this.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = LocalizerConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model = DriftSenseLocalizer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.align_offset.fill_(ckpt["align_offset"])
    model.eval()
    print(f"loaded {args.checkpoint} (step={ckpt.get('step')}, "
          f"best_acc={ckpt.get('best_acc')})")

    n_canvases = -(-args.n_samples // cfg.crops_per_canvas)  # ceil div
    if cfg.test_seed_lo + n_canvases > cfg.test_seed_hi:
        raise SystemExit(
            f"need {n_canvases} canvases but test split only has "
            f"{cfg.test_seed_hi - cfg.test_seed_lo} -- reduce --n-samples.")

    profile_rng = np.random.default_rng(args.seed)

    gx, gy, px, py, conf, profiles = [], [], [], [], [], []
    ref_batch, srch_batch = [], []

    def flush():
        if not ref_batch:
            return
        with torch.no_grad():
            r = torch.stack(ref_batch, 0).to(device)
            s = torch.stack(srch_batch, 0).to(device)
            out = model.predict(r, s)
        px.extend(float(v) for v in out["x"])
        py.extend(float(v) for v in out["y"])
        conf.extend(float(v) for v in out["confidence"])
        ref_batch.clear()
        srch_batch.clear()

    n_done = 0
    for c in range(n_canvases):
        seed = cfg.test_seed_lo + c
        profile = "harsh" if profile_rng.random() < 0.5 else "normal"
        bundle = generate_canvas_bundle(seed, "normal", profile)
        for k in range(cfg.crops_per_canvas):
            if n_done >= args.n_samples:
                break
            s = sample_pair(bundle, seed, k, profile, "normal")
            gx.append(s["gt_x"]); gy.append(s["gt_y"])
            profiles.append(profile)
            ref_batch.append(s["reference_img"].unsqueeze(0))
            srch_batch.append(s["search_img"].unsqueeze(0))
            n_done += 1
            if len(ref_batch) >= args.batch_size:
                flush()
        if n_done % 1000 < cfg.crops_per_canvas:
            print(f"  {n_done}/{args.n_samples} samples generated+scored...")
    flush()

    profiles = np.array(profiles)
    overall = summarize(px, py, gx, gy, conf)
    print(f"\noverall (n={len(px)}): acc@1px={overall['acc@1px']:.3f} "
          f"acc@3px={overall['acc@3px']:.3f} acc@5px={overall['acc@5px']:.3f} "
          f"median={overall['median_error_px']:.2f}px mean={overall['mean_error_px']:.2f}px "
          f"AP={overall['ap']:.3f}")

    for label in ("normal", "harsh"):
        idx = np.where(profiles == label)[0]
        if len(idx) == 0:
            continue
        sub = summarize([px[i] for i in idx], [py[i] for i in idx],
                         [gx[i] for i in idx], [gy[i] for i in idx],
                         [conf[i] for i in idx])
        print(f"{label:>8} (n={len(idx)}): acc@1px={sub['acc@1px']:.3f} "
              f"acc@3px={sub['acc@3px']:.3f} acc@5px={sub['acc@5px']:.3f} "
              f"median={sub['median_error_px']:.2f}px mean={sub['mean_error_px']:.2f}px "
              f"AP={sub['ap']:.3f}")


if __name__ == "__main__":
    main()
