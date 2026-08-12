#!/usr/bin/env python3
"""Generate a persisted Drift-Sense localization dataset split: reference/
search PNG pairs plus a manifest.csv of ground truth and generation
metadata, using the exact same on-the-fly generator (src/localizer/data.py)
the training pipeline itself draws from -- so a dataset written by this
script is representative of what the model was actually trained/evaluated
on, not a separately-diverged generator.

Example:
    python generate_dataset.py --split test --num-samples 50 --seed 200000 \
        --output-dir ./output --imaging-noise-profile harsh \
        --geometric-profile drift
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from src.localizer.config import LocalizerConfig
from src.localizer.data import generate_canvas_bundle, sample_pair, split_seed_range
from src.localizer.geometry import REF_PX as REFERENCE_HIRES_PX


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="which canvas-disjoint seed range to draw from")
    p.add_argument("--seed", type=int, default=None,
                   help="first canvas seed; defaults to the chosen split's "
                        "own range start. Must fall inside that range.")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--jitter-profile", default="normal",
                   choices=["normal", "zero", "shifted"])
    p.add_argument("--imaging-noise-profile", default="normal",
                   choices=["normal", "harsh"])
    p.add_argument("--geometric-profile", default="normal",
                   choices=["normal", "drift"])
    p.add_argument("--crops-per-canvas", type=int, default=1,
                   help="reference crops to draw per generated canvas "
                        "before moving to the next canvas seed")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = LocalizerConfig()
    lo, hi = split_seed_range(args.split, cfg)
    start_seed = args.seed if args.seed is not None else lo
    if not (lo <= start_seed < hi):
        raise SystemExit(
            f"--seed {start_seed} is outside the {args.split} split's own "
            f"canvas-disjoint range [{lo}, {hi}) -- pick a seed inside that "
            f"range so generated data doesn't overlap canvases used for a "
            f"different split.")

    ref_dir = os.path.join(args.output_dir, "reference")
    search_dir = os.path.join(args.output_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    fieldnames = ["id", "architecture", "reference_path", "search_path", "gt_x", "gt_y",
                 "canvas_seed", "crop_index", "jitter_profile",
                 "imaging_noise_profile", "geometric_profile",
                 "scale_ratio", "rotation_deg"]

    i = 0
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        seed = start_seed
        while i < args.num_samples:
            if seed >= hi:
                raise SystemExit(
                    f"ran out of canvas seeds in the {args.split} split's "
                    f"range before reaching --num-samples {args.num_samples} "
                    f"(wrote {i}) -- lower --num-samples, raise "
                    f"--crops-per-canvas, or start from a lower --seed.")
            bundle = generate_canvas_bundle(seed, args.jitter_profile,
                                            args.imaging_noise_profile)
            for k in range(args.crops_per_canvas):
                if i >= args.num_samples:
                    break
                s = sample_pair(bundle, seed, k, args.imaging_noise_profile,
                                args.geometric_profile)

                ref_path = os.path.join(ref_dir, f"{i:05d}.png")
                search_path = os.path.join(search_dir, f"{i:05d}.png")
                # Persisted at REFERENCE_HIRES_PX (matching the spec's "1000 x
                # 1000 image representing a 100x close-up view") by upscaling
                # the exact 100x100 tensor the model actually trains/predicts
                # on -- not a separate, higher-detail render -- so localize.py
                # downsampling it straight back (INTER_AREA, see
                # load_standardized) reconstructs that same tensor almost
                # exactly, the same small round-trip fuzziness any real
                # save-to-disk-then-reload workflow has.
                ref_hires = cv2.resize(s["reference_img_u8"],
                                       (REFERENCE_HIRES_PX, REFERENCE_HIRES_PX),
                                       interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(ref_path, ref_hires)
                cv2.imwrite(search_path, bundle["search_img"])

                writer.writerow({
                    "id": i, "architecture": "dram",
                    "reference_path": ref_path, "search_path": search_path,
                    "gt_x": s["gt_x"], "gt_y": s["gt_y"], "canvas_seed": seed,
                    "crop_index": k, "jitter_profile": args.jitter_profile,
                    "imaging_noise_profile": args.imaging_noise_profile,
                    "geometric_profile": args.geometric_profile,
                    "scale_ratio": round(s["scale_ratio"], 4),
                    "rotation_deg": round(s["rotation_deg"], 4),
                })
                print(f"[{i + 1}/{args.num_samples}] seed={seed} crop={k} -> "
                      f"gt=({s['gt_x']:.1f}, {s['gt_y']:.1f})")
                i += 1
            seed += 1

    print(f"wrote {i} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
