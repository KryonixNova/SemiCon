#!/usr/bin/env python3
"""Predict the reference's centre in a search image.

Matches the contract of baseline_solution/infer.py: takes two image paths and
prints the predicted centre. Outputs exactly the three contract values --
predicted_x, predicted_y, confidence. localization_error is NOT emitted: it
requires ground truth and is an evaluation metric only.

Example:
    python scripts/predict.py --checkpoint checkpoints/m3_hn_r24/best.pt \
        --reference ref.png --search search.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.geometry import REF_DS_PX, SEARCH_PX
from src.localizer.model import DriftSenseLocalizer


def load_standardized(path: str, target_px: int | None = None) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not read {path}")
    if target_px is not None and img.shape[0] != target_px:
        img = cv2.resize(img, (target_px, target_px), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(img)).float()
    t = (t - t.mean()) / t.std().clamp_min(1e-6)
    return t.unsqueeze(0).unsqueeze(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--search", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model = DriftSenseLocalizer(LocalizerConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.align_offset.fill_(ckpt["align_offset"])
    model.eval()

    # The reference is deliberately force-resized to REF_DS_PX regardless of
    # its input size -- that's intentional (matches how the model was
    # trained: reference crops are always downsampled to REF_DS_PX). It
    # implicitly assumes the reference is already at the correct nm/px scale
    # *before* this resize; a reference captured at a genuinely different
    # physical scale will be silently misinterpreted (resized to the right
    # pixel count without ever being rescaled to the right physical extent).
    ref = load_standardized(args.reference, REF_DS_PX).to(device)
    search = load_standardized(args.search).to(device)
    if search.shape[-2:] != (SEARCH_PX, SEARCH_PX):
        raise ValueError(
            f"search image must be {SEARCH_PX}x{SEARCH_PX} px, got "
            f"{search.shape[-2]}x{search.shape[-1]} px ({args.search}). "
            f"Unlike the reference, the search image is not auto-resized: "
            f"a wrong size means the physical scale is likely wrong too."
        )
    with torch.no_grad():
        out = model.predict(ref, search)

    x, y = float(out["x"][0]), float(out["y"][0])
    conf = float(out["confidence"][0])
    if args.verbose:
        print(f"predicted_x={x:.2f} predicted_y={y:.2f} confidence={conf:.4f}")
    else:
        print(f"{x:.2f},{y:.2f},{conf:.4f}")


if __name__ == "__main__":
    main()
