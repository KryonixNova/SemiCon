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

import torch

from src.localizer.inference import load_model, predict_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--search", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    result = predict_pair(model, args.reference, args.search, device)

    if args.verbose:
        print(f"predicted_x={result['x']:.2f} predicted_y={result['y']:.2f} "
              f"confidence={result['confidence']:.4f}")
    else:
        print(f"{result['x']:.2f},{result['y']:.2f},{result['confidence']:.4f}")


if __name__ == "__main__":
    main()
