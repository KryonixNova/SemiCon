#!/usr/bin/env python3
"""Predict reference-in-search centres: a single pair, or an evaluator-
provided batch, through one shared inference path
(src/localizer/inference.py) -- no source-code changes needed to switch
between the two modes.

Single pair (same contract as scripts/predict.py; --checkpoint defaults to
checkpoints/production_v3/best.pt, so this also runs with just
--reference/--search given):
    python localize.py --checkpoint checkpoints/production_v3/best.pt \
        --reference ref.png --search search.png

For an external grader expecting exactly one "(x, y)" coordinate and
nothing else on stdout, add --xy-only:
    python localize.py --reference ref.png --search search.png --xy-only

Batch (reads reference_path/search_path columns from any manifest,
including one an evaluator supplies -- e.g. generate_dataset.py's output).
Every input column (ground truth, generation metadata) is carried through
into the output CSV alongside the new prediction columns, so the result is
one self-contained file with paths, true coordinates, predictions, and
metadata together:
    python localize.py --checkpoint checkpoints/production_v3/best.pt \
        --manifest output/test/manifest.csv --output predictions.csv
"""

import argparse
import csv
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch

from src.localizer.inference import load_model, predict_pair

DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "production_v3", "best.pt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help=f"default: {DEFAULT_CHECKPOINT}")
    p.add_argument("--reference", help="single-pair mode: reference image path")
    p.add_argument("--search", help="single-pair mode: search image path")
    p.add_argument("--manifest", help="batch mode: CSV with reference_path/search_path columns")
    p.add_argument("--output", help="batch mode: where to write predictions.csv")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--xy-only", action="store_true",
                   help="single-pair mode: print only 'x,y' with no confidence -- "
                        "the exact contract an external grader expecting a bare "
                        "(x, y) coordinate needs")
    args = p.parse_args()
    if bool(args.reference) != bool(args.search):
        p.error("--reference and --search must be given together")
    if args.manifest and args.reference:
        p.error("give either --reference/--search (single pair) or --manifest (batch), not both")
    if not args.manifest and not args.reference:
        p.error("must give either --reference/--search or --manifest")
    if args.manifest and not args.output:
        p.error("--manifest requires --output")
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    if args.manifest:
        with open(args.manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        out_rows = []
        pred_fields = ["predicted_x", "predicted_y", "confidence", "runtime_ms"]
        for row in rows:
            t0 = time.perf_counter()
            result = predict_pair(model, row["reference_path"], row["search_path"], device)
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            # Every input column (ground truth, generation metadata --
            # whatever the manifest already carries) is passed through
            # as-is, with prediction columns appended -- one self-contained
            # CSV with paths, true coordinates, predictions, and metadata
            # together, rather than a predictions-only file a reader has
            # to separately join back to the source manifest.
            out_rows.append({
                **row,
                "predicted_x": f"{result['x']:.3f}",
                "predicted_y": f"{result['y']:.3f}",
                "confidence": f"{result['confidence']:.4f}",
                "runtime_ms": f"{runtime_ms:.2f}",
            })
        fieldnames = (list(rows[0].keys()) if rows else ["id"]) + pred_fields
        out_parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_parent, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"wrote {len(out_rows)} predictions to {args.output}")
    else:
        result = predict_pair(model, args.reference, args.search, device)
        if args.xy_only:
            print(f"{result['x']:.2f},{result['y']:.2f}")
        elif args.verbose:
            print(f"predicted_x={result['x']:.2f} predicted_y={result['y']:.2f} "
                  f"confidence={result['confidence']:.4f}")
        else:
            print(f"{result['x']:.2f},{result['y']:.2f},{result['confidence']:.4f}")


if __name__ == "__main__":
    main()
