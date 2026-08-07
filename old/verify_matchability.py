"""
Diagnostic script: probe whether the current rendering produces enough
locally-distinctive content for SuperPoint+LightGlue to find real
correspondences between a reference patch and its containing search image.

Not part of the reviewed benchmark pipeline (matcher.py/localizer.py) --
this bypasses LightGlue's default match-confidence threshold to see the
RAW best confidence across the full candidate grid, which is what actually
diagnoses a content/matchability problem (a low but nonzero post-threshold
match count can hide an even worse raw-confidence ceiling).

Usage:
    /home/nihal/miniconda3/envs/royl/bin/python verify_matchability.py --n 5
"""
import argparse
import tempfile

import cv2
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd

from reference_search_pairs import generate_sample


def probe_sample(seed: int, tmp_dir: str, extractor, matcher, device: str) -> float:
    """Return the best raw LightGlue match confidence for one sample,
    ignoring the usual match-acceptance threshold entirely."""
    sample = generate_sample(seed=seed, tmp_dir=tmp_dir)

    low_w = round(sample.reference_img.shape[1] / sample.zoom_ratio)
    low_h = round(sample.reference_img.shape[0] / sample.zoom_ratio)
    ref_lowres = cv2.resize(sample.reference_img, (low_w, low_h), interpolation=cv2.INTER_AREA)

    t_ref = numpy_image_to_torch(ref_lowres).to(device)
    t_search = numpy_image_to_torch(sample.search_img).to(device)

    with torch.no_grad():
        feats_ref = extractor.extract(t_ref)
        feats_search = extractor.extract(t_search)
        result = matcher({"image0": feats_ref, "image1": feats_search})
        r = rbd(result)
        return float(r["matching_scores0"].max().item())


def main():
    ap = argparse.ArgumentParser(description="Probe raw SuperPoint+LightGlue match confidence")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint", filter_threshold=0.0).eval().to(device)

    print(f"\nProbing raw match confidence on {args.n} samples (device={device})\n")

    scores = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(args.n):
            seed = args.seed_start + i
            best = probe_sample(seed, tmp_dir, extractor, matcher, device)
            scores.append(best)
            print(f"  seed={seed:>4}  best_raw_confidence={best:.4f}")

    mean_score = sum(scores) / len(scores)
    print(f"\nMean best raw confidence: {mean_score:.4f}")
    print("(Pre-rework baseline measured during investigation was ~0.006-0.010")
    print(" across several seeds -- essentially noise. A confident correct match")
    print(" on natural images typically scores in the 0.5-1.0 range.)")
    if mean_score < 0.05:
        print("\nStill very low -- consider increasing reference_search_pairs.py's")
        print("DEFAULT_PARAMS_KW variation parameters further before re-running the")
        print("full benchmark.")
    else:
        print("\nMeaningful rise over baseline -- reasonable to proceed to the full")
        print("benchmark.py --n 30 run.")


if __name__ == "__main__":
    main()
