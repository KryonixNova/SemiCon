#!/usr/bin/env python3
"""Generate a DRAM-only dataset of m x n multi-region die/array samples.

Each sample tiles an independent m x n grid (1 <= m, n <= 5, chosen per
sample) of DRAM "mats" -- each mat gets its own randomly-picked DRAM preset
(dram_1x/dense/loose/wide/compact/legacy) -- separated by routing-strip
material, onto the fine canvas. Pattern collapse is disabled
(collapse_threshold_nm=0.0), so no lines ever bridge/merge. Imaging noise
(dose, drift, beam spot) is mildly randomized per sample so every Search
image is visually distinct on top of the structural differences.

Unlike src/patterns/zones.py's generate_zone_canvas (which always produces a
square k x k grid, since it reuses the same mat_size_nm for both axes), this
picks the row-count and column-count independently to get true m x n grids.

Example:
    python generate_mxn_dram_dataset.py --num-samples 100 --output-dir "./add here" --seed 42
"""

import argparse
import csv
import os

import cv2
import numpy as np

from src import sem_imaging
from src.pipeline import (
    GenerationParams,
    FINE_CANVAS_SIZE_PX,
    REFERENCE_SIZE_PX,
    SCALE_FACTOR,
    PIXEL_SIZE_REF_NM,
    PIXEL_SIZE_SEARCH_NM,
)
from src.patterns.mxn import generate_mxn_canvas

STRIP_WIDTH_NM_RANGE = (120.0, 400.0)


def _pick_crop_origin(strip_rects, boundary_bias: float, rng: np.random.Generator) -> tuple:
    max_offset = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    if strip_rects and rng.random() < boundary_bias:
        sx, sy, sw, sh = strip_rects[int(rng.integers(0, len(strip_rects)))]
        scx, scy = sx + sw / 2.0, sy + sh / 2.0
        x0 = scx - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
        y0 = scy - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
        return int(np.clip(x0, 0, max_offset)), int(np.clip(y0, 0, max_offset))
    return int(rng.integers(0, max_offset + 1)), int(rng.integers(0, max_offset + 1))


def generate_one(rng: np.random.Generator) -> dict:
    m = int(rng.integers(1, 6))
    n = int(rng.integers(1, 6))
    strip_width_nm = float(rng.uniform(*STRIP_WIDTH_NM_RANGE))

    zone = generate_mxn_canvas(FINE_CANVAS_SIZE_PX, m, n, strip_width_nm, rng)
    fine_canvas = zone["canvas"]

    x0, y0 = _pick_crop_origin(zone["strip_rects"], boundary_bias=0.35, rng=rng)
    crop = fine_canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    # Mild per-sample randomization of imaging conditions (structure is
    # already unique via m/n + per-mat preset mix + crop location -- this
    # adds acquisition-condition variety on top).
    beam_spot_size_nm = float(rng.uniform(3.0, 8.0))
    dose_reference = float(rng.uniform(1200.0, 3000.0))
    dose_search = float(rng.uniform(80.0, 400.0))
    shear_amplitude_px = float(rng.uniform(0.3, 2.5))
    drift_jitter_px = float(rng.uniform(0.1, 1.0))
    detector_noise_sigma_ref = float(rng.uniform(1.0, 3.0))
    detector_noise_sigma_search = float(rng.uniform(3.0, 7.0))

    reference_img = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=beam_spot_size_nm,
        dose=dose_reference,
        rng=rng,
        detector_noise_sigma=detector_noise_sigma_ref,
        drift_jitter_px=drift_jitter_px * 0.2,
    )
    search_img = sem_imaging.image_search(
        fine_canvas,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=beam_spot_size_nm,
        dose=dose_search,
        rng=rng,
        shear_amplitude_px=shear_amplitude_px,
        drift_jitter_px=drift_jitter_px,
        detector_noise_sigma=detector_noise_sigma_search,
    )

    box_w = box_h = REFERENCE_SIZE_PX // SCALE_FACTOR
    gt_x0, gt_y0 = x0 / SCALE_FACTOR, y0 / SCALE_FACTOR
    gt_cx, gt_cy = gt_x0 + box_w / 2.0, gt_y0 + box_h / 2.0

    return {
        "reference_img": reference_img,
        "search_img": search_img,
        "gt_x": gt_cx, "gt_y": gt_cy,
        "gt_box": (gt_x0, gt_y0, box_w, box_h),
        "m": m, "n": n, "strip_width_nm": strip_width_nm,
        "mat_feature_sizes_nm": zone["mat_feature_sizes_nm"],
        "beam_spot_size_nm": beam_spot_size_nm,
        "dose_reference": dose_reference, "dose_search": dose_search,
        "shear_amplitude_px": shear_amplitude_px, "drift_jitter_px": drift_jitter_px,
        "detector_noise_sigma_ref": detector_noise_sigma_ref,
        "detector_noise_sigma_search": detector_noise_sigma_search,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--output-dir", default="./add here")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ref_dir = os.path.join(args.output_dir, "reference")
    search_dir = os.path.join(args.output_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    fieldnames = [
        "id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
        "m", "n", "strip_width_nm", "mat_feature_sizes_nm",
        "beam_spot_size_nm", "dose_reference", "dose_search",
        "shear_amplitude_px", "drift_jitter_px",
        "detector_noise_sigma_ref", "detector_noise_sigma_search",
        "collapse_threshold_nm", "seed",
    ]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(args.num_samples):
            sample = generate_one(rng)

            ref_path = os.path.join(ref_dir, f"{i:05d}.png")
            search_path = os.path.join(search_dir, f"{i:05d}.png")
            cv2.imwrite(ref_path, sample["reference_img"])
            cv2.imwrite(search_path, sample["search_img"])

            gx0, gy0, gw, gh = sample["gt_box"]
            writer.writerow({
                "id": i,
                "reference_path": ref_path,
                "search_path": search_path,
                "gt_x": sample["gt_x"], "gt_y": sample["gt_y"],
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "m": sample["m"], "n": sample["n"],
                "strip_width_nm": round(sample["strip_width_nm"], 2),
                "mat_feature_sizes_nm": ";".join(str(v) for v in sample["mat_feature_sizes_nm"]),
                "beam_spot_size_nm": round(sample["beam_spot_size_nm"], 3),
                "dose_reference": round(sample["dose_reference"], 2),
                "dose_search": round(sample["dose_search"], 2),
                "shear_amplitude_px": round(sample["shear_amplitude_px"], 3),
                "drift_jitter_px": round(sample["drift_jitter_px"], 3),
                "detector_noise_sigma_ref": round(sample["detector_noise_sigma_ref"], 3),
                "detector_noise_sigma_search": round(sample["detector_noise_sigma_search"], 3),
                "collapse_threshold_nm": 0.0,
                "seed": args.seed,
            })
            print(f"[{i + 1}/{args.num_samples}] {sample['m']}x{sample['n']} die -> "
                  f"gt=({sample['gt_x']:.1f}, {sample['gt_y']:.1f})")

    print(f"Wrote {args.num_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
