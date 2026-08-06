"""CLI harness: run N randomized reference/search localization samples,
report per-image compute time and success rate within a stated tolerance.
"""
import argparse
import json
import statistics
import tempfile
from pathlib import Path

import cv2
import numpy as np

from matcher import SuperPointLightGlueMatcher
from localizer import DeepMatchLocalizer
from reference_search_pairs import generate_sample


def run_benchmark(n: int, tolerance_px: float, out_dir: Path, n_examples: int = 5,
                   seed_start: int = 0) -> dict:
    matcher = SuperPointLightGlueMatcher()
    localizer = DeepMatchLocalizer(matcher)

    records = []
    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(n):
            seed = seed_start + i
            sample = generate_sample(seed, tmp_dir)
            result = localizer.localize(sample.reference_img, sample.search_img, sample.zoom_ratio)

            record = {
                "seed": seed,
                "success": result.success,
                "failure_reason": result.failure_reason,
                "predicted_center_px": result.predicted_center_px,
                "true_center_px": sample.true_center_px,
                "confidence": result.confidence,
                "num_matches": result.num_matches,
                "num_inliers": result.num_inliers,
                "elapsed_ms": result.elapsed_ms,
            }
            if result.success:
                err = float(np.hypot(result.predicted_center_px[0] - sample.true_center_px[0],
                                      result.predicted_center_px[1] - sample.true_center_px[1]))
                record["pixel_error"] = err
                record["within_tolerance"] = err <= tolerance_px
            else:
                record["pixel_error"] = None
                record["within_tolerance"] = False
            records.append(record)

            if i < n_examples:
                out_dir.mkdir(parents=True, exist_ok=True)
                _save_example(sample, result, out_dir / f"example_{seed:04d}.png")

    n_success = sum(r["success"] for r in records)
    n_within = sum(r["within_tolerance"] for r in records)
    times = [r["elapsed_ms"] for r in records]
    errors = [r["pixel_error"] for r in records if r["pixel_error"] is not None]

    summary = {
        "n_samples": n,
        "tolerance_px": tolerance_px,
        "n_success": n_success,
        "n_within_tolerance": n_within,
        "success_rate_pct": 100.0 * n_within / n,
        "compute_time_ms": {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p95": _percentile(times, 95),
            "max": max(times),
        },
        "pixel_error": {
            "mean": statistics.mean(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "max": max(errors) if errors else None,
        },
        "records": records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def _percentile(values: list, pct: float):
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1))))
    return s[idx]


def _save_example(sample, result, path: Path) -> None:
    canvas = cv2.cvtColor(sample.search_img, cv2.COLOR_GRAY2BGR)
    tx, ty = [int(round(v)) for v in sample.true_center_px]
    cv2.drawMarker(canvas, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    if result.success:
        px, py = [int(round(v)) for v in result.predicted_center_px]
        cv2.drawMarker(canvas, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
    cv2.imwrite(str(path), canvas)


def main():
    ap = argparse.ArgumentParser(description="SuperPoint+LightGlue localization benchmark")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tolerance-px", type=float, default=5.0)
    ap.add_argument("--out-dir", type=str, default="localization_results")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=5)
    args = ap.parse_args()

    summary = run_benchmark(args.n, args.tolerance_px, Path(args.out_dir),
                             n_examples=args.n_examples, seed_start=args.seed_start)

    print(f"\n{args.n} samples, tolerance={args.tolerance_px}px")
    print(f"  Success rate      : {summary['success_rate_pct']:.1f}%  "
          f"({summary['n_within_tolerance']}/{args.n} within tolerance)")
    print(f"  Localizer success : {summary['n_success']}/{args.n} (matches+RANSAC found a hypothesis)")
    print(f"  Compute time (ms) : mean={summary['compute_time_ms']['mean']:.1f}  "
          f"median={summary['compute_time_ms']['median']:.1f}  "
          f"p95={summary['compute_time_ms']['p95']:.1f}")
    if summary["pixel_error"]["mean"] is not None:
        print(f"  Pixel error       : mean={summary['pixel_error']['mean']:.2f}  "
              f"median={summary['pixel_error']['median']:.2f}  "
              f"max={summary['pixel_error']['max']:.2f}")
    print(f"  Results written to: {args.out_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()
