#!/usr/bin/env python3
"""Spec-required validation report: runs the localizer across a noise x
geometry condition matrix and reports Euclidean error (mean/median/worst),
pass rate @5/4/2/1px, sub-pixel detail, runtime per pair (with hardware/
Python version/timing method), and one visualized failure case with a
root-cause note.

Example:
    python scripts/validation_report.py \
        --checkpoint checkpoints/production_v3/best.pt \
        --n-per-condition 50 --out-dir results
"""

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.data import LocalizerDataset
from src.localizer.metrics import localization_error
from src.localizer.model import DriftSenseLocalizer

THRESHOLDS_PX = (5.0, 4.0, 2.0, 1.0)
CONDITIONS = [
    {"imaging_noise_profile": "normal", "geometric_profile": "normal"},
    {"imaging_noise_profile": "harsh", "geometric_profile": "normal"},
    {"imaging_noise_profile": "normal", "geometric_profile": "drift"},
    {"imaging_noise_profile": "harsh", "geometric_profile": "drift"},
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-per-condition", type=int, default=50,
                   help="pairs evaluated per noise x geometry condition; "
                        "the spec's minimum total across all conditions is 30")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def _timing_method_note():
    return ("wall-clock via time.perf_counter() around the single call to "
            "model.predict(reference, search); excludes checkpoint/model "
            "loading (a one-time cost, reported separately as "
            "model_load_time_s) and PNG decode (not part of the "
            "localization algorithm itself)")


def run_condition(model, cfg, device, condition, n):
    ds = LocalizerDataset("test", cfg, shuffle_buffer_size=1, **condition)
    errors, runtimes_ms = [], []
    worst = {"error": -1.0}
    for i, s in enumerate(ds):
        if i >= n:
            break
        ref = s["reference_img"].unsqueeze(0).to(device)
        search = s["search_img"].unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.predict(ref, search)
        runtimes_ms.append((time.perf_counter() - t0) * 1000.0)
        x, y = float(out["x"][0]), float(out["y"][0])
        conf = float(out["confidence"][0])
        gt_x, gt_y = float(s["gt_x"]), float(s["gt_y"])
        err = float(localization_error([x], [y], [gt_x], [gt_y])[0])
        errors.append(err)
        if err > worst["error"]:
            worst = {"error": err, "pred_x": x, "pred_y": y, "gt_x": gt_x,
                     "gt_y": gt_y, "confidence": conf,
                     "search_img": s["search_img"].squeeze(0).numpy()}
    errors = np.asarray(errors)
    result = {
        "n": len(errors),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "worst_error_px": float(errors.max()),
        "p90_error_px": float(np.percentile(errors, 90)),
        "runtime_ms_mean": float(np.mean(runtimes_ms)),
        "runtime_ms_median": float(np.median(runtimes_ms)),
    }
    for t in THRESHOLDS_PX:
        result[f"pass_rate@{t}px"] = float((errors <= t).mean())
    return result, worst


def render_failure_case(worst, condition, out_path):
    img = worst["search_img"]
    img = (img - img.min()) / max(float(img.max() - img.min()), 1e-6)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.scatter([worst["gt_x"]], [worst["gt_y"]], c="lime", marker="+", s=200,
              linewidths=2, label="true center")
    ax.scatter([worst["pred_x"]], [worst["pred_y"]], c="red", marker="x", s=200,
              linewidths=2, label="predicted center")
    ax.set_title(f"worst case: {condition['imaging_noise_profile']}/"
                f"{condition['geometric_profile']}, error={worst['error']:.1f}px, "
                f"confidence={worst['confidence']:.3f}")
    ax.legend(loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else (platform.processor() or "cpu")
    cfg = LocalizerConfig()

    t_load0 = time.perf_counter()
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model = DriftSenseLocalizer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.align_offset.fill_(ckpt["align_offset"])
    model.eval()
    load_time_s = time.perf_counter() - t_load0

    report = {
        "checkpoint": args.checkpoint, "device": device, "hardware": hardware,
        "python_version": sys.version, "timing_method": _timing_method_note(),
        "model_load_time_s": load_time_s, "conditions": {},
    }
    worst_overall, worst_condition = None, None
    for condition in CONDITIONS:
        key = f"noise={condition['imaging_noise_profile']}_geom={condition['geometric_profile']}"
        result, worst = run_condition(model, cfg, device, condition, args.n_per_condition)
        report["conditions"][key] = result
        print(f"{key}: n={result['n']} mean={result['mean_error_px']:.2f}px "
              f"median={result['median_error_px']:.2f}px "
              f"pass@5px={result['pass_rate@5.0px']:.3f} "
              f"pass@1px={result['pass_rate@1.0px']:.3f} "
              f"runtime_median={result['runtime_ms_median']:.1f}ms")
        if worst_overall is None or worst["error"] > worst_overall["error"]:
            worst_overall, worst_condition = worst, condition

    n_total = sum(c["n"] for c in report["conditions"].values())
    pooled_mean = sum(c["mean_error_px"] * c["n"]
                      for c in report["conditions"].values()) / n_total
    report["pooled"] = {"n_total": n_total, "mean_error_px": pooled_mean}

    failure_png = os.path.join(args.out_dir, "failure_case.png")
    render_failure_case(worst_overall, worst_condition, failure_png)
    if worst_overall["confidence"] < 0.15:
        root_cause = (
            f"Worst case ({worst_overall['error']:.1f}px error) occurred under "
            f"imaging_noise_profile={worst_condition['imaging_noise_profile']}, "
            f"geometric_profile={worst_condition['geometric_profile']}, with model "
            f"confidence={worst_overall['confidence']:.3f}. Low confidence at the "
            f"point of failure indicates the model itself flagged this as an "
            f"ambiguous/uncertain match (consistent with repeated-pattern "
            f"ambiguity or heavy noise obscuring the true structure), rather "
            f"than a confident-but-wrong error.")
    else:
        root_cause = (
            f"Worst case ({worst_overall['error']:.1f}px error) occurred under "
            f"imaging_noise_profile={worst_condition['imaging_noise_profile']}, "
            f"geometric_profile={worst_condition['geometric_profile']}, with model "
            f"confidence={worst_overall['confidence']:.3f}. The model was still "
            f"relatively confident despite the large error, suggesting a genuine "
            f"structural look-alike (repeated-pattern ambiguity) rather than an "
            f"easily-flagged low-confidence guess.")
    report["failure_case"] = {
        "error_px": worst_overall["error"], "confidence": worst_overall["confidence"],
        "condition": worst_condition, "image": "failure_case.png",
        "root_cause": root_cause,
    }

    with open(os.path.join(args.out_dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    md_lines = ["# Drift-Sense validation report", "",
               f"- checkpoint: `{args.checkpoint}`", f"- device: {hardware}",
               f"- python: {sys.version.split()[0]}",
               f"- timing method: {report['timing_method']}", "",
               "| condition | n | mean err (px) | median err (px) | worst err (px) | "
               "pass@5px | pass@4px | pass@2px | pass@1px | median runtime (ms) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in report["conditions"].items():
        md_lines.append(
            f"| {key} | {r['n']} | {r['mean_error_px']:.2f} | {r['median_error_px']:.2f} | "
            f"{r['worst_error_px']:.2f} | {r['pass_rate@5.0px']:.3f} | "
            f"{r['pass_rate@4.0px']:.3f} | {r['pass_rate@2.0px']:.3f} | "
            f"{r['pass_rate@1.0px']:.3f} | {r['runtime_ms_median']:.1f} |")
    md_lines += ["", f"**Pooled:** n={report['pooled']['n_total']}, "
                     f"mean error={report['pooled']['mean_error_px']:.2f}px", "",
                "## Failure case", "", "![failure case](failure_case.png)", "",
                root_cause]
    with open(os.path.join(args.out_dir, "validation_report.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nwrote {args.out_dir}/validation_report.json, "
          f"{args.out_dir}/validation_report.md, {args.out_dir}/failure_case.png")


if __name__ == "__main__":
    main()
