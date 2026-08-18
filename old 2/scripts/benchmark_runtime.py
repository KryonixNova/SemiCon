#!/usr/bin/env python3
"""Per-stage runtime breakdown, FP32 and FP16, batch size 1.

Methodology: torch.cuda.Event timing with explicit synchronisation, 20
warm-up iterations discarded, median of 100 runs.

Example:
    python scripts/benchmark_runtime.py
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from baseline_solution.zncc import zncc_match
from src.localizer.config import LocalizerConfig
from src.localizer.context_head import ContextHead
from src.localizer.correlation import dense_correlation
from src.localizer.decode import decode
from src.localizer.encoder import SiameseEncoder
from src.localizer.geometry import REF_PX, SEARCH_PX, SCALE

WARMUP, RUNS = 20, 100


def time_cuda(fn, warmup=WARMUP, runs=RUNS):
    """Median wall time in ms for `fn`, using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="./eval_results")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark")
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")

    results = {}
    for dtype, name in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        enc = SiameseEncoder().to(dev).eval().to(dtype)
        head = ContextHead().to(dev).eval().to(dtype)
        ref_raw = torch.randn(1, 1, REF_PX, REF_PX, device=dev, dtype=dtype)
        srch = torch.randn(1, 1, SEARCH_PX, SEARCH_PX, device=dev, dtype=dtype)

        with torch.no_grad():
            stages = {}
            stages["preprocess"] = time_cuda(lambda: torch.nn.functional.avg_pool2d(
                (ref_raw - ref_raw.mean()) / ref_raw.std(), SCALE))
            ref_ds = torch.nn.functional.avg_pool2d(ref_raw, SCALE)

            stages["encoder_search"] = time_cuda(lambda: enc(srch))
            stages["encoder_reference"] = time_cuda(lambda: enc(ref_ds))
            sf, rf = enc(srch), enc(ref_ds)

            stages["correlation"] = time_cuda(lambda: dense_correlation(sf, rf))
            corr = dense_correlation(sf, rf)

            stages["context_head"] = time_cuda(lambda: head(corr))
            hm, off = head(corr)
            hm = torch.sigmoid(hm).float()

            stages["decode"] = time_cuda(lambda: decode(hm, off.float()))
            stages["total"] = sum(stages.values())

        results[name] = stages
        print(f"\n--- {name} ---")
        for k, v in stages.items():
            print(f"  {k:<20} {v:>8.2f} ms")

    # B0 reference point, on CPU as it is actually deployed.
    rng = np.random.default_rng(0)
    ref_u8 = rng.integers(0, 255, (REF_PX, REF_PX), dtype=np.uint8)
    srch_u8 = rng.integers(0, 255, (SEARCH_PX, SEARCH_PX), dtype=np.uint8)
    t0 = time.time()
    for _ in range(10):
        zncc_match(ref_u8, srch_u8)
    results["B0_zncc_cpu_ms"] = (time.time() - t0) / 10 * 1000
    print(f"\nB0 ZNCC (CPU): {results['B0_zncc_cpu_ms']:.1f} ms")
    print(f"speedup vs B0 (fp16): "
          f"{results['B0_zncc_cpu_ms'] / results['fp16']['total']:.1f}x")

    with open(os.path.join(args.out_dir, "runtime.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_dir}/runtime.json")


if __name__ == "__main__":
    main()
