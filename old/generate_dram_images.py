"""
Generate N synthetic DRAM layout images (1000x1000 PNG) as grayscale
SEM-style renders, matched to the competition's synthetic-SEM format.

Usage:
    python generate_dram_images.py --n 50 --out-dir ./dram_dataset
    python generate_dram_images.py --n 200 --rows 128 --cols 128 --workers 8
    python generate_dram_images.py --n 50 --debug-images   # also save QA overlays

Each sample produces:
    <out-dir>/<i:04d>_seed<seed>.png        - 1000x1000 grayscale SEM image (MODEL INPUT)
    <out-dir>/<i:04d>_seed<seed>_meta.json  - full metadata (18 keys)
    <out-dir>/<i:04d>_seed<seed>_debug.png  - RGB layer view + GT overlay (only with --debug-images)

reference_center_px in the JSON is the localization ground truth. It is
NEVER drawn into the .png the model sees -- only into the optional _debug.png,
which exists for human QA and must not be used for training/eval.

Rendering pipeline (see review notes, 2026-08-06):
    GDS -> per-layer opaque binary masks, rendered at 4x supersampling
        -> grayscale intensity assignment (opaque painter's algorithm,
           no alpha blending -> no color-mixing artifacts)
        -> edge emphasis (SE yield is higher at edges/sidewalls)
        -> downsample 4x -> 1x (area-average antialiasing)
        -> Gaussian blur (finite beam spot)
        -> Poisson shot noise (SE emission statistics)
        -> Gaussian read/detector noise
        -> per-image contrast/brightness jitter
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

from dram_layout_generator import DRAMParams, DRAMGenerator
from sem_render import (
    IMG_SIZE, render_sem_image, render_debug_image,
)


def _generate_one(args: tuple) -> dict:
    """Worker function - generates one GDS, rasterizes it, saves PNG(s) + JSON."""
    idx, seed, params_kw, out_dir, save_debug = args

    stem      = f"{idx:04d}_seed{seed}"
    gds_path  = str(Path(out_dir) / f"{stem}.gds")
    json_path = str(Path(out_dir) / f"{stem}_meta.json")
    png_path  = str(Path(out_dir) / f"{stem}.png")

    p    = DRAMParams(**params_kw, seed=seed, output_gds=gds_path, output_json=json_path)
    gen  = DRAMGenerator(p)
    meta = gen.generate()

    rng = np.random.default_rng(seed)
    sem_img = render_sem_image(gds_path, meta, rng)
    Image.fromarray(sem_img, mode="L").save(png_path)

    if save_debug:
        debug_path = str(Path(out_dir) / f"{stem}_debug.png")
        dbg_img = render_debug_image(gds_path, meta)
        Image.fromarray(dbg_img, mode="RGB").save(debug_path)

    os.remove(gds_path)  # keep JSON + PNG(s), drop intermediate GDS

    return {
        "idx":   idx,
        "seed":  seed,
        "png":   png_path,
        "defects": len(meta["defect_locations"]),
        "ref_center_px": meta["reference_center_px"],
    }


def main():
    ap = argparse.ArgumentParser(description="Batch synthetic-SEM DRAM image generator")
    ap.add_argument("--n",           type=int,   default=20,         help="Number of images to generate")
    ap.add_argument("--out-dir",     type=str,   default="dram_dataset", help="Output directory")
    ap.add_argument("--seed-start",  type=int,   default=0,          help="First RNG seed (increments by 1)")
    ap.add_argument("--workers",     type=int,   default=4,          help="Parallel workers (set 1 to disable)")
    ap.add_argument("--debug-images", action="store_true",
                     help="Also save <stem>_debug.png with layer colors + GT overlay (QA only, not for training)")
    # Array geometry
    ap.add_argument("--rows",              type=int,   default=64)
    ap.add_argument("--cols",              type=int,   default=64)
    ap.add_argument("--cell-pitch-nm",     type=float, default=80.0)
    ap.add_argument("--cell-pitch-bl-nm",  type=float, default=60.0)
    ap.add_argument("--wl-width-nm",       type=float, default=24.0)
    ap.add_argument("--bl-width-nm",       type=float, default=18.0)
    ap.add_argument("--contact-size-nm",   type=float, default=14.0)
    ap.add_argument("--capacitor-size-nm", type=float, default=30.0)
    # Defect knobs
    ap.add_argument("--n-particles",       type=int,   default=3)
    ap.add_argument("--n-scratches",       type=int,   default=2)
    ap.add_argument("--p-broken-wl",       type=float, default=0.02)
    ap.add_argument("--p-broken-bl",       type=float, default=0.02)
    ap.add_argument("--p-cmp-dishing",     type=float, default=0.10)
    # Variation sigmas
    ap.add_argument("--overlay-sigma-nm",  type=float, default=3.0)
    ap.add_argument("--linewidth-sigma-nm",type=float, default=2.0)
    ap.add_argument("--ler-amplitude-nm",  type=float, default=1.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params_kw = dict(
        rows=args.rows,
        cols=args.cols,
        cell_pitch_nm=args.cell_pitch_nm,
        cell_pitch_bl_nm=args.cell_pitch_bl_nm,
        wl_width_nm=args.wl_width_nm,
        bl_width_nm=args.bl_width_nm,
        contact_size_nm=args.contact_size_nm,
        capacitor_size_nm=args.capacitor_size_nm,
        n_particles=args.n_particles,
        n_scratches=args.n_scratches,
        p_broken_wl=args.p_broken_wl,
        p_broken_bl=args.p_broken_bl,
        p_cmp_dishing=args.p_cmp_dishing,
        overlay_sigma_nm=args.overlay_sigma_nm,
        linewidth_sigma_nm=args.linewidth_sigma_nm,
        ler_amplitude_nm=args.ler_amplitude_nm,
    )

    work = [
        (i, args.seed_start + i, params_kw, str(out_dir), args.debug_images)
        for i in range(args.n)
    ]

    print(f"\nGenerating {args.n} images -> {out_dir}/")
    print(f"  Array : {args.rows}x{args.cols} cells  "
          f"({args.rows*args.cell_pitch_nm:.0f}x{args.cols*args.cell_pitch_bl_nm:.0f} nm)")
    print(f"  Workers: {args.workers}")
    print(f"  Debug images: {'on' if args.debug_images else 'off'}\n")

    results = []
    t0 = time.time()

    if args.workers == 1:
        for i, w in enumerate(work):
            r = _generate_one(w)
            results.append(r)
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (args.n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>{len(str(args.n))}}/{args.n}]  seed={r['seed']}  "
                  f"defects={r['defects']:>3}  "
                  f"ref=({r['ref_center_px'][0]:.0f},{r['ref_center_px'][1]:.0f})  "
                  f"eta {eta:.0f}s")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_generate_one, w): w for w in work}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                done += 1
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (args.n - done) / rate if rate > 0 else 0
                print(f"  [{done:>{len(str(args.n))}}/{args.n}]  seed={r['seed']}  "
                      f"defects={r['defects']:>3}  "
                      f"ref=({r['ref_center_px'][0]:.0f},{r['ref_center_px'][1]:.0f})  "
                      f"eta {eta:.0f}s")

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["idx"])

    summary_path = out_dir / "dataset_index.json"
    summary = {
        "n": args.n,
        "image_size_px": [IMG_SIZE, IMG_SIZE],
        "image_mode": "grayscale_sem",
        "array_rows": args.rows,
        "array_cols": args.cols,
        "samples": [
            {"idx": r["idx"], "seed": r["seed"],
             "png": os.path.basename(r["png"]),
             "meta": os.path.basename(r["png"]).replace(".png", "_meta.json"),
             "defects": r["defects"],
             "reference_center_px": r["ref_center_px"]}
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nDone: {args.n} images in {elapsed:.1f}s  ({elapsed/args.n:.2f}s/image)")
    print(f"   Output  : {out_dir}/")
    print(f"   Index   : {summary_path}")


if __name__ == "__main__":
    main()
