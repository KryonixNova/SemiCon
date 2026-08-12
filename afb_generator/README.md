# AFB's DRAM SEM generator (generator-only)

Procedural synthetic DRAM SEM reference/search image-pair generator,
extracted from AFB's original `dram_localization_pipeline.py` for the
KryonixNova submission. See `../docs/afb-dram-localization-pipeline-guide.pdf`
and `../docs/afb-dram-pipeline-usage-guide.pdf` for the full, from-scratch
conceptual walkthrough (what a GDS file is, what an SEM image is, what the
DRAM structures mean) — this file is a short, submission-specific reference
covering what's actually present in this folder and how it differs from
those guides' original pipeline.

> **Note on scope.** AFB's original pipeline (as described in the two guide
> PDFs above) also included a SuperPoint+LightGlue matcher and a RANSAC
> localizer (sections 4-6 of the original single-file build) plus a
> benchmark CLI that exercised them. None of that is in this copy: only the
> generator (GDS layout generation, SEM-style rasterization, and
> reference/search sampling — sections 1-3) is included. puthere's
> deep-learned model (`../puthere/`) is this submission's one localization
> approach, so `afb_generator/` has no torch/lightglue dependency and the
> guide PDFs' matcher/benchmark-CLI/KLayout-inspection sections don't apply
> here. See `../README.md`'s "Design notes" for why.

## What it generates

`generate_sample()` invents a small, fictional DRAM memory array (rows/
columns of storage cells wired with word lines, bit lines, and contacts),
lays it out as GDS chip geometry, renders it as a grayscale SEM-style
image, and cuts out a **reference** patch and the full **search** image it
came from, with the reference patch's exact location recorded as ground
truth. None of the modeled structures are proprietary — see the module
docstring at the top of `dram_localization_pipeline.py` for the specific
public references (Dennard 1968, Sze & Ng 2006, Razavi 2002, Jaeger 2002,
Postek & Vladar 2011, JEDEC JESD79) each structure is grounded in.

Beyond the single fixed geometry, the generator also supports (see the
guide PDFs and `dram_localization_pipeline.py`'s docstrings for full
detail on each):

- **`DRAM_PRESETS`** — six named density variants (`dram_1x`, `dram_dense`,
  `dram_loose`, `dram_wide`, `dram_compact`, `dram_legacy`).
- **Multi-region dies** — an `N×M` grid of independently re-randomized
  DRAM blocks composited into one die, via `die_block_rows`/`die_block_cols`.
- **`block_presets`** — mixed-density dies, where each block in a
  multi-region die independently draws its own preset.
- **`boundary_bias`** — probability that a reference-patch crop is
  deliberately chosen to straddle a block/street seam.
- **True zoom** — `reference_width_px`/`zoom_ratio` control rendering the
  reference patch at higher pixel density than the search image (this
  submission's `generate_dataset.py` CLI always uses `zoom_ratio=10`).
- **Single training defect mode** — `single_training_defect=True` places
  exactly one small diagonal defect on the die (all other defect types
  suppressed) with its location and containing tile recorded as ground
  truth.

## Generating a dataset split

```bash
python generate_dataset.py --num-samples 20 --split train --output-dir ./output --seed 42 \
    --presets dram_1x dram_dense
```

Writes `output/train/reference/*.png`, `output/train/search/*.png`, and
`output/train/manifest.csv`. See `generate_dataset.py --help` for the full
flag list.

## Testing

```bash
python -m pytest tests/ -v
```

## Project layout

```
dram_localization_pipeline.py   generator code (GDS layout, rasterization, sampling)
generate_dataset.py             CLI: generate a PNG pair + manifest.csv dataset split
tests/                          pytest suite (44 tests)
```
