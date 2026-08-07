# DRAM Localization Pipeline
# SemiCon
# DRAM Synthetic SEM + Reference/Search Localization Benchmark

Synthetic DRAM SEM-image generation, defect injection, and
SuperPoint+LightGlue localization benchmarking — a single-file,
dependency-ordered build.

**New to this project?** Read
[`dram_localization_pipeline_guide.pdf`](dram_localization_pipeline_guide.pdf)
first — it explains every concept from scratch (what a GDS file is, what an
SEM image is, what the DRAM structures mean) and walks through the pipeline
step by step. This README is a fast reference for people who already know
the codebase.

## What it does, in one paragraph

The script invents a small, fictional DRAM memory array (rows/columns of
storage cells wired up with word lines, bit lines, and contacts), lays it
out as real chip geometry (a GDS file, the format chip designers use),
renders that geometry as a grayscale image that looks like an SEM
(scanning electron microscope) photo, and cuts out two images from it: a
small **reference** patch and the full **search** image it came from — with
the reference patch's exact location recorded as ground truth. Those pairs
are the training/benchmark data for a localization task: "given this small
patch, find where it goes in the big image." The file also includes a
pretrained SuperPoint+LightGlue matcher and RANSAC localizer to actually
attempt that task and score itself against the recorded ground truth.

None of the modeled structures are proprietary — capacitor nodes, word
lines, bit lines, and contacts are textbook 1T-1C DRAM concepts with public
references cited in the module docstring. `cell_pitch_nm` and friends are
free parameters; no real process node is implied.

## The six sections (in file order)

1. **GDS layout generation** — array geometry + process variation + defects
2. **SEM-style rasterization** — GDS → grayscale raster image
3. **Reference/search sampling** — `generate_sample()`, the dataset entry point
4. **SuperPoint + LightGlue matching** — keypoint extraction + matching
5. **RANSAC localization** — matches → one estimated location
6. **Benchmark CLI** — runs many samples, reports success rate / pixel error

## Setup

Everything runs in the `royl` conda environment (KLayout, OpenCV, NumPy,
PyTorch, LightGlue, pytest):

```bash
conda activate royl
```

First run of the matcher downloads SuperPoint/LightGlue weights via
`torch.hub` (cached afterward in `~/.cache/torch/hub/checkpoints/`) —
needs network access once.

## Quick start — generate one sample

```python
from dram_localization_pipeline import generate_sample

sample = generate_sample(seed=1, tmp_dir="/tmp/dram_gen")

sample.reference_img      # uint8 grayscale, 100x100
sample.search_img         # uint8 grayscale, 1000x1000
sample.true_center_px     # ground-truth (cx, cy) of the patch in search_img
```

`tmp_dir` is scratch space for the intermediate GDS/JSON files — they're
deleted automatically before `generate_sample` returns. Same `seed` always
produces the exact same output (fully deterministic — see "Current render
state" below).

## Current render state: no noise, no blur

The image renderer (`render_sem_image`/`render_sem_patch`) currently applies
only two things:

1. A deterministic per-layer intensity fill (GDS geometry → flat gray levels)
2. A deterministic Sobel-based **edge enhancement** — brightens shape
   boundaries, mimicking the real SEM "edge brightening" effect

There is **no** beam blur, shot/read noise, or gain/offset jitter right now
— those were deliberately stripped out to get a clean, fully deterministic
baseline. The `rng` parameter is still threaded through both render
functions (currently unused) so noise can be reintroduced later without an
API change.

## The array is square by default — fills the canvas, no margin

`pixels_per_nm` is chosen uniformly for both axes (so cells aren't
stretched), which means if the array's physical footprint isn't square,
one axis leaves an empty margin. The defaults are tuned to avoid this:

- Single-block default: `rows=64, cols=85` → `cols*cell_pitch_bl_nm ≈
  rows*cell_pitch_nm` (both ≈5100-5120nm) → square → fills 1000x1000 exactly.
- Multi-block default: `die_block_width_nm = die_block_height_nm = 5120.0`
  → each block is square too.

Square blocks only make the **total die** square when `die_block_rows ==
die_block_cols`. For a rectangular grid, use `square_die_block_width_nm()`
to solve for the block width that keeps the total square:

```python
from dram_localization_pipeline import square_die_block_width_nm, generate_sample

rows, cols, street, block_h = 2, 4, 40.0, 5120.0
block_w = square_die_block_width_nm(rows, cols, street, block_h)

sample = generate_sample(seed=1, tmp_dir="/tmp/dram_gen",
                          die_block_rows=rows, die_block_cols=cols,
                          die_block_width_nm=block_w, die_block_height_nm=block_h,
                          die_street_width_nm=street)
```

Any margin that *does* remain (e.g. if you override these on purpose) is
left as flat background — never a repeated/tiled pattern, so it never looks
like fabricated content.

## Multi-region dies (multiple blocks per die)

```python
sample = generate_sample(seed=1, tmp_dir="/tmp/dram_gen",
                          die_block_rows=3, die_block_cols=3)
```

Composits an `N×M` grid of independently re-randomized DRAM blocks
(`die_block_variation_frac`, default ±40%) into one die, separated by a
street layer. `die_block_rows=die_block_cols=1` (the default) is an exact
passthrough — identical output to not using this feature at all.

## Single training defect mode

Opt-in dataset mode for training a defect-detection model: exactly one
small diagonal defect is placed on the die (all other defect types —
particles, scratches, CMP dishing, missing capacitors/contacts, broken
word/bit lines — are suppressed everywhere on the die), and its exact
location and containing tile (10x10 grid of 100x100px tiles over the
1000x1000 search image) are recorded as ground truth. Works with any grid
size, including multi-region dies.

```python
sample = generate_sample(seed=1, tmp_dir="/tmp/dram_gen",
                          single_training_defect=True)
# or combined with a multi-region die:
sample = generate_sample(seed=1, tmp_dir="/tmp/dram_gen",
                          die_block_rows=4, die_block_cols=4,
                          single_training_defect=True)

sample.training_defect_center_px   # (cx, cy) in search-image px
sample.training_defect_bbox_px     # (x0, y0, x1, y1) in search-image px
sample.training_defect_tile        # (row, col), each 0-9
```

## Benchmark CLI

Run the full matching + localization benchmark:

```bash
python dram_localization_pipeline.py --n 30 --tolerance-px 5.0 --out-dir localization_results
```

| Flag | Default | Meaning |
|---|---|---|
| `--n` | 30 | number of reference/search pairs to benchmark |
| `--tolerance-px` | 5.0 | max pixel error counted as a localization success |
| `--seed-start` | 0 | first random seed used (sample *i* uses `seed_start + i`) |
| `--n-examples` | 5 | number of annotated example PNGs to save |
| `--out-dir` | `localization_results` | output directory for results + examples |

Writes `benchmark_results.json` (success rate, pixel error, timing stats)
and a handful of annotated example PNGs to `--out-dir`.

## Inspecting a real GDS file in KLayout

`generate_sample()` deletes its intermediate GDS/JSON files. To get a real
`.gds` file to open in KLayout yourself:

```python
from dram_localization_pipeline import DRAMParams, generate_dram_layout

p = DRAMParams(output_gds="my_array.gds", output_json="my_array.json", seed=1)
generate_dram_layout(p)
```

```bash
klayout my_array.gds
```

## Testing

```bash
conda run -n royl python -m pytest test_dram_localization_pipeline.py -v
```

## Project layout

```
dram_localization_pipeline.py         single-file pipeline (this repo's only source file)
test_dram_localization_pipeline.py    pytest suite
dram_localization_pipeline_guide.pdf  full beginner-friendly walkthrough
docs/superpowers/specs/               design specs for past features
docs/superpowers/plans/               implementation plans for past features
```
