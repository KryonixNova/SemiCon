# Drift-Sense DRAM-SEM Reference Localizer

A deep-learning replacement for classical ZNCC template matching, for the task
of finding where a small **Reference** SEM image sits inside a larger
**Search** SEM image at 10x zoom difference (a Reference patch, imaged at
higher resolution, must be located within a wider Search image of the same
sample region imaged at lower resolution).

**Core idea:** dense correlation between Siamese-encoded features, followed by
a global-receptive-field context head to resolve the periodic-DRAM-lattice
ambiguity that a purely local encoder can't — a template can look identical
to several lattice repeats, and only wide spatial context (mat/die boundaries,
aperiodic imaging noise) disambiguates which repeat is the true one.

This folder is a **self-contained package**: everything here runs
independently, with no dependency on the original development repository.

---

## 1. What's in here

```
src/
  pipeline.py, sem_imaging.py, presets.py, structural_defects.py
                           # synthetic SEM canvas generator (shared infra)
  patterns/                # DRAM / FinFET / zone-routing pattern generators
  localizer/                # the model itself
    geometry.py            #   coordinate-mapping constants (scale, stride, offsets)
    config.py               #   LocalizerConfig — every tunable in one dataclass
    data.py                  #   on-the-fly training/val/test data generation
    encoder.py                #  Siamese ResNet-18 (destrided, no pretrained weights)
    correlation.py             # dense depthwise cross-correlation
    context_head.py             # 9-layer dilated conv stack (global receptive field)
    targets.py                   # Gaussian heatmap + sub-cell offset targets
    losses.py                     # focal heatmap loss, offset L1, hard-negative margin
    decode.py                      # heatmap -> (x, y, confidence)
    model.py                        # assembles everything into DriftSenseLocalizer
    metrics.py                       # acc@k, AP, PR curve
    inference.py                     # shared checkpoint-load + single-pair predict, used by predict.py and localize.py

baseline_solution/         # classical ZNCC (B0) and 10x10-grid (B1) baselines,
                            # kept for comparison — see scripts/evaluate_localizer.py

scripts/
  train.py                 # train a model
  predict.py                # run inference on a reference/search image pair
  evaluate_localizer.py      # compare B0/B1/B3 side by side
  validation_report.py        # spec-required threshold/runtime/failure-case report across noise x geometry conditions
  calibrate_tie_ratio.py        # calibrate decode's tie-break threshold on validation
  benchmark_runtime.py           # per-stage FP32/FP16 timing
  ablation_jitter.py              # A1: does the model rely on synthetic-noise fingerprints?
  ablation_negative.py             # A2: is confidence informative when the reference is absent?

generate_dataset.py         # persisted reference/search PNG pairs + manifest.csv, spec-compliant dataset generator
localize.py                  # single-pair or evaluator-batch inference, no source changes needed between modes

checkpoints/m3_hn_r24/best.pt   # a trained checkpoint (read the caveat in §6 before trusting numbers)

tests/                     # pytest suite (mirrors src/localizer/ + a generator sanity check)

docs/superpowers/
  specs/2026-08-08-dram-sem-reference-localization-design.md   # the design spec (architecture rationale)
  specs/2026-08-10-hackathon-compliance-design.md                # rotation/scale robustness + submission packaging design
  plans/2026-08-08-dram-sem-localizer.md                        # the full build plan (21 tasks)
  plans/2026-08-10-hackathon-compliance.md                       # the hackathon-compliance implementation plan
  plans/results-m1-m3.md                                          # milestone training results

generate_mxn_dram_dataset.py   # older standalone DRAM dataset generator (kept for compatibility, see §8)
references/CITATIONS.md         # public sources for DRAM structure, SEM noise modeling, and augmentation practice
results/                         # validation_report.py output (JSON + Markdown + failure-case PNG)
requirements.txt
pytest.ini
```

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.13.0+cu130` / `torchvision==0.28.0+cu130`.
Those are **local version labels** — `pip` can only resolve them from
PyTorch's own wheel index, not plain PyPI:

```bash
pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

If you're on a different CUDA version (or CPU-only), install whatever
`torch`/`torchvision` build matches your machine instead — nothing in this
code is tied to that specific build, it's just what was validated during
development. CPU-only works fine for `predict.py` on a single image pair;
training is impractically slow without a GPU.

Verify the install:

```bash
python -m pytest tests/ -q -m "not slow"
```

Should print `96 passed, 2 deselected` (the two `slow`-marked tests run a
real receptive-field measurement and a few live optimizer steps — include
them with `-m slow` or drop `-m "not slow"` entirely if you want the full
suite; expect ~40s instead of ~35s).

All commands below assume you're running from this folder's root, with the
venv active.

---

## 3. Quick start: predict on your own images

```bash
python scripts/predict.py \
    --checkpoint checkpoints/m3_hn_r24/best.pt \
    --reference path/to/reference.png \
    --search path/to/search.png \
    --verbose
```

Output:
```
predicted_x=484.77 predicted_y=898.32 confidence=0.0745
```

(`--verbose` prints labeled fields; omit it for machine-readable
`x,y,confidence` on one line.)

**Input requirements**, enforced by the script (it raises a clear error if
violated rather than failing deep inside the model):
- **Search image** must be exactly `1000x1000` px, grayscale.
- **Reference image** is auto-resized to `100x100` px — this assumes it's
  already at the correct physical scale (the Reference is imaged at 10x the
  Search image's resolution over the same real-world area). A reference at a
  genuinely different physical scale will be silently misinterpreted; there's
  no way for the script to detect that from pixel data alone.

Confidence is a **peak margin**, not a raw score (the classical ZNCC baseline
scored a misleadingly high 0.9+ even when wrong ~35% of the time, so absolute
score alone doesn't separate right from wrong here) — treat it as a ranking
signal across multiple predictions, not a calibrated probability. It can be
slightly negative when the decode logic's centre-tiebreak overrides raw peak
ranking; that's expected, not an error.

**Coordinate convention:** origin `(0, 0)` is the search image's top-left
corner; `x` increases rightward, `y` increases downward — standard image-array
convention, not math/plot convention. Predicted coordinates are always given
in **search-image pixels**, regardless of the reference's native resolution.

**Multiple matches:** if the reference pattern genuinely repeats within the
search image (a real possibility for periodic DRAM lattices), the decoder's
NMS + centre-tiebreak logic (`src/localizer/decode.py`) selects the
candidate closest to the search image's centre, matching the spec's
tie-break rule — this is inherent to the decode step, not a separate flag.

---

## 4. Training your own model

```bash
python scripts/train.py --run-name my_run --max-steps 40000
```

Key flags (see `python scripts/train.py --help` for the full list):

| Flag | Default | Notes |
|---|---|---|
| `--run-name` | *required* | checkpoints land in `checkpoints/<run-name>/best.pt` |
| `--max-steps` | `40000` | the real budget; expect several hours on a single consumer GPU |
| `--no-context` | off | trains the B2 ablation (no context head) instead of the real model |
| `--jitter-profile` | `normal` | `normal` / `zero` / `shifted` — see `ablation_jitter.py` below |
| `--imaging-noise-profile` | `normal` | `normal` / `harsh` — widens acquisition-noise and polygon-distortion knobs; see `src/localizer/data.py`'s `IMAGING_NOISE_PROFILES` |
| `--geometric-profile` | `normal` | `normal` / `drift` — adds ~1-2 degree rotation and 9:1-11:1 scale-ratio jitter to reference crops; see `src/localizer/data.py`'s `GEOMETRIC_PROFILES` |
| `--init-from` | *(none)* | warm-start weights from a different run's checkpoint, then train fresh from step 0 under this run's own schedule/profiles (for fine-tuning into a new profile after a prior run's LR schedule has already decayed) |
| `--lambda-hn` | `0.0` | hard-negative loss weight; the *provided checkpoint* was trained with `0.5` |
| `--hn-radius`, `--lr`, `--batch-size` | *(config default)* | override `LocalizerConfig`'s calibrated defaults if omitted |
| `--val-every` / `--val-batches` | `1000` / `40` | validation cadence and sample count |

Every validation step prints two lines:
```
[val] acc@50px 0.860  acc@5px 0.840  median 1.0px  AP 0.846
[val] std_x 258.40px  std_y 246.12px  distinct_preds 64/64
```
The second line is a **collapse monitor** — if `std_x`/`std_y` shrink toward
0 or `distinct_preds` shrinks toward 1, the model is predicting the same
point regardless of input (a real failure mode this architecture hit
multiple times during development; see the design/plan docs for the
post-mortems). A healthy model keeps both stds large and `distinct_preds`
near the sample count.

`--jitter-profile zero` trains against an exactly-periodic (zero aperiodic
noise) lattice — useful only for the A1 ablation below, not for a model you
intend to actually use, since it's a deliberately degenerate case.

### Robustness profiles

Two independent augmentation axes, both opt-in (default `normal` reproduces
original behavior exactly):

- **`--imaging-noise-profile harsh`** — wider acquisition-noise and
  polygon-distortion ranges (dose, drift, astigmatism, vignette, barrel
  distortion, charging streaks, speckle, salt-and-pepper, CD bias, corner
  rounding).
- **`--geometric-profile drift`** — the reference crop is scale-jittered
  (0.9-1.1x, i.e. an effective 9:1-11:1 relationship against the nominal
  10:1) and rotated (±2 degrees) *after* imaging, simulating a real
  capture's calibration/stage drift. Ground truth is never affected — only
  the reference's content is perturbed, the same way the position-jitter
  profiles already work.

Both were trained into `checkpoints/production_v2` via `--init-from`
fine-tuning after its original schedule had already fully decayed; see
`scripts/validation_report.py`'s output for per-condition accuracy.

---

## 5. Evaluating and comparing against baselines

```bash
python scripts/evaluate_localizer.py \
    --checkpoint checkpoints/m3_hn_r24/best.pt --n-samples 500
```

Runs B0 (classical ZNCC), B1 (10x10 grid search), and B3 (this model) on the
same canvas-disjoint test-split samples, writes `eval_results/comparison.json`,
and prints an accuracy table. Pass `--no-context` to evaluate a B2
(no-context-head) checkpoint instead of B3.

---

## 6. About the provided checkpoint — read before trusting its numbers

`checkpoints/m3_hn_r24/best.pt` was trained for **1000 steps**, not the
`40000`-step production budget above. This was a deliberate choice during
development: validate that the whole pipeline (data generation, loss,
training loop, decoding) works end-to-end and that loss decreases sensibly,
without spending hours of GPU time per experiment.

At 1000 steps it already reaches **acc@50px = 0.86–0.88** on held-out test
samples (see `docs/superpowers/plans/results-m1-m3.md` for the full
milestone table) — good enough to demo and sanity-check the pipeline, and
`hard_negative_radius_cells=24` (baked into this checkpoint) is a real,
validated result from a 4-way radius sweep (6/12/24/48), not a guess. But it
has **not** been run at the real budget, so don't treat its absolute accuracy
as representative of what the architecture can do — retrain with
`--max-steps 40000` (or higher) for a production-quality model.

---

## 7. Calibration and diagnostic scripts

**`calibrate_tie_ratio.py`** — `decode()`'s tie-break threshold
(`peak_tie_ratio`) is calibrated on validation data, never picked
arbitrarily. Re-run this after retraining if you want a freshly-calibrated
threshold instead of the shipped default (`0.98`):
```bash
python scripts/calibrate_tie_ratio.py --checkpoint checkpoints/m3_hn_r24/best.pt
```
Prints a sensitivity curve over τ ∈ [0.80, 1.00] and the selected value —
update `LocalizerConfig.peak_tie_ratio` in `src/localizer/config.py` by hand
if you want to bake in a new value for future runs.

**`benchmark_runtime.py`** — per-stage FP32/FP16 timing (encoder, correlation,
context head, decode), CUDA-event-measured, compared against the classical
ZNCC baseline:
```bash
python scripts/benchmark_runtime.py
```

**`ablation_jitter.py`** (A1) — tests whether the model depends on
generator-specific noise fingerprints rather than genuine aperiodicity. Needs
a second checkpoint trained with `--jitter-profile zero` to fill in the
`zero`-trained row:
```bash
python scripts/train.py --run-name zero_jitter_model --jitter-profile zero --max-steps 40000
python scripts/ablation_jitter.py \
    --checkpoint checkpoints/m3_hn_r24/best.pt \
    --checkpoint-zero checkpoints/zero_jitter_model/best.pt
```
Prints a 2x3 train-profile x test-profile accuracy matrix. Read the script's
own docstring for how to interpret the two "drop" figures — a large
normal→zero drop is expected (zero-jitter data is genuinely
information-theoretically degenerate), a large normal→shifted drop would be
the actual red flag (would mean the model overfit one specific noise
distribution instead of aperiodicity in general).

**`ablation_negative.py`** (A2, diagnostic — gates nothing) — pairs each
Search image with a Reference from a genuinely unrelated canvas, measuring
whether `confidence` is a real signal for "this reference isn't actually
here" (which the classical ZNCC baseline is bad at — it scored misleadingly
high even ~35% of the time it was wrong):
```bash
python scripts/ablation_negative.py --checkpoint checkpoints/m3_hn_r24/best.pt
```
Prints an AUROC; well above 0.5 means the margin-based confidence is
genuinely informative, near 0.5 means it isn't.

---

## 8. Generating a persisted dataset

```bash
python generate_dataset.py --split test --num-samples 30 --output-dir ./output
```

Writes `output/reference/*.png`, `output/search/*.png`, and
`output/manifest.csv` (ground truth + generation metadata per pair), drawing
from the same on-the-fly generator (`src/localizer/data.py`) that training
itself uses — so a dataset written here is representative of what the model
was actually trained/evaluated on. `--split` picks a canvas-disjoint seed
range (`train`/`val`/`test`, matching `LocalizerConfig`), and
`--imaging-noise-profile`/`--geometric-profile` control the same robustness
axes described in §4.

Note `scripts/train.py`/`evaluate_localizer.py`/the ablation scripts don't
need a persisted dataset at all — they generate data on-the-fly from a seed,
which is why they never take a `--data-dir` flag. This script exists for
external tooling and for the hackathon submission's manifest requirement.

An older, independently-written generator, `generate_mxn_dram_dataset.py`,
still exists alongside this one (different manifest columns, no robustness
profiles) — kept for backward compatibility, not the recommended entry
point going forward.

## 9. Batch localization

```bash
python localize.py --checkpoint checkpoints/production_v2/best.pt \
    --manifest output/manifest.csv --output predictions.csv
```

Reads `reference_path`/`search_path` columns from any manifest (including
one an evaluator supplies, or `generate_dataset.py`'s own output) and writes
`predictions.csv` with `id, predicted_x, predicted_y, confidence,
runtime_ms` — one row per input pair, no source-code edits needed between a
single pair and a full batch. Single-pair mode
(`--reference X --search Y`) matches `scripts/predict.py`'s exact CLI
contract, kept for compatibility.

## 10. Validation report

```bash
python scripts/validation_report.py \
    --checkpoint checkpoints/production_v2/best.pt \
    --n-per-condition 50 --out-dir results
```

Runs the model across all four `{imaging_noise_profile} x
{geometric_profile}` combinations and writes `results/validation_report.md`
(human-readable per-condition table: mean/median/worst Euclidean error,
pass rate @5/4/2/1px, median runtime), `results/validation_report.json`
(the same data, machine-readable), and `results/failure_case.png` (the
single worst prediction across all conditions, with true/predicted centres
marked and a root-cause note in the report).

---

## 11. Further reading

- `docs/superpowers/specs/2026-08-08-dram-sem-reference-localization-design.md`
  — the architecture rationale: why dense correlation + a context head, why
  a Gaussian heatmap target, why the loss is structured the way it is.
- `docs/superpowers/plans/2026-08-08-dram-sem-localizer.md` — the full build
  plan, including exact coordinate-mapping math, every configurable
  hyperparameter, and the milestone accuracy gates.
- `docs/superpowers/plans/results-m1-m3.md` — the actual milestone training
  results (with the scaled-down-run caveat spelled out explicitly).
- `docs/superpowers/specs/2026-08-10-hackathon-compliance-design.md` and
  `docs/superpowers/plans/2026-08-10-hackathon-compliance.md` — the design
  and implementation plan behind §4's robustness profiles and §8-10's
  submission-shaped scripts.
- `references/CITATIONS.md` — public sources backing the DRAM structure,
  SEM noise modeling, and scale/rotation augmentation design choices.
- `src/localizer/decode.py` and `src/localizer/model.py`'s module
  docstrings/comments explain several non-obvious design decisions in place
  (why confidence can go negative, why sigmoid must always be applied, why
  the centre-tiebreak never appears in any loss term) — worth reading before
  modifying either file.
