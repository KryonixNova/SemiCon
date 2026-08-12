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

> **Note on this copy.** This is `puthere`'s generator + localizer as
> consolidated into the **KryonixNova** submission repo, one level under
> that repo's root (`KryonixNova/puthere/`). A few things that lived
> alongside this code in the original development repository were
> intentionally not brought over — `baseline_solution/` (the classical
> ZNCC/grid baselines; a from-scratch deep-learned localizer is this
> submission's one localization approach, so baseline comparison is out of
> scope here), `docs/superpowers/` (internal design/plan docs), and
> `references/CITATIONS.md` (its content now lives in KryonixNova's
> top-level `README.md`, under "Citations"). See `../README.md` for the
> whole-submission overview. Everything else below still applies as
> written, run from *this* folder (`puthere/`) unless a path is explicitly
> given as `../...`.

`../solution_presentation.pptx` (repo root, shared across both
`afb_generator/` and `puthere/`) is the mandatory hackathon submission
slide deck (12 slides: problem, workflow, dataset design, noise/augmentation
citations, method, execution commands, experiments, results, robustness/
ablations, an honest failure case, conclusion).

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

scripts/
  train.py                 # train a model
  predict.py                # run inference on a reference/search image pair
  validation_report.py        # spec-required threshold/runtime/failure-case report across noise x geometry conditions
  calibrate_tie_ratio.py        # calibrate decode's tie-break threshold on validation
  ablation_jitter.py              # A1: does the model rely on synthetic-noise fingerprints?
  ablation_negative.py             # A2: is confidence informative when the reference is absent?

generate_dataset.py         # persisted reference/search PNG pairs + manifest.csv, spec-compliant dataset generator
localize.py                  # single-pair or evaluator-batch inference, no source changes needed between modes

checkpoints/production_v3/best.pt   # the shipped/default model -- see §5 for the full lineage
checkpoints/production_v5/best.pt   # AFB-real-data specialist -- strong there, weak on synthetic harsh noise (see §5)

tests/                     # pytest suite (mirrors src/localizer/ + a generator sanity check)

requirements.txt
pytest.ini

# one level up, at the KryonixNova repo root (not inside this folder):
../results/                # validation_report.py output (JSON + Markdown + failure-case PNG)
  sample_dataset/             # 32-pair generate_dataset.py + localize.py example (see §9)
../solution_presentation.pptx / .pdf   # the mandatory hackathon slide deck
../docs/puthere-drift-sense-localizer-guide.pdf     # conceptual walkthrough (see §10)
../docs/puthere-ryzen-cpu-testing-guide.pdf         # CPU-only inference guide (see §10)
```

Not present in this copy (present in the original `puthere` development
repository, not part of the KryonixNova submission — see the note above):
`baseline_solution/` (classical ZNCC/grid baselines), `docs/superpowers/`
(design/plan docs), `generate_mxn_dram_dataset.py` (an older, superseded
dataset generator), `references/CITATIONS.md` (superseded by
`../README.md`'s "Citations" section), `scripts/evaluate_localizer.py` and
`scripts/benchmark_runtime.py` (both depended on `baseline_solution/`).

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

Should print `121 passed, 15 deselected` (the `slow`-marked tests run real
subprocess training/inference runs, live optimizer steps, and a receptive-
field measurement — include them with `-m slow` or drop `-m "not slow"`
entirely for the full suite; ~5 minutes instead of under a minute). This
count is 4 lower than the original development repository's because
`tests/localizer/test_grid_baseline.py` (which exercises
`baseline_solution/`, not copied into this submission — see the note at
the top of this file) isn't present here.
`tests/localizer/test_model.py::test_predictions_lie_inside_the_valid_range`
is a known pre-existing flaky test (unseeded random weight init, not related
to any change here) — if it fails, re-run just that test in isolation to
confirm it's this, not a real regression.

All commands below assume you're running from this folder's root, with the
venv active.

---

## 3. Quick start: predict on your own images

```bash
python scripts/predict.py \
    --checkpoint checkpoints/production_v3/best.pt \
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
multiple times during development; see the original development repository's
design/plan docs for the post-mortems, not included in this copy). A healthy
model keeps both stds large and `distinct_preds`
near the sample count.

**Automatic safety nets against training collapse.** Three layers, run
automatically, no flag needed:
1. **Non-finite input skip** — a rare edge case in the synthetic imaging
   pipeline can occasionally emit an inf/nan pixel value under
   `--imaging-noise-profile harsh`. If either input tensor for a batch isn't
   fully finite, that batch is skipped *before* it ever reaches the model
   (so it can't corrupt anything), printed as `WARNING: non-finite input at
   step N -- skipping this batch`, and doesn't count toward
   `--steps-this-run`/`--max-steps`.
2. **Non-finite loss skip** — belt-and-suspenders: if the loss itself comes
   out non-finite even from finite inputs, the optimizer step is skipped
   (`WARNING: non-finite loss at step N -- skipping optimizer step`).
3. **Auto-rollback on collapsed validation** — layers 1-2 only stop *weight*
   corruption via backward()/optimizer.step(); they can't stop a forward
   pass from updating BatchNorm's running mean/var, which happens
   regardless. If a validation ever comes back with `nan` predictions (the
   BatchNorm-corruption signature), training automatically reloads
   model/optimizer/scheduler/scaler from the last known-good checkpoint and
   continues from there (`WARNING: validation collapsed (nan predictions)
   -- rolling back ...`) — nothing gets saved over a collapsed state, and
   you don't need to notice and intervene manually.

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

`checkpoints/production_v3` is the result of a training lineage: an
original 40000-step run, fine-tuned under `--imaging-noise-profile harsh`
alone (via `--init-from` after its own schedule had fully decayed), then
continued again the same way adding `--geometric-profile drift` on top --
so the final weights are trained under both robustness axes together. Only
the final checkpoint is kept on disk; the intermediate stage isn't shipped
separately. See `scripts/validation_report.py`'s output for per-condition
accuracy.

`checkpoints/production_v5` is an exact-weights copy of `production_v3`,
further fine-tuned via `scripts/finetune_on_manifest.py` on 5000 real
AFB-rendered samples (`dram_loose`/`dram_wide`/`dram_legacy` presets --
AFB is an independent DRAM-SEM generation pipeline, not this project's own
synthetic generator, so this is genuinely out-of-distribution real data,
not more of the same synthetic source). On an independent 2000-sample AFB
test set spanning all six presets, `production_v5` reaches 100% accuracy at
every tested tolerance (down to 1px) with ~0.14px mean error -- including on
the three presets it was never fine-tuned on, i.e. genuine generalization
to real imagery, not memorization of the fine-tuning set.

**But it is not the default, and this is an honest documented trade-off,
not an oversight.** Running `scripts/validation_report.py` -- this
project's own noise x geometry validation matrix, the same methodology the
submission spec requires -- against both checkpoints tells a different
story:

| checkpoint | normal/normal | harsh/normal | normal/drift | harsh/drift | pooled mean |
|---|---|---|---|---|---|
| `production_v3` | mean 0.64px, pass@5px 1.000 | mean 4.32px, pass@5px 0.640 | mean 0.67px, pass@5px 1.000 | mean 4.28px, pass@5px 0.640 | **2.47px** |
| `production_v5` | mean 0.86px, pass@5px 1.000 | mean 80.79px, pass@5px 0.260 | mean 0.86px, pass@5px 1.000 | mean 81.71px, pass@5px 0.260 | **41.06px** |

Fine-tuning on real AFB imagery bought genuine cross-domain generalization
but cost most of `production_v3`'s harsh-imaging-noise robustness -- the
same catastrophic-forgetting pattern an earlier, narrower preset-restricted
fine-tune attempt (not shipped) already showed on a different axis, just
recovered here on the *noise* axis instead of the *preset* axis. Since the
actual sponsor test data's noise characteristics aren't known in advance,
`production_v3`'s consistent behavior across every tested condition is the
safer default for the graded submission. `production_v5` is kept and
documented because it's a genuinely interesting, real result -- strong
evidence the architecture generalizes to real SEM-style imagery, not just
its own synthetic generator -- just not the checkpoint this repo bets the
submission score on.

---

## 5. About the provided checkpoints

Two checkpoints ship in `checkpoints/`, both real, full-budget (40000-step)
training runs -- no scaled-down pipeline-sanity-check checkpoint is
included:

- **`production_v3`** (the default — see §4 above) — reached
  acc@50px=0.981 at step 40000 under `--imaging-noise-profile harsh` +
  `--geometric-profile drift` (see §4's Robustness profiles for the full
  training lineage). `hard_negative_radius_cells=24` (baked into both
  checkpoints below) is a real, validated result from a 4-way radius sweep
  (6/12/24/48), not a guess. Consistently strong across every condition in
  `scripts/validation_report.py`'s noise x geometry matrix (pooled mean
  error 2.47px).
- **`production_v5`** — fine-tunes `production_v3` on 5000 real
  AFB-rendered samples and reaches 100% accuracy at every tested tolerance
  (down to 1px) on an independent 2000-sample AFB test set spanning all six
  DRAM presets, including the three it was never fine-tuned on. But it
  loses most of `production_v3`'s harsh-noise robustness in the process
  (pooled mean error 41.06px on the same validation matrix) -- see §4 for
  the full comparison and why this makes it the wrong default for the
  graded submission despite the strong AFB result.

Use `production_v3` for the submission; evaluate `production_v5` if you
specifically want to see how the architecture performs on real,
independently-rendered SEM-style imagery instead of this project's own
synthetic generator.

---

## 6. Calibration and diagnostic scripts

**`calibrate_tie_ratio.py`** — `decode()`'s tie-break threshold
(`peak_tie_ratio`) is calibrated on validation data, never picked
arbitrarily. Re-run this after retraining if you want a freshly-calibrated
threshold instead of the shipped default (`0.98`):
```bash
python scripts/calibrate_tie_ratio.py --checkpoint checkpoints/production_v3/best.pt
```
Prints a sensitivity curve over τ ∈ [0.80, 1.00] and the selected value —
update `LocalizerConfig.peak_tie_ratio` in `src/localizer/config.py` by hand
if you want to bake in a new value for future runs.

**`ablation_jitter.py`** (A1) — tests whether the model depends on
generator-specific noise fingerprints rather than genuine aperiodicity. Needs
a second checkpoint trained with `--jitter-profile zero` to fill in the
`zero`-trained row:
```bash
python scripts/train.py --run-name zero_jitter_model --jitter-profile zero --max-steps 40000
python scripts/ablation_jitter.py \
    --checkpoint checkpoints/production_v3/best.pt \
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
python scripts/ablation_negative.py --checkpoint checkpoints/production_v3/best.pt
```
Prints an AUROC; well above 0.5 means the margin-based confidence is
genuinely informative, near 0.5 means it isn't.

---

## 7. Generating a persisted dataset

```bash
python generate_dataset.py --split test --num-samples 30 --output-dir ./output
```

Writes `output/reference/*.png` (1000x1000, the "100x close-up view" format
the submission spec requires — upscaled from the model's native 100x100
representation, so `localize.py`'s downsample-back-to-100x100 round-trips
almost exactly), `output/search/*.png` (1000x1000, the 10x view), and
`output/manifest.csv` (`architecture`, ground truth, and generation
metadata per pair — random seed, transformations, noise settings, scale,
rotation), drawing from the same on-the-fly generator (`src/localizer/data.py`)
that training itself uses — so a dataset written here is representative of
what the model was actually trained/evaluated on. `--split` picks a
canvas-disjoint seed range (`train`/`val`/`test`, matching
`LocalizerConfig`), and `--imaging-noise-profile`/`--geometric-profile`
control the same robustness axes described in §4.

Note `scripts/train.py`/the ablation scripts don't need a persisted dataset
at all — they generate data on-the-fly from a seed, which is why they never
take a `--data-dir` flag. This script exists for external tooling and for
the hackathon submission's manifest requirement.

## 8. Batch localization

```bash
python localize.py --checkpoint checkpoints/production_v3/best.pt \
    --manifest output/manifest.csv --output predictions.csv
```

Reads `reference_path`/`search_path` columns from any manifest (including
one an evaluator supplies, or `generate_dataset.py`'s own output) and writes
`predictions.csv` with every input column carried through (ground truth,
generation metadata, whatever the manifest already has) plus `predicted_x,
predicted_y, confidence, runtime_ms` appended — one self-contained row per
input pair, no source-code edits needed between a single pair and a full
batch. Single-pair mode (`--reference X --search Y`) matches
`scripts/predict.py`'s exact CLI contract, kept for compatibility.
`--checkpoint` defaults to `checkpoints/production_v3/best.pt` (resolved
relative to `localize.py` itself, so it works from any working directory)
— see §5 for why it's the safer default over `production_v5`; only pass
`--checkpoint` explicitly to point at a different one.

**Phase 2 submission contract** (single reference/search pair in, exactly
one `x,y` coordinate out, nothing else on stdout):

```bash
python localize.py --reference ref.png --search search.png --xy-only
```

## 9. Validation report

```bash
python scripts/validation_report.py \
    --checkpoint checkpoints/production_v3/best.pt \
    --n-per-condition 50 --out-dir ../results
```

`--out-dir` is `../results` (not `results`) because, in this KryonixNova
copy, the committed `results/` directory lives at the repo root, one level
above this `puthere/` folder — not inside it. Running the command above
regenerates it in place; run with a different `--out-dir` if you'd rather
not overwrite the committed copy.

Runs the model across all four `{imaging_noise_profile} x
{geometric_profile}` combinations and writes `../results/validation_report.md`
(human-readable per-condition table: mean/median/worst Euclidean error,
pass rate @5/4/2/1px, median runtime), `../results/validation_report.json`
(the same data, machine-readable), and `../results/failure_case.png` (the
single worst prediction across all conditions, with true/predicted centres
marked and a root-cause note in the report). Already run against
`production_v3` and committed — pooled mean error 2.47px over 200 samples.
`../results/success_case.png` (the same true/predicted-centre visualization,
for one of the cleanest predictions in `../results/sample_dataset/` below —
0.24px error) is committed alongside it as the success-case counterpart
used in `../solution_presentation.pptx`/`.pdf`.

`../results/sample_dataset/` is a committed, concrete example of the full
generate → predict pipeline: 32 pairs (8 each across all four noise x
geometry conditions), generated via `generate_dataset.py` and run through
`localize.py --manifest`. `../results/sample_dataset/predictions.csv` is
the combined output — every generation column (architecture, ground truth,
seed, noise/geometry profile, scale, rotation) alongside the model's
`predicted_x`/`predicted_y`/`confidence`/`runtime_ms` for the same 32 rows,
in one file. Reproduce it directly:

```bash
python generate_dataset.py --split test --num-samples 8 --seed 200000 \
    --output-dir /tmp/ds_nn --imaging-noise-profile normal --geometric-profile normal
# ...repeat with --seed 200100/harsh/normal, 200200/normal/drift, 200300/harsh/drift...
python localize.py --checkpoint checkpoints/production_v3/best.pt \
    --manifest <merged manifest> --output predictions.csv
```

(mean error 2.00px, pass@5px 0.906 on this specific 32-pair sample.)

---

## 10. Further reading

- `../docs/puthere-drift-sense-localizer-guide.pdf` — a from-scratch
  conceptual walkthrough of the whole pipeline (what a reference/search
  pair is, how the model works, how to read the metrics).
- `../docs/puthere-ryzen-cpu-testing-guide.pdf` — running inference (not
  training) on a plain CPU machine with no NVIDIA GPU, for testing on
  hardware that doesn't match the original development environment.
- KryonixNova's top-level `../README.md` — the "Citations" section there
  covers the public sources backing the DRAM structure, SEM noise
  modeling, and scale/rotation augmentation design choices (this copy
  doesn't include the original development repo's `references/CITATIONS.md`
  or `docs/superpowers/` design/plan docs — see the note at the top of this
  file).
- `src/localizer/decode.py` and `src/localizer/model.py`'s module
  docstrings/comments explain several non-obvious design decisions in place
  (why confidence can go negative, why sigmoid must always be applied, why
  the centre-tiebreak never appears in any loss term) — worth reading before
  modifying either file.
