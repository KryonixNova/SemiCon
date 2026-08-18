# KryonixNova

Drift-Sense hackathon submission: AI-powered navigation-error recovery for
wafer inspection tools (SEMI x IESA Hackathon 2026, Applied Materials
problem statement — see `docs/problem-statement.pdf`).

This repo consolidates two subsystems into one submission:

1. **`afb_generator/`** — AFB's procedural DRAM SEM image generator
   (GDS-style layout + rasterization + imaging-noise pipeline).
2. **`puthere/`** — puthere's own SEM image generator, plus its
   deep-learned reference-localization model (training, inference, and
   evaluation scripts).

`afb_generator/` and `puthere/` are independent, self-contained Python
projects, each with its own `requirements.txt`. The top-level
`requirements.txt` is their union, for a single `pip install -r
requirements.txt` covering everything.

`requirements.txt` pins `torch`/`torchvision` to CUDA 13.0 builds
(`+cu130`, via the PyTorch wheel index in the `--extra-index-url` line at
the top of the file). **For CPU-only or a different CUDA version**, drop
that `--extra-index-url` line and the `+cu130` suffix from the two lines
and let plain PyPI resolve them instead — nothing in `puthere/`'s inference
path requires CUDA specifically; `puthere/localize.py`/`predict.py` run
fine on CPU (training does not, practically speaking).

## Running the AFB generator

```bash
cd afb_generator
pip install -r requirements.txt
python generate_dataset.py --num-samples 20 --split train --output-dir ./output --seed 42
```

See `afb_generator/README.md` for generator-specific detail (presets,
multi-region dies, true zoom, single-training-defect mode) and how this
copy differs from the original pipeline described in the `docs/afb-*.pdf`
guides.

## Running puthere's generator + localizer

```bash
cd puthere
pip install -r requirements.txt
python generate_dataset.py --num-samples 20 --split test --output-dir ./output --seed 200000
python localize.py --manifest ./output/manifest.csv --output ./predictions.csv
```

(`--split test` requires a seed in `[200000, 200500)` — `production_v3` was
trained on canvas seeds `[0, 100000)`, so this uses the held-out test split
rather than training data.)

Two checkpoints ship in `puthere/checkpoints/`: **`production_v3`** is this
submission's default (consistently accurate across every noise/geometry
condition tested, pooled mean error 2.47px) and **`production_v5`** is kept
for reference only — a real-data specialist fine-tuned on real AFB-rendered
imagery, with a documented accuracy/robustness trade-off against
`production_v3`. See `puthere/README.md` for full detail on everything
above — architecture, the full `production_v3`/`production_v5` results
table and why v3 is the default, training lineage, evaluation methodology,
and reproduction steps.

See `puthere/scripts/train.py` and `puthere/scripts/validation_report.py`
for training and spec-compliance validation respectively.

## Running the tests

Each subsystem has its own pytest suite, run independently:

```bash
cd afb_generator && python -m pytest tests/ -v      # 44 tests
cd puthere && python -m pytest tests/ -v -m "not slow"   # 121 passed, 15 deselected
```

`puthere/tests/localizer/test_model.py::test_predictions_lie_inside_the_valid_range`
is a known pre-existing flaky test (unseeded random weight init in the test
itself, not a regression) — it fails intermittently under a full-suite run
but passes when re-run in isolation; this reproduces identically in the
original, untouched `puthere` development repository, so it's documented
here rather than treated as a bug to fix.

## Deliverables

- `solution_presentation.pptx` / `.pdf` — the mandatory presentation
  (Component 1 of the problem statement).
- `results/` — a sample generated dataset, predictions, and a
  spec-validation report against `puthere/checkpoints/production_v3`
  (Component 2's expected deliverables).
- `docs/` — the official problem statement plus both source projects'
  technical guides. (Note: AFB's guides describe the full original
  pipeline including its SuperPoint+LightGlue matcher, which is not
  included in `afb_generator/` — see Design notes below.)

## Design notes

- AFB's own SuperPoint+LightGlue matcher and RANSAC localizer are
  intentionally not included — puthere's deep-learned model is the one
  localization approach in this submission, so `afb_generator/` only
  carries generation code and has no torch/lightglue dependency.
- `puthere/`'s generator and localizer ship together rather than split
  into separate folders: puthere's training-time data loader
  (`src/localizer/data.py`) generates samples on the fly by calling
  directly into the same `src/patterns/`/`src/presets.py` code its
  standalone `generate_dataset.py` CLI uses — splitting them would just
  duplicate those files.

## Citations

Public sources backing this project's synthetic-data and noise-modeling
design choices, per the hackathon spec's requirement to justify structures
and augmentations against credible sources:

### DRAM 1T-1C cell structure (word lines, bit lines, capacitor storage)

- imec, "DRAM peripheral transistors technology platform."
  <https://www.imec-int.com/en/articles/technology-platform-thermally-stable-dram-peripheral-transistors>
- SemiAnalysis, "The Memory Wall: Past, Present, and Future of DRAM."
  <https://newsletter.semianalysis.com/p/the-memory-wall>

### SEM imaging noise and degradation modeling

- "Correction of Scanning Electron Microscope Imaging Artifacts in a Novel
  Digital Image Correlation Framework," *Experimental Mechanics* (Springer).
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6541586/>
- "Scanning Electron Microscope Image Signal-to-Noise Ratio Monitoring for
  Micro-Nanomanipulation." <https://hal.science/hal-01051309/document>

### Data augmentation for scale/rotation robustness in matching tasks

- "An Efficient Deep Template Matching and In-Plane Pose Estimation Method
  via Template-Aware Dynamic Convolution," arXiv.
  <https://arxiv.org/html/2510.01678>
- "Who Handles Orientation? Investigating Invariance in Feature Matching,"
  arXiv. <https://arxiv.org/html/2604.11809v1>

Full citation-to-code mapping: see `puthere`'s own development history
(`references/CITATIONS.md` in the original `puthere` repo) for which
specific parameter/function each source backs.
