# Drift-Sense hackathon-compliance design

Date: 2026-08-10
Status: approved, pending implementation plan

## Context

`puthere/` is a standalone deliverable package built around the Drift-Sense
localizer (Siamese ResNet-18 encoder -> dense cross-correlation -> dilated-conv
context head -> heatmap+offset decode). The SEMICON India 2026 / Applied
Materials "Drift-Sense" hackathon problem statement (`~/Desktop/problem
statement.pdf`) defines the actual target spec this package is being built
against. Comparing the current `puthere/` state to that spec surfaced four
gaps, scoped into four sub-projects below. The PPTX deliverable and the
literal `submission/` folder rename are explicitly out of scope for this
pass (deferred by the user).

Already satisfied by the existing build, not revisited here: 1000x1000
grayscale search images, DRAM synthetic structures, top-left coordinate
convention, GPU-trained checkpoint, single-pair `predict.py`, canvas-disjoint
train/val/test splits with per-pair seed reproducibility.

## Sub-project 1: Rotation + scale robustness

**Problem:** the spec requires the solution to handle a nominal 10:1
reference:search scale with robustness testing at ~9:1-11:1, plus ~1-2
degrees of rotation between the two captures. Nothing in the current
pipeline (`src/sem_imaging.py`, `src/localizer/data.py`,
`src/localizer/geometry.py`) applies any rotation, and the reference/search
scale relationship is a hardcoded constant (`SCALE = 10` in `geometry.py`,
`SCALE_FACTOR = 10` in `pipeline.py`).

**Approach (training-time augmentation, chosen over inference-time
multi-hypothesis search):** teach the model robustness via a new
augmentation profile, keeping `predict.py` inference simple and fast since
runtime is graded. This mirrors the existing `JITTER_PROFILES` /
`IMAGING_NOISE_PROFILES` pattern already in `data.py`.

### `GEOMETRIC_PROFILES` (new, in `src/localizer/data.py`)

```python
GEOMETRIC_PROFILES = {
    "normal": {"scale_ratio": (1.0, 1.0), "rotation_deg": (0.0, 0.0)},
    "drift":  {"scale_ratio": (0.9, 1.1), "rotation_deg": (-2.0, 2.0)},
}
```

`"normal"` is a no-op (backward compatible, matches current behavior
exactly). `"drift"` covers the spec's stated ranges: scale_ratio 0.9-1.1
corresponds to an effective 9:1-11:1 relationship relative to the nominal
10:1, and rotation_deg covers "about 1-2 degrees" symmetrically.

### `sample_pair()` changes

After the existing area-average downsample produces `ref_ds` (the
`REF_DS_PX` x `REF_DS_PX`, i.e. 100x100, reference the model actually
consumes), apply two new perturbations, in this order:

1. **Scale jitter.** Draw `scale_ratio ~ U(*range)`. Resize `ref_ds` down to
   `round(REF_DS_PX / scale_ratio)` then back up to `REF_DS_PX` via
   `cv2.resize`. This simulates the exact failure mode a real off-nominal
   capture produces: `predict.py`'s `load_standardized` force-resizes
   *any* input reference to `REF_DS_PX` regardless of its native
   resolution, so a real reference captured at 11x instead of 10x ends up
   with its content subtly the wrong apparent size inside that fixed
   100x100 canvas -- training on the same distortion closes that gap.
2. **Rotation jitter.** Draw `rotation_deg ~ U(*range)`. Apply via
   `cv2.warpAffine(ref_ds, cv2.getRotationMatrix2D(center, rotation_deg,
   1.0), ..., borderMode=cv2.BORDER_REFLECT)`.

Ground truth (`gt_x`, `gt_y`) is unchanged by either step -- only the
reference image's content is perturbed; the search image and its true
center are untouched. This is the same philosophy as the existing
position-jitter profiles: perturb the input, keep the label fixed, force
the network to learn invariance.

`sample_pair()` gains a `geometric_profile: str = "normal"` parameter.
`LocalizerDataset.__init__` and `_raw_items` thread it through exactly like
`imaging_noise_profile` does today.

### `train.py` changes

- New `--geometric-profile {normal,drift}` CLI flag, default `normal`
  (fully backward compatible -- omitting it reproduces current behavior
  exactly).
- Checkpoint dicts (`best.pt`/`last.pt`) gain a `"geometric_profile"` key,
  same pattern as the `"imaging_noise_profile"` key added earlier this
  session.
- The existing profile-change/`best_acc`-reset guard (added to handle a
  resumed run's noise profile changing) is extended to also compare
  `geometric_profile` on resume, resetting `best_acc = -1.0` if either
  profile differs from what the checkpoint was trained under.

### New `--init-from` warm-start flag

`production_v2`'s `CosineAnnealingLR` schedule has already fully decayed at
its step-40000 ceiling (`T_max=40000`, and `--resume` restores the
scheduler's state including that `T_max`). Simply `--resume`-ing further
would train at a near-zero learning rate the whole time -- not useful for
absorbing a new augmentation profile.

Add `--init-from <checkpoint-path>`, distinct from `--resume`: loads
*only* the model weights and `align_offset` from the given checkpoint
(which may belong to a different run), then starts fresh at `step=0`,
`best_acc=-1.0`, with a new optimizer and scheduler built against this
run's own `--max-steps`. `--resume` and `--init-from` are mutually
exclusive (error if both given). This is a warm-start / fine-tune-into-a-
new-run mechanism, distinct from `--resume`'s full-state same-run
continuation.

### Training run

Once implemented, fine-tune a new run from `checkpoints/production_v2/best.pt`
(currently: step 40000, best acc@50px=0.981, trained under
`imaging_noise_profile=harsh`) via `--init-from` under
`--imaging-noise-profile harsh --geometric-profile drift`, for a step
budget to be decided at execution time (matching this session's existing
practice of running in resumable chunks and observing validation curves
rather than pre-committing to an exact number).

## Sub-project 2: Submission-shaped scripts

### `generate_dataset.py` (new, top-level)

Wraps the same generator the training pipeline already uses
(`generate_canvas_bundle` / `sample_pair` in `src/localizer/data.py`) --
not the older, slightly-diverged `generate_mxn_dram_dataset.py`, which is
left untouched since it's still referenced elsewhere and working. Writes
persisted reference/search PNG pairs plus `manifest.csv`.

Manifest columns: `id, reference_path, search_path, gt_x, gt_y,
canvas_seed, crop_index, m, n, jitter_profile, imaging_noise_profile,
geometric_profile, scale_ratio, rotation_deg`.

CLI flags: `--num-samples`, `--seed`, `--output-dir`, `--split`
(`train`/`val`/`test`, reusing `split_seed_range` so generated data stays
canvas-disjoint from whatever the model was trained on), `--jitter-profile`,
`--imaging-noise-profile`, `--geometric-profile`.

### `localize.py` (new, top-level)

The spec's required "process a pair or evaluator-provided batch without
manual source-code changes" entry point.

Refactor: extract `scripts/predict.py`'s model-loading and single-pair
prediction logic into a new `src/localizer/inference.py`
(`load_model(checkpoint_path, device)`, `predict_pair(model, ref_path,
search_path, device)`), so both the existing single-pair CLI and the new
batch path share one implementation instead of duplicating it.
`scripts/predict.py` is updated to call into this module; its own CLI
contract and output format are unchanged.

`localize.py` supports two modes:
- `--checkpoint ckpt.pt --reference X --search Y` -- same single-pair
  contract as `predict.py` (kept for compatibility with existing docs/
  scripts referencing it).
- `--checkpoint ckpt.pt --manifest manifest.csv --output predictions.csv`
  -- batch mode: reads `reference_path`/`search_path` columns from any
  manifest (including one an evaluator supplies), predicts every row,
  writes `predictions.csv` with `id, predicted_x, predicted_y, confidence,
  runtime_ms`.

## Sub-project 3: Validation report

New `scripts/validation_report.py`, reusing `src/localizer/metrics.py`
rather than duplicating its accuracy/error math.

Runs a configurable number of pairs (default higher than the spec's
minimum of 30) across a matrix of `{imaging_noise_profile: normal, harsh}
x {geometric_profile: normal, drift}`, so results are reported "across
multiple noise levels ... scales and rotations" as the spec requires --
each combination gets its own row plus a pooled summary.

Per-condition and pooled metrics:
- Euclidean error: mean, median, worst-case.
- Pass rate @ 5px / 4px / 2px / 1px -- the spec's exact thresholds (not
  this codebase's own internal 5/10/50px convention used elsewhere).
- Sub-pixel detail (median/p90 error reported to hundredths of a pixel,
  since the offset head is a continuous regression, not a discrete grid
  classifier).
- Runtime per pair via `time.perf_counter()`, with device
  (`torch.cuda.get_device_name()` or `"cpu"`), Python version (`sys.version`),
  and the timing method stated inline in the report output.

Failure case: the single worst-error sample from the run gets a rendered
side-by-side overlay PNG (search image with true-center and
predicted-center markers, via matplotlib) plus a short root-cause note
(e.g. confidence value correlated with the miss, referencing whichever
condition produced it).

Outputs: `results/validation_report.json` (machine-readable),
`results/validation_report.md` (formatted per-condition tables),
`results/failure_case.png`.

## Sub-project 4: Docs & packaging

- Extend (not rewrite) `README.md`: exact commands for
  `generate_dataset.py`, `localize.py`, and `validation_report.py`; a
  coordinate-convention/assumptions section matching the spec's checklist
  wording; a short section documenting the `geometric_profile` /
  `imaging_noise_profile` robustness knobs.
- Confirm `requirements.txt` is a complete, dry-run-clean dependency list
  for a fresh environment (spec checklist item) -- no content changes
  expected, just verification.
- New `references/CITATIONS.md`: 2-3 real, independently verifiable public
  sources backing (a) DRAM 1T-1C cell structure, (b) SEM imaging
  noise/degradation modeling, (c) data augmentation for scale/rotation
  robustness in matching tasks. Sourced via web search, not invented --
  fabricated citations would actively hurt the submission more than having
  none.

Explicitly not done in this pass: renaming/restructuring the repo into the
PDF's literal `submission/` folder tree (packaging cosmetics, revisit only
if requested before final submission), and the PPTX deck (separate
deliverable, deferred by the user until real results exist to pull numbers
from).

## Testing

- Unit tests for `GEOMETRIC_PROFILES` application in `sample_pair()`
  (scale-ratio resize round-trips to the right shape; rotation preserves
  shape; `"normal"` profile is a byte-identical no-op vs. current
  behavior).
- Unit test for `--init-from` (loads weights, does not restore
  optimizer/scheduler/step; mutually exclusive with `--resume`).
- Unit tests for `src/localizer/inference.py`'s extracted functions.
- Existing test suite (currently 96/96 passing) must stay green throughout.
- End-to-end smoke test of `generate_dataset.py` -> `localize.py` batch
  mode -> `validation_report.py`, on a small sample count, before the real
  training run and full-scale report.
