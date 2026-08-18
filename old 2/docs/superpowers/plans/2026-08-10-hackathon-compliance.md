# Hackathon Compliance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap between `puthere/`'s current build and the Applied Materials "Drift-Sense" hackathon problem statement (`~/Desktop/problem statement.pdf`): rotation/scale robustness, submission-shaped generator/localizer scripts, a spec-compliant validation report, and doc/citation packaging.

**Architecture:** Extend the existing on-the-fly synthetic generator (`src/localizer/data.py`) with a new augmentation profile trained into the model, not a new inference-time search path. Extract the model-loading/single-pair-predict logic already in `scripts/predict.py` into a shared `src/localizer/inference.py` module so the new batch-capable `localize.py` and the existing single-pair CLI never diverge. Add two new top-level scripts (`generate_dataset.py`, `localize.py`) matching the spec's recommended submission layout, plus `scripts/validation_report.py` for the spec's required threshold/runtime/failure-case reporting.

**Tech Stack:** Python, PyTorch, OpenCV (`cv2`), NumPy, matplotlib, pytest. No new dependencies — everything used below is already in `requirements.txt`.

## Global Constraints

- Reference images are `1000x1000` native, downsampled to `100x100` (`REF_DS_PX`) for the model; search images are `1000x1000` (`SEARCH_PX`) — exact values from `src/localizer/geometry.py`. Do not change these.
- Nominal reference:search scale is `10:1`; robustness testing may use `9:1`-`11:1` (relative ratio `0.9`-`1.1` of nominal).
- Rotation robustness target: approximately 1-2 degrees between reference and search.
- Coordinate convention: origin `(0, 0)` top-left, `x` increases right, `y` increases down (already correct throughout the codebase — verify, don't change).
- All new CLI flags default to values that reproduce **current** behavior exactly (`--geometric-profile normal`, etc.) — nothing changes for existing callers who don't opt in.
- Existing test suite (96 passed, 2 deselected via `-m "not slow"`) must stay green after every task. Run `python -m pytest tests/ -q -m "not slow"` from `puthere/` after each task's changes, and the `slow`-marked tests for any task that adds one.
- No fabricated citations — Task 11's sources are real, verifiable URLs, already checked during planning.

---

### Task 1: `GEOMETRIC_PROFILES` and scale/rotation jitter in `sample_pair()`

**Files:**
- Modify: `src/localizer/data.py`
- Test: `tests/localizer/test_data.py`

**Interfaces:**
- Produces: `GEOMETRIC_PROFILES: dict[str, dict]` (module-level, keys `"normal"`/`"drift"`, each `{"scale_ratio": (lo, hi), "rotation_deg": (lo, hi)}`).
- Produces: `sample_pair(bundle, canvas_seed, crop_index, imaging_noise_profile="normal", geometric_profile="normal") -> dict` — return dict gains three new keys beyond the current five: `"reference_img_u8"` (`np.uint8`, shape `(100, 100)`, the geometry-jittered reference *before* standardization — needed by Task 6's PNG writer), `"scale_ratio"` (float), `"rotation_deg"` (float).

- [ ] **Step 1: Write the failing tests**

Add to `tests/localizer/test_data.py` (near the other `sample_pair`/`generate_canvas_bundle` tests):

```python
def test_geometric_profiles_normal_is_a_noop():
    a = sample_pair(generate_canvas_bundle(5), 5, 0, geometric_profile="normal")
    b = sample_pair(generate_canvas_bundle(5), 5, 0)  # default is also "normal"
    assert torch.equal(a["reference_img"], b["reference_img"])
    assert a["scale_ratio"] == 1.0
    assert a["rotation_deg"] == 0.0


def test_drift_profile_perturbs_reference_but_not_ground_truth():
    normal = sample_pair(generate_canvas_bundle(5), 5, 0, geometric_profile="normal")
    drift = sample_pair(generate_canvas_bundle(5), 5, 0, geometric_profile="drift")
    assert not torch.equal(normal["reference_img"], drift["reference_img"])
    assert normal["gt_x"] == drift["gt_x"]
    assert normal["gt_y"] == drift["gt_y"]
    assert normal["reference_img"].shape == drift["reference_img"].shape


def test_drift_profile_scale_and_rotation_stay_within_declared_range():
    scales, rotations = [], []
    for k in range(20):
        s = sample_pair(generate_canvas_bundle(5), 5, k, geometric_profile="drift")
        scales.append(s["scale_ratio"])
        rotations.append(s["rotation_deg"])
    assert all(0.9 <= v <= 1.1 for v in scales)
    assert all(-2.0 <= v <= 2.0 for v in rotations)
    assert len(set(scales)) > 1, "scale_ratio should vary across crops, not be constant"


def test_sample_pair_includes_raw_uint8_reference():
    s = sample_pair(generate_canvas_bundle(0), 0, 0)
    assert s["reference_img_u8"].dtype == np.uint8
    assert s["reference_img_u8"].shape == (100, 100)
```

Add `import numpy as np` to the top of the test file if not already present (it is — the file already imports `numpy as np` for other tests).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd puthere && python -m pytest tests/localizer/test_data.py -k geometric_profiles_normal -v`
Expected: FAIL with `TypeError: sample_pair() got an unexpected keyword argument 'geometric_profile'` (or `KeyError: 'reference_img_u8'` / `KeyError: 'scale_ratio'` once the keyword issue is worked around) — confirms the feature doesn't exist yet.

- [ ] **Step 3: Implement**

In `src/localizer/data.py`, add `import cv2` to the imports at the top of the file (alongside the existing `numpy as np` / `torch` imports).

Add the new profile dict right after `IMAGING_NOISE_PROFILES` (before `def split_seed_range`):

```python
# Robustness to the reference/search scale relationship drifting off its
# nominal 10:1 (tested 9:1-11:1) and to a few degrees of rotation between
# the two captures. "normal" is a no-op (current behavior, unchanged);
# "drift" is named for this project's own branding -- stage/calibration
# drift is exactly the real-world cause of both effects.
GEOMETRIC_PROFILES = {
    "normal": {"scale_ratio": (1.0, 1.0), "rotation_deg": (0.0, 0.0)},
    "drift":  {"scale_ratio": (0.9, 1.1), "rotation_deg": (-2.0, 2.0)},
}


def _apply_geometric_jitter(ref_ds: np.ndarray, scale_ratio: float,
                             rotation_deg: float) -> np.ndarray:
    """Perturb an already-downsampled REF_DS_PX x REF_DS_PX reference to
    simulate a real capture at scale_ratio*nominal_scale and rotation_deg
    off nominal. Applied to the reference only -- the search image and the
    crop's true center are untouched, so ground truth never changes; this
    is the same philosophy as the position-jitter profiles above (perturb
    the input, keep the label fixed, force the network to learn
    invariance). scale_ratio=1.0/rotation_deg=0.0 is a byte-identical
    no-op, matching "normal"'s current behavior exactly.
    """
    if scale_ratio == 1.0 and rotation_deg == 0.0:
        return ref_ds
    h, w = ref_ds.shape
    out = ref_ds
    if scale_ratio != 1.0:
        inter_size = max(1, int(round(w / scale_ratio)))
        small = cv2.resize(out, (inter_size, inter_size), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rotation_deg != 0.0:
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        out = cv2.warpAffine(out, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return out
```

Modify `sample_pair`'s signature and body (the area-average downsample and return statement):

```python
def sample_pair(bundle: dict, canvas_seed: int, crop_index: int,
                 imaging_noise_profile: str = "normal",
                 geometric_profile: str = "normal") -> dict:
    """Cut one Reference crop from the canvas and image it. The Search image
    is reused from the bundle unchanged."""
    noise = IMAGING_NOISE_PROFILES[imaging_noise_profile]["reference"]
    rng = np.random.default_rng(canvas_seed * 1_000_003 + crop_index)
    max_off = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    x0 = int(rng.integers(0, max_off + 1))
    y0 = int(rng.integers(0, max_off + 1))

    crop = bundle["fine_canvas"][y0:y0 + REFERENCE_SIZE_PX,
                                 x0:x0 + REFERENCE_SIZE_PX]
    ref_full = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=float(rng.uniform(*noise["spot_size_nm"])),
        dose=float(rng.uniform(*noise["dose"])),
        rng=rng,
        detector_noise_sigma=float(rng.uniform(*noise["detector_noise_sigma"])),
        drift_jitter_px=float(rng.uniform(*noise["drift_jitter_px"])),
        astigmatism_ratio=float(rng.uniform(*noise["astigmatism_ratio"])),
        vignette_strength=float(rng.uniform(*noise["vignette_strength"])),
        barrel_distortion_k=float(rng.uniform(*noise["barrel_distortion_k"])),
        charging_streak_prob=float(rng.uniform(*noise["charging_streak_prob"])),
        charging_streak_intensity=float(rng.uniform(*noise["charging_streak_intensity"])),
        speckle_sigma=float(rng.uniform(*noise["speckle_sigma"])),
        salt_pepper_prob=float(rng.uniform(*noise["salt_pepper_prob"])),
    )
    # Area-average to the search scale, matching how image_search downsamples.
    ref_ds = sem_imaging.downsample_area_average(ref_full, SCALE_FACTOR)

    geom = GEOMETRIC_PROFILES[geometric_profile]
    scale_ratio = float(rng.uniform(*geom["scale_ratio"]))
    rotation_deg = float(rng.uniform(*geom["rotation_deg"]))
    ref_ds = _apply_geometric_jitter(ref_ds, scale_ratio, rotation_deg)

    half = REFERENCE_SIZE_PX / SCALE_FACTOR / 2.0        # 50 px
    return {
        "reference_img": _standardize(ref_ds),
        "reference_img_u8": ref_ds,
        "search_img": _standardize(bundle["search_img"]),
        "gt_x": x0 / SCALE_FACTOR + half,
        "gt_y": y0 / SCALE_FACTOR + half,
        "canvas_seed": canvas_seed,
        "scale_ratio": scale_ratio,
        "rotation_deg": rotation_deg,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd puthere && python -m pytest tests/localizer/test_data.py -v`
Expected: PASS, including all pre-existing tests in the file (they only check a subset of keys/behavior, so the new keys don't break them — confirm `test_dataset_yields_batchable_items` and friends still pass too since `LocalizerDataset._raw_items` builds its own narrower dict and isn't affected by `sample_pair` gaining extra keys).

- [ ] **Step 5: Run the full fast test suite**

Run: `cd puthere && python -m pytest tests/ -q -m "not slow"`
Expected: `100 passed, 2 deselected` (96 previous + 4 new).

- [ ] **Step 6: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "feat(data): add GEOMETRIC_PROFILES for scale/rotation robustness augmentation" 2>&1 || echo "no independent git repo here -- see puthere/docs/superpowers/specs/2026-08-10-hackathon-compliance-design.md for why; skip commit, changes stay on disk"
```

(`puthere/` has no git repo of its own — see the design doc's context note. If this errors or commits into an unrelated repo, do not force it; just leave the change on disk and move to the next task.)

---

### Task 2: Thread `geometric_profile` through `LocalizerDataset`

**Files:**
- Modify: `src/localizer/data.py`
- Test: `tests/localizer/test_data.py`

**Interfaces:**
- Consumes: `sample_pair(..., geometric_profile=...)` from Task 1.
- Produces: `LocalizerDataset.__init__(self, split, config, jitter_profile="normal", shuffle_buffer_size=256, seed_offset=0, imaging_noise_profile="normal", geometric_profile="normal")`.

- [ ] **Step 1: Write the failing test**

Add to `tests/localizer/test_data.py`:

```python
def test_dataset_geometric_profile_changes_reference_content():
    cfg = LocalizerConfig(crops_per_canvas=1, val_seed_lo=0, val_seed_hi=1)
    normal_item = next(iter(LocalizerDataset(
        "val", cfg, geometric_profile="normal", shuffle_buffer_size=1)))
    drift_item = next(iter(LocalizerDataset(
        "val", cfg, geometric_profile="drift", shuffle_buffer_size=1)))
    assert not torch.equal(normal_item["reference_img"], drift_item["reference_img"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd puthere && python -m pytest tests/localizer/test_data.py -k dataset_geometric_profile -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'geometric_profile'`.

- [ ] **Step 3: Implement**

In `src/localizer/data.py`, modify `LocalizerDataset.__init__` and `_raw_items`:

```python
    def __init__(self, split: str, config: LocalizerConfig,
                 jitter_profile: str = "normal", shuffle_buffer_size: int = 256,
                 seed_offset: int = 0, imaging_noise_profile: str = "normal",
                 geometric_profile: str = "normal"):
        self.split = split
        self.config = config
        self.jitter_profile = jitter_profile
        self.imaging_noise_profile = imaging_noise_profile
        self.geometric_profile = geometric_profile
        self.shuffle_buffer_size = shuffle_buffer_size
        self.lo, self.hi = split_seed_range(split, config)
        self.seed_offset = seed_offset

    def _raw_items(self, wid: int, nw: int):
        span = self.hi - self.lo
        start = self.lo + (self.seed_offset % span if span > 0 else 0)
        for seed in range(start + wid, self.hi, nw):
            bundle = generate_canvas_bundle(seed, self.jitter_profile,
                                            self.imaging_noise_profile)
            for k in range(self.config.crops_per_canvas):
                s = sample_pair(bundle, seed, k, self.imaging_noise_profile,
                                self.geometric_profile)
                yield {
                    "reference_img": s["reference_img"].unsqueeze(0),
                    "search_img": s["search_img"].unsqueeze(0),
                    "gt_x": torch.tensor(s["gt_x"], dtype=torch.float32),
                    "gt_y": torch.tensor(s["gt_y"], dtype=torch.float32),
                    "canvas_seed": torch.tensor(seed, dtype=torch.long),
                }
```

(Only the `__init__` signature/body and the `sample_pair(...)` call inside `_raw_items` change; the rest of the class is untouched. Keep the existing docstring comment above `self.seed_offset = seed_offset` as-is.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd puthere && python -m pytest tests/localizer/test_data.py -v`
Expected: PASS, all tests including the new one.

- [ ] **Step 5: Run the full fast test suite**

Run: `cd puthere && python -m pytest tests/ -q -m "not slow"`
Expected: `101 passed, 2 deselected`.

- [ ] **Step 6: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "feat(data): thread geometric_profile through LocalizerDataset" 2>&1 || true
```

---

### Task 3: `--geometric-profile` CLI flag + checkpoint metadata in `train.py`

**Files:**
- Modify: `scripts/train.py`
- Test: `tests/localizer/test_train_integration.py`

**Interfaces:**
- Consumes: `LocalizerDataset(..., geometric_profile=...)` from Task 2.
- Produces: checkpoints (`best.pt`/`last.pt`) gain a `"geometric_profile"` key; the resume path resets `best_acc` if either `imaging_noise_profile` or `geometric_profile` changed since the checkpoint was last trained.

- [ ] **Step 1: Write the failing test**

Add to `tests/localizer/test_train_integration.py` (needs `import subprocess`, `import sys as _sys`, and `from pathlib import Path` at the top if not already present — check the existing imports first and add only what's missing):

```python
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_train(tmp_path, extra_args):
    cmd = [_sys.executable, "scripts/train.py", "--run-name", "geo_test",
           "--out-dir", str(tmp_path), "--max-steps", "4", "--steps-this-run", "2",
           "--val-every", "2", "--val-batches", "1", "--num-workers", "0"] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return result


@pytest.mark.slow
def test_geometric_profile_persisted_and_resets_best_acc_on_change(tmp_path):
    import torch as _torch

    r1 = _run_train(tmp_path, ["--geometric-profile", "normal"])
    assert r1.returncode == 0, r1.stderr
    ckpt1 = _torch.load(tmp_path / "geo_test" / "last.pt", weights_only=False)
    assert ckpt1["geometric_profile"] == "normal"

    r2 = _run_train(tmp_path, ["--geometric-profile", "drift", "--resume"])
    assert r2.returncode == 0, r2.stderr
    assert "NOTE: profile changed" in r2.stdout
    ckpt2 = _torch.load(tmp_path / "geo_test" / "last.pt", weights_only=False)
    assert ckpt2["geometric_profile"] == "drift"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd puthere && python -m pytest tests/localizer/test_train_integration.py -k geometric_profile_persisted -v -m slow`
Expected: FAIL — `r1.returncode == 0` assertion fails because `--geometric-profile` isn't a recognized flag yet (`argparse` error, nonzero exit).

- [ ] **Step 3: Implement**

In `scripts/train.py`, add the flag in `parse_args()` right after the existing `--imaging-noise-profile` block:

```python
    p.add_argument("--geometric-profile", default="normal",
                   choices=["normal", "drift"],
                   help="'drift' adds ~1-2 degree rotation and 9:1-11:1 "
                        "scale-ratio jitter to reference crops during "
                        "training, for robustness to stage-drift/"
                        "calibration variation between the reference and "
                        "search captures (see src/localizer/data.py's "
                        "GEOMETRIC_PROFILES).")
```

In `main()`, update the config print line:

```python
    print(f"config: batch_size={cfg.batch_size} lr={cfg.lr} "
          f"hard_negative_radius_cells={cfg.hard_negative_radius_cells} "
          f"lambda_hard_negative={cfg.lambda_hard_negative} "
          f"imaging_noise_profile={args.imaging_noise_profile} "
          f"geometric_profile={args.geometric_profile}")
```

Update both `LocalizerDataset(...)` constructions (`val_loader` and `train_loader`) to pass `geometric_profile=args.geometric_profile`:

```python
    val_loader = DataLoader(
        LocalizerDataset("val", cfg, args.jitter_profile,
                         imaging_noise_profile=args.imaging_noise_profile,
                         geometric_profile=args.geometric_profile),
        batch_size=cfg.batch_size, num_workers=2)
```

```python
    train_loader = DataLoader(
        LocalizerDataset("train", cfg, args.jitter_profile,
                         seed_offset=train_seed_offset,
                         imaging_noise_profile=args.imaging_noise_profile,
                         geometric_profile=args.geometric_profile),
        batch_size=cfg.batch_size, num_workers=args.num_workers, pin_memory=True)
```

Update the `--resume` branch's profile check:

```python
        prev_profile = ckpt.get("imaging_noise_profile", "normal")
        prev_geom = ckpt.get("geometric_profile", "normal")
        print(f"resumed {run_dir} at step {step}/{args.max_steps} "
              f"(best acc@50px so far: {best_acc:.3f}, trained under "
              f"imaging_noise_profile={prev_profile}, "
              f"geometric_profile={prev_geom})")
        if prev_profile != args.imaging_noise_profile or prev_geom != args.geometric_profile:
            print(f"  NOTE: profile changed (imaging_noise_profile {prev_profile} -> "
                  f"{args.imaging_noise_profile}, geometric_profile {prev_geom} -> "
                  f"{args.geometric_profile}) -- validation difficulty just changed, "
                  f"so the old best_acc={best_acc:.3f} isn't a fair bar for the new "
                  f"profile(s). Resetting best-so-far to -1.0 so checkpoints save "
                  f"normally under the new conditions.")
            best_acc = -1.0
```

Add `"geometric_profile": args.geometric_profile` to both `torch.save(...)` dicts (the `best_path` save and the `last_path` save):

```python
                torch.save({"model": model.state_dict(), "config": cfg.as_dict(),
                            "align_offset": align, "step": step, "metrics": m,
                            "imaging_noise_profile": args.imaging_noise_profile,
                            "geometric_profile": args.geometric_profile},
                           best_path)
```

```python
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "config": cfg.as_dict(), "align_offset": align,
                    "step": step, "best_acc": best_acc, "metrics": m,
                    "imaging_noise_profile": args.imaging_noise_profile,
                    "geometric_profile": args.geometric_profile},
                   last_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd puthere && python -m pytest tests/localizer/test_train_integration.py -k geometric_profile_persisted -v -m slow`
Expected: PASS. This actually runs two tiny live training invocations (4 steps each), so allow up to ~60s.

- [ ] **Step 5: Run the full fast test suite**

Run: `cd puthere && python -m pytest tests/ -q -m "not slow"`
Expected: `101 passed, 2 deselected` (unchanged — the new test is `slow`-marked, so it's deselected here; that's expected).

- [ ] **Step 6: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "feat(train): add --geometric-profile flag, checkpoint metadata, and best_acc-reset guard" 2>&1 || true
```

---

### Task 4: `--init-from` warm-start flag in `train.py`

**Files:**
- Modify: `scripts/train.py`
- Test: `tests/localizer/test_train_integration.py`

**Interfaces:**
- Produces: `--init-from <checkpoint-path>` CLI flag, mutually exclusive with `--resume`. Loads model weights + `align_offset` only; starts `step=0`, `best_acc=-1.0`, fresh optimizer/scheduler.

- [ ] **Step 1: Write the failing test**

Add to `tests/localizer/test_train_integration.py` (builds its own two
distinctly-named runs directly, rather than reusing Task 3's `_run_train`
helper, since `_run_train` is hardcoded to the `geo_test` run-name):

```python
@pytest.mark.slow
def test_init_from_warm_starts_weights_but_resets_step_and_schedule(tmp_path):
    import torch as _torch

    src_cmd = [_sys.executable, "scripts/train.py", "--run-name", "init_src",
               "--out-dir", str(tmp_path), "--max-steps", "4", "--steps-this-run", "4",
               "--val-every", "2", "--val-batches", "1", "--num-workers", "0"]
    src_result = subprocess.run(src_cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert src_result.returncode == 0, src_result.stderr
    src_ckpt_path = tmp_path / "init_src" / "best.pt"
    assert src_ckpt_path.exists()

    dst_cmd = [_sys.executable, "scripts/train.py", "--run-name", "init_dst",
               "--out-dir", str(tmp_path), "--max-steps", "2", "--steps-this-run", "2",
               "--val-every", "2", "--val-batches", "1", "--num-workers", "0",
               "--init-from", str(src_ckpt_path)]
    dst_result = subprocess.run(dst_cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert dst_result.returncode == 0, dst_result.stderr
    assert "initialized weights from" in dst_result.stdout

    dst_ckpt = _torch.load(tmp_path / "init_dst" / "last.pt", weights_only=False)
    assert dst_ckpt["step"] == 2
    assert dst_ckpt["best_acc"] >= -1.0


def test_resume_and_init_from_together_is_rejected(tmp_path):
    cmd = [_sys.executable, "scripts/train.py", "--run-name", "conflict_test",
           "--out-dir", str(tmp_path), "--max-steps", "2", "--resume",
           "--init-from", "checkpoints/production_v2/best.pt"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "mutually exclusive" in (result.stdout + result.stderr)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd puthere && python -m pytest tests/localizer/test_train_integration.py -k init_from -v -m slow`
Expected: FAIL — `--init-from` not recognized (argparse error) on the first invocation that uses it; `test_resume_and_init_from_together_is_rejected` fails because there's no mutual-exclusivity check yet (it will currently fail for a *different* reason: `--init-from` unrecognized, still a nonzero exit, so check the error message assertion specifically fails since it won't contain "mutually exclusive" yet — argparse's own "unrecognized arguments" message will show instead).

- [ ] **Step 3: Implement**

In `scripts/train.py`, add the flag in `parse_args()` after `--resume`:

```python
    p.add_argument("--init-from", default=None,
                   help="warm-start model weights (and align_offset) from "
                        "another run's checkpoint, then start THIS run at "
                        "step=0 with a fresh optimizer/scheduler under this "
                        "invocation's own --max-steps/profiles. Unlike "
                        "--resume (which restores optimizer/scheduler/step "
                        "from this run's OWN last.pt), --init-from only "
                        "transplants weights -- for fine-tuning into a new "
                        "augmentation profile after a prior run's LR "
                        "schedule has already fully decayed. Mutually "
                        "exclusive with --resume.")
```

In `main()`, right before the existing `if args.resume:` block, add the exclusivity check:

```python
    if args.resume and args.init_from:
        raise SystemExit(
            "--resume and --init-from are mutually exclusive -- --resume "
            "continues THIS run's own last.pt (full optimizer/scheduler/"
            "step state); --init-from warm-starts fresh from a DIFFERENT "
            "checkpoint's weights only. Pick one.")
```

Change the `if args.resume: ... else:` into `if args.resume: ... elif args.init_from: ... else:`:

```python
    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"--resume given but no checkpoint at {last_path} -- "
                              f"run without --resume first to start this run-name fresh.")
        ckpt = torch.load(last_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.align_offset.fill_(ckpt["align_offset"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        align = ckpt["align_offset"]
        step = ckpt["step"]
        best_acc = ckpt["best_acc"]
        prev_profile = ckpt.get("imaging_noise_profile", "normal")
        prev_geom = ckpt.get("geometric_profile", "normal")
        print(f"resumed {run_dir} at step {step}/{args.max_steps} "
              f"(best acc@50px so far: {best_acc:.3f}, trained under "
              f"imaging_noise_profile={prev_profile}, "
              f"geometric_profile={prev_geom})")
        if prev_profile != args.imaging_noise_profile or prev_geom != args.geometric_profile:
            print(f"  NOTE: profile changed (imaging_noise_profile {prev_profile} -> "
                  f"{args.imaging_noise_profile}, geometric_profile {prev_geom} -> "
                  f"{args.geometric_profile}) -- validation difficulty just changed, "
                  f"so the old best_acc={best_acc:.3f} isn't a fair bar for the new "
                  f"profile(s). Resetting best-so-far to -1.0 so checkpoints save "
                  f"normally under the new conditions.")
            best_acc = -1.0
    elif args.init_from:
        if not os.path.exists(args.init_from):
            raise SystemExit(f"--init-from checkpoint not found: {args.init_from}")
        src_ckpt = torch.load(args.init_from, weights_only=False, map_location=device)
        model.load_state_dict(src_ckpt["model"])
        align = float(src_ckpt["align_offset"])
        model.align_offset.fill_(align)
        step, best_acc = 0, -1.0
        src_step = src_ckpt.get("step", "?")
        src_acc = src_ckpt.get("metrics", {}).get("acc@50px", float("nan"))
        print(f"initialized weights from {args.init_from} "
              f"(its step={src_step}, acc@50px={src_acc:.3f}); "
              f"starting this run fresh at step 0 with a new optimizer/scheduler")
    else:
        align = model.calibrate()
        print(f"calibrated align_offset = {align:.3f} px")
        step, best_acc = 0, -1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd puthere && python -m pytest tests/localizer/test_train_integration.py -k "init_from or resume_and_init_from" -v -m slow`
Expected: PASS (the mutual-exclusivity test is not `slow`-marked and runs instantly since it fails before any real training starts; the warm-start test is `slow`-marked and takes ~30-60s for two tiny training invocations).

- [ ] **Step 5: Run the full fast test suite**

Run: `cd puthere && python -m pytest tests/ -q -m "not slow"`
Expected: `102 passed, 2 deselected` (the mutual-exclusivity test isn't `slow`-marked, so it's included in this count).

- [ ] **Step 6: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "feat(train): add --init-from warm-start flag for fine-tuning into a new run" 2>&1 || true
```

---

### Task 5: Extract `src/localizer/inference.py` from `predict.py`

**Files:**
- Create: `src/localizer/inference.py`
- Modify: `scripts/predict.py`
- Create: `tests/conftest.py`
- Test: `tests/localizer/test_inference.py`

**Interfaces:**
- Produces: `load_model(checkpoint_path: str, device: str) -> DriftSenseLocalizer`.
- Produces: `predict_pair(model, reference_path: str, search_path: str, device: str) -> dict` with keys `"x"`, `"y"`, `"confidence"` (all `float`). Raises `ValueError` if the search image isn't `SEARCH_PX x SEARCH_PX`.
- Produces (test fixture, reused by Tasks 7 and 8): `tiny_checkpoint(tmp_path_factory) -> tuple[str, str]` returning `(checkpoint_path, device)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`:

```python
import pytest


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """A real (untrained, randomly-initialized) DriftSenseLocalizer
    checkpoint, session-scoped so building the encoder (a real ResNet-18)
    only happens once across the whole test run. Untrained weights are
    fine here -- these tests only check inference plumbing (shapes, CSV
    columns, error handling), never prediction accuracy."""
    import torch

    from src.localizer.config import LocalizerConfig
    from src.localizer.model import DriftSenseLocalizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DriftSenseLocalizer(LocalizerConfig()).to(device)
    align = model.calibrate()
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save({"model": model.state_dict(), "align_offset": align,
               "step": 0, "metrics": {"acc@50px": 0.0}}, path)
    return str(path), device
```

Create `tests/localizer/test_inference.py`:

```python
import numpy as np
import pytest


@pytest.mark.slow
def test_predict_pair_returns_xyz(tmp_path, tiny_checkpoint):
    import cv2

    from src.localizer.inference import load_model, predict_pair

    ckpt_path, device = tiny_checkpoint
    rng = np.random.default_rng(0)
    ref_path = tmp_path / "ref.png"
    search_path = tmp_path / "search.png"
    cv2.imwrite(str(ref_path), rng.integers(0, 255, (100, 100), dtype=np.uint8))
    cv2.imwrite(str(search_path), rng.integers(0, 255, (1000, 1000), dtype=np.uint8))

    model = load_model(ckpt_path, device)
    result = predict_pair(model, str(ref_path), str(search_path), device)

    assert set(result) == {"x", "y", "confidence"}
    assert isinstance(result["x"], float) and isinstance(result["y"], float)
    assert isinstance(result["confidence"], float)


@pytest.mark.slow
def test_predict_pair_rejects_wrong_search_size(tmp_path, tiny_checkpoint):
    import cv2

    from src.localizer.inference import load_model, predict_pair

    ckpt_path, device = tiny_checkpoint
    rng = np.random.default_rng(0)
    ref_path = tmp_path / "ref.png"
    bad_search_path = tmp_path / "bad_search.png"
    cv2.imwrite(str(ref_path), rng.integers(0, 255, (100, 100), dtype=np.uint8))
    cv2.imwrite(str(bad_search_path), rng.integers(0, 255, (500, 500), dtype=np.uint8))

    model = load_model(ckpt_path, device)
    with pytest.raises(ValueError, match="must be"):
        predict_pair(model, str(ref_path), str(bad_search_path), device)


def test_load_standardized_resizes_reference_to_target_px(tmp_path):
    import cv2

    from src.localizer.inference import load_standardized

    rng = np.random.default_rng(0)
    path = tmp_path / "big_ref.png"
    cv2.imwrite(str(path), rng.integers(0, 255, (1000, 1000), dtype=np.uint8))

    t = load_standardized(str(path), target_px=100)
    assert t.shape == (1, 1, 100, 100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd puthere && python -m pytest tests/localizer/test_inference.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.localizer.inference'`.

- [ ] **Step 3: Implement**

Create `src/localizer/inference.py`:

```python
"""Shared single-pair inference: checkpoint loading + prediction, used by
both scripts/predict.py's CLI and localize.py's batch mode so the two never
diverge in how a reference/search pair is turned into a prediction.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.geometry import REF_DS_PX, SEARCH_PX
from src.localizer.model import DriftSenseLocalizer


def load_standardized(path: str, target_px: int | None = None) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not read {path}")
    if target_px is not None and img.shape[0] != target_px:
        img = cv2.resize(img, (target_px, target_px), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(img)).float()
    t = (t - t.mean()) / t.std().clamp_min(1e-6)
    return t.unsqueeze(0).unsqueeze(0)


def load_model(checkpoint_path: str, device: str) -> DriftSenseLocalizer:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = DriftSenseLocalizer(LocalizerConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.align_offset.fill_(ckpt["align_offset"])
    model.eval()
    return model


def predict_pair(model: DriftSenseLocalizer, reference_path: str, search_path: str,
                  device: str) -> dict:
    """Returns {"x": float, "y": float, "confidence": float}.

    The reference is deliberately force-resized to REF_DS_PX regardless of
    its input size -- that's intentional (matches how the model was
    trained: reference crops are always downsampled to REF_DS_PX). It
    implicitly assumes the reference is already at the correct nm/px scale
    *before* this resize; a reference captured at a genuinely different
    physical scale will be silently misinterpreted (resized to the right
    pixel count without ever being rescaled to the right physical extent).

    Unlike the reference, the search image is NOT auto-resized: a wrong
    size means the physical scale is likely wrong too, so this raises
    ValueError rather than silently resizing.
    """
    ref = load_standardized(reference_path, REF_DS_PX).to(device)
    search = load_standardized(search_path).to(device)
    if search.shape[-2:] != (SEARCH_PX, SEARCH_PX):
        raise ValueError(
            f"search image must be {SEARCH_PX}x{SEARCH_PX} px, got "
            f"{search.shape[-2]}x{search.shape[-1]} px ({search_path}). "
            f"Unlike the reference, the search image is not auto-resized: "
            f"a wrong size means the physical scale is likely wrong too."
        )
    with torch.no_grad():
        out = model.predict(ref, search)
    return {"x": float(out["x"][0]), "y": float(out["y"][0]),
            "confidence": float(out["confidence"][0])}
```

Replace `scripts/predict.py`'s contents entirely with:

```python
#!/usr/bin/env python3
"""Predict the reference's centre in a search image.

Matches the contract of baseline_solution/infer.py: takes two image paths and
prints the predicted centre. Outputs exactly the three contract values --
predicted_x, predicted_y, confidence. localization_error is NOT emitted: it
requires ground truth and is an evaluation metric only.

Example:
    python scripts/predict.py --checkpoint checkpoints/m3_hn_r24/best.pt \
        --reference ref.png --search search.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.localizer.inference import load_model, predict_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--search", required=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    result = predict_pair(model, args.reference, args.search, device)

    if args.verbose:
        print(f"predicted_x={result['x']:.2f} predicted_y={result['y']:.2f} "
              f"confidence={result['confidence']:.4f}")
    else:
        print(f"{result['x']:.2f},{result['y']:.2f},{result['confidence']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd puthere && python -m pytest tests/localizer/test_inference.py -v -m slow`
Expected: PASS, all three tests.

- [ ] **Step 5: Verify `predict.py`'s CLI contract is unchanged**

Run (using the real shipped checkpoint and any existing test image pair from `test_examples/`, e.g.):
```bash
cd puthere && python scripts/predict.py --checkpoint checkpoints/m3_hn_r24/best.pt \
    --reference test_examples/ref_00.png --search test_examples/search_00.png --verbose
```
Expected: prints `predicted_x=... predicted_y=... confidence=...` exactly as before the refactor (same output format, same numeric values as any prior run recorded in `test_examples/predictions.csv` for sample 0, since the model/logic is unchanged — only its location moved).

- [ ] **Step 6: Run the full fast test suite**

Run: `cd puthere && python -m pytest tests/ -q -m "not slow"`
Expected: `103 passed, 2 deselected` (the `load_standardized` shape test isn't `slow`-marked).

- [ ] **Step 7: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "refactor(predict): extract load_model/predict_pair into src/localizer/inference.py" 2>&1 || true
```

---

### Task 6: `generate_dataset.py` (new, top-level)

**Files:**
- Create: `generate_dataset.py`
- Test: `tests/test_generate_dataset.py`

**Interfaces:**
- Consumes: `generate_canvas_bundle`, `sample_pair`, `split_seed_range` from `src/localizer/data.py` (Tasks 1-2 additions included).
- Produces: `manifest.csv` with columns `id, reference_path, search_path, gt_x, gt_y, canvas_seed, crop_index, jitter_profile, imaging_noise_profile, geometric_profile, scale_ratio, rotation_deg` — this exact column set is relied on by Task 7's batch-mode test.

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate_dataset.py`:

```python
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_generate_dataset_writes_pngs_and_manifest(tmp_path):
    out_dir = tmp_path / "out"
    cmd = [sys.executable, "generate_dataset.py", "--num-samples", "3",
           "--split", "test", "--output-dir", str(out_dir),
           "--geometric-profile", "drift"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    manifest_path = out_dir / "manifest.csv"
    assert manifest_path.exists()
    rows = list(csv.DictReader(open(manifest_path)))
    assert len(rows) == 3
    expected_cols = {"id", "reference_path", "search_path", "gt_x", "gt_y",
                     "canvas_seed", "crop_index", "jitter_profile",
                     "imaging_noise_profile", "geometric_profile",
                     "scale_ratio", "rotation_deg"}
    assert set(rows[0]) == expected_cols
    for row in rows:
        assert row["geometric_profile"] == "drift"
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        assert ref_img.shape == (100, 100)
        assert search_img.shape == (1000, 1000)


def test_generate_dataset_rejects_seed_outside_split_range(tmp_path):
    out_dir = tmp_path / "out"
    cmd = [sys.executable, "generate_dataset.py", "--num-samples", "1",
           "--split", "test", "--seed", "0", "--output-dir", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "outside" in (result.stdout + result.stderr)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd puthere && python -m pytest tests/test_generate_dataset.py -v`
Expected: FAIL — `generate_dataset.py` doesn't exist yet (`FileNotFoundError` inside the subprocess, or `result.returncode` nonzero / Python reporting "can't open file").

- [ ] **Step 3: Implement**

Create `generate_dataset.py` at `puthere/`'s root:

```python
#!/usr/bin/env python3
"""Generate a persisted Drift-Sense localization dataset split: reference/
search PNG pairs plus a manifest.csv of ground truth and generation
metadata, using the exact same on-the-fly generator (src/localizer/data.py)
the training pipeline itself draws from -- so a dataset written by this
script is representative of what the model was actually trained/evaluated
on, not a separately-diverged generator.

Example:
    python generate_dataset.py --split test --num-samples 50 --seed 200000 \
        --output-dir ./output --imaging-noise-profile harsh \
        --geometric-profile drift
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from src.localizer.config import LocalizerConfig
from src.localizer.data import generate_canvas_bundle, sample_pair, split_seed_range


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="which canvas-disjoint seed range to draw from")
    p.add_argument("--seed", type=int, default=None,
                   help="first canvas seed; defaults to the chosen split's "
                        "own range start. Must fall inside that range.")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--jitter-profile", default="normal",
                   choices=["normal", "zero", "shifted"])
    p.add_argument("--imaging-noise-profile", default="normal",
                   choices=["normal", "harsh"])
    p.add_argument("--geometric-profile", default="normal",
                   choices=["normal", "drift"])
    p.add_argument("--crops-per-canvas", type=int, default=1,
                   help="reference crops to draw per generated canvas "
                        "before moving to the next canvas seed")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = LocalizerConfig()
    lo, hi = split_seed_range(args.split, cfg)
    start_seed = args.seed if args.seed is not None else lo
    if not (lo <= start_seed < hi):
        raise SystemExit(
            f"--seed {start_seed} is outside the {args.split} split's own "
            f"canvas-disjoint range [{lo}, {hi}) -- pick a seed inside that "
            f"range so generated data doesn't overlap canvases used for a "
            f"different split.")

    ref_dir = os.path.join(args.output_dir, "reference")
    search_dir = os.path.join(args.output_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    fieldnames = ["id", "reference_path", "search_path", "gt_x", "gt_y",
                 "canvas_seed", "crop_index", "jitter_profile",
                 "imaging_noise_profile", "geometric_profile",
                 "scale_ratio", "rotation_deg"]

    i = 0
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        seed = start_seed
        while i < args.num_samples:
            if seed >= hi:
                raise SystemExit(
                    f"ran out of canvas seeds in the {args.split} split's "
                    f"range before reaching --num-samples {args.num_samples} "
                    f"(wrote {i}) -- lower --num-samples, raise "
                    f"--crops-per-canvas, or start from a lower --seed.")
            bundle = generate_canvas_bundle(seed, args.jitter_profile,
                                            args.imaging_noise_profile)
            for k in range(args.crops_per_canvas):
                if i >= args.num_samples:
                    break
                s = sample_pair(bundle, seed, k, args.imaging_noise_profile,
                                args.geometric_profile)

                ref_path = os.path.join(ref_dir, f"{i:05d}.png")
                search_path = os.path.join(search_dir, f"{i:05d}.png")
                cv2.imwrite(ref_path, s["reference_img_u8"])
                cv2.imwrite(search_path, bundle["search_img"])

                writer.writerow({
                    "id": i, "reference_path": ref_path, "search_path": search_path,
                    "gt_x": s["gt_x"], "gt_y": s["gt_y"], "canvas_seed": seed,
                    "crop_index": k, "jitter_profile": args.jitter_profile,
                    "imaging_noise_profile": args.imaging_noise_profile,
                    "geometric_profile": args.geometric_profile,
                    "scale_ratio": round(s["scale_ratio"], 4),
                    "rotation_deg": round(s["rotation_deg"], 4),
                })
                print(f"[{i + 1}/{args.num_samples}] seed={seed} crop={k} -> "
                      f"gt=({s['gt_x']:.1f}, {s['gt_y']:.1f})")
                i += 1
            seed += 1

    print(f"wrote {i} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd puthere && python -m pytest tests/test_generate_dataset.py -v -m slow`
Expected: PASS, both tests.

- [ ] **Step 5: Run the full fast test suite**

Run: `cd puthere && python -m pytest tests/ -q -m "not slow"`
Expected: `104 passed, 2 deselected` (the seed-range-rejection test isn't `slow`-marked).

- [ ] **Step 6: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "feat: add generate_dataset.py (persisted PNG pairs + manifest.csv)" 2>&1 || true
```

---

### Task 7: `localize.py` (new, top-level, batch-capable)

**Files:**
- Create: `localize.py`
- Test: `tests/test_localize.py`

**Interfaces:**
- Consumes: `load_model`, `predict_pair` from `src/localizer/inference.py` (Task 5); the `tiny_checkpoint` fixture (Task 5); manifest CSVs shaped like Task 6's output (columns `reference_path`, `search_path` at minimum).
- Produces: `predictions.csv` with columns `id, predicted_x, predicted_y, confidence, runtime_ms`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_localize.py`:

```python
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_pair(tmp_path, name):
    rng = np.random.default_rng(hash(name) % (2**31))
    ref_path = tmp_path / f"{name}_ref.png"
    search_path = tmp_path / f"{name}_search.png"
    cv2.imwrite(str(ref_path), rng.integers(0, 255, (100, 100), dtype=np.uint8))
    cv2.imwrite(str(search_path), rng.integers(0, 255, (1000, 1000), dtype=np.uint8))
    return str(ref_path), str(search_path)


@pytest.mark.slow
def test_localize_single_pair_matches_predict_py_format(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    ref_path, search_path = _write_pair(tmp_path, "single")
    cmd = [sys.executable, "localize.py", "--checkpoint", ckpt_path,
           "--reference", ref_path, "--search", search_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    parts = result.stdout.strip().split(",")
    assert len(parts) == 3
    float(parts[0]); float(parts[1]); float(parts[2])  # all parse as numbers


@pytest.mark.slow
def test_localize_batch_mode_writes_predictions_csv(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    ref1, search1 = _write_pair(tmp_path, "a")
    ref2, search2 = _write_pair(tmp_path, "b")

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "reference_path", "search_path"])
        writer.writeheader()
        writer.writerow({"id": 0, "reference_path": ref1, "search_path": search1})
        writer.writerow({"id": 1, "reference_path": ref2, "search_path": search2})

    out_path = tmp_path / "predictions.csv"
    cmd = [sys.executable, "localize.py", "--checkpoint", ckpt_path,
           "--manifest", str(manifest_path), "--output", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    rows = list(csv.DictReader(open(out_path)))
    assert len(rows) == 2
    assert set(rows[0]) == {"id", "predicted_x", "predicted_y", "confidence", "runtime_ms"}
    for row in rows:
        float(row["predicted_x"]); float(row["confidence"]); float(row["runtime_ms"])


def test_localize_requires_reference_and_search_together():
    cmd = [sys.executable, "localize.py", "--checkpoint", "x.pt", "--reference", "r.png"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "together" in (result.stdout + result.stderr)


def test_localize_requires_some_mode():
    cmd = [sys.executable, "localize.py", "--checkpoint", "x.pt"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd puthere && python -m pytest tests/test_localize.py -v`
Expected: FAIL — `localize.py` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `localize.py` at `puthere/`'s root:

```python
#!/usr/bin/env python3
"""Predict reference-in-search centres: a single pair, or an evaluator-
provided batch, through one shared inference path
(src/localizer/inference.py) -- no source-code changes needed to switch
between the two modes.

Single pair (same contract as scripts/predict.py):
    python localize.py --checkpoint checkpoints/production_v2/best.pt \
        --reference ref.png --search search.png

Batch (reads reference_path/search_path columns from any manifest,
including one an evaluator supplies -- e.g. generate_dataset.py's output):
    python localize.py --checkpoint checkpoints/production_v2/best.pt \
        --manifest output/test/manifest.csv --output predictions.csv
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.localizer.inference import load_model, predict_pair


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", help="single-pair mode: reference image path")
    p.add_argument("--search", help="single-pair mode: search image path")
    p.add_argument("--manifest", help="batch mode: CSV with reference_path/search_path columns")
    p.add_argument("--output", help="batch mode: where to write predictions.csv")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if bool(args.reference) != bool(args.search):
        p.error("--reference and --search must be given together")
    if args.manifest and args.reference:
        p.error("give either --reference/--search (single pair) or --manifest (batch), not both")
    if not args.manifest and not args.reference:
        p.error("must give either --reference/--search or --manifest")
    if args.manifest and not args.output:
        p.error("--manifest requires --output")
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    if args.manifest:
        with open(args.manifest, newline="") as f:
            rows = list(csv.DictReader(f))
        out_rows = []
        for row in rows:
            t0 = time.perf_counter()
            result = predict_pair(model, row["reference_path"], row["search_path"], device)
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            out_rows.append({
                "id": row.get("id", ""),
                "predicted_x": f"{result['x']:.3f}",
                "predicted_y": f"{result['y']:.3f}",
                "confidence": f"{result['confidence']:.4f}",
                "runtime_ms": f"{runtime_ms:.2f}",
            })
        out_parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_parent, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "predicted_x", "predicted_y",
                                                    "confidence", "runtime_ms"])
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"wrote {len(out_rows)} predictions to {args.output}")
    else:
        result = predict_pair(model, args.reference, args.search, device)
        if args.verbose:
            print(f"predicted_x={result['x']:.2f} predicted_y={result['y']:.2f} "
                  f"confidence={result['confidence']:.4f}")
        else:
            print(f"{result['x']:.2f},{result['y']:.2f},{result['confidence']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd puthere && python -m pytest tests/test_localize.py -v -m slow`
Expected: PASS, all four tests.

- [ ] **Step 5: Run the full fast test suite**

Run: `cd puthere && python -m pytest tests/ -q -m "not slow"`
Expected: `106 passed, 2 deselected` (two of the four new tests aren't `slow`-marked).

- [ ] **Step 6: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "feat: add localize.py (batch-capable evaluator entry point)" 2>&1 || true
```

---

### Task 8: `scripts/validation_report.py`

**Files:**
- Create: `scripts/validation_report.py`
- Test: `tests/test_validation_report.py`

**Interfaces:**
- Consumes: `LocalizerDataset` with `imaging_noise_profile`/`geometric_profile` (Tasks 1-2); `localization_error` from `src/localizer/metrics.py`; the `tiny_checkpoint` fixture (Task 5).
- Produces: `<out-dir>/validation_report.json`, `<out-dir>/validation_report.md`, `<out-dir>/failure_case.png`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validation_report.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_validation_report_produces_required_outputs(tmp_path, tiny_checkpoint):
    ckpt_path, _device = tiny_checkpoint
    out_dir = tmp_path / "results"
    cmd = [sys.executable, "scripts/validation_report.py",
           "--checkpoint", ckpt_path, "--n-per-condition", "3",
           "--out-dir", str(out_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr

    report = json.load(open(out_dir / "validation_report.json"))
    assert set(report["conditions"]) == {
        "noise=normal_geom=normal", "noise=harsh_geom=normal",
        "noise=normal_geom=drift", "noise=harsh_geom=drift",
    }
    for cond in report["conditions"].values():
        assert cond["n"] == 3
        for t in ("5.0", "4.0", "2.0", "1.0"):
            assert f"pass_rate@{t}px" in cond
        assert "mean_error_px" in cond and "median_error_px" in cond
        assert "worst_error_px" in cond
        assert "runtime_ms_mean" in cond and "runtime_ms_median" in cond

    assert "hardware" in report
    assert "python_version" in report
    assert "timing_method" in report
    assert report["failure_case"]["root_cause"]

    assert (out_dir / "validation_report.md").exists()
    assert (out_dir / "failure_case.png").stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd puthere && python -m pytest tests/test_validation_report.py -v -m slow`
Expected: FAIL — `scripts/validation_report.py` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `scripts/validation_report.py`:

```python
#!/usr/bin/env python3
"""Spec-required validation report: runs the localizer across a noise x
geometry condition matrix and reports Euclidean error (mean/median/worst),
pass rate @5/4/2/1px, sub-pixel detail, runtime per pair (with hardware/
Python version/timing method), and one visualized failure case with a
root-cause note.

Example:
    python scripts/validation_report.py \
        --checkpoint checkpoints/production_v2/best.pt \
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd puthere && python -m pytest tests/test_validation_report.py -v -m slow`
Expected: PASS.

- [ ] **Step 5: Run the full fast test suite**

Run: `cd puthere && python -m pytest tests/ -q -m "not slow"`
Expected: `106 passed, 3 deselected` (this new test is `slow`-marked, so the passed count is unchanged from Task 7 and the deselected count grows by one).

- [ ] **Step 6: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "feat: add scripts/validation_report.py (spec threshold/runtime/failure-case report)" 2>&1 || true
```

---

### Task 9: README updates

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: exact CLI contracts from Tasks 3, 4, 6, 7, 8 (flag names, output file names).

- [ ] **Step 1: Update the file tree (§1)**

In `README.md`, find the file-tree code block (starts `src/` at line 23) and:
- After the `localizer/` block's `metrics.py` line, add:
  ```
      inference.py             # shared checkpoint-load + single-pair predict, used by predict.py and localize.py
  ```
- After the `scripts/` block's `predict.py` line, add:
  ```
    validation_report.py      # spec-required threshold/runtime/failure-case report across noise x geometry conditions
  ```
- After the closing of the `scripts/` block (before `checkpoints/m3_hn_r24/best.pt`), add two new top-level entries:
  ```
  generate_dataset.py         # persisted reference/search PNG pairs + manifest.csv, spec-compliant dataset generator
  localize.py                  # single-pair or evaluator-batch inference, no source changes needed between modes
  ```
- After the `generate_mxn_dram_dataset.py` line near the bottom, add:
  ```
  references/CITATIONS.md    # public sources for DRAM structure, SEM noise modeling, and augmentation practice
  results/                    # validation_report.py output (JSON + Markdown + failure-case PNG)
  ```

- [ ] **Step 2: Document coordinate convention and the multiple-matches rule (§3)**

In the "Quick start" section (§3), right after the existing "**Input requirements**" bullet list (ends `...there's no way for the script to detect that from pixel data alone.`), add:

```markdown
**Coordinate convention:** origin `(0, 0)` is the search image's top-left
corner; `x` increases rightward, `y` increases downward — standard image-array
convention, not math/plot convention. Predicted coordinates are always given
in **search-image pixels**, regardless of the reference's native resolution.

**Multiple matches:** if the reference pattern genuinely repeats within the
search image (a real possibility for periodic DRAM lattices), the decoder's
NMS + centre-tiebreak logic (`src/localizer/decode.py`) selects the
candidate closest to the search image's centre, matching the spec's
tie-break rule — this is inherent to the decode step, not a separate flag.
```

- [ ] **Step 3: Add flags to the training table (§4) and a new robustness-profile section**

In the training flags table in §4, add two rows after the `--jitter-profile` row:

```markdown
| `--imaging-noise-profile` | `normal` | `normal` / `harsh` — widens acquisition-noise and polygon-distortion knobs; see `src/localizer/data.py`'s `IMAGING_NOISE_PROFILES` |
| `--geometric-profile` | `normal` | `normal` / `drift` — adds ~1-2 degree rotation and 9:1-11:1 scale-ratio jitter to reference crops; see `src/localizer/data.py`'s `GEOMETRIC_PROFILES` |
| `--init-from` | *(none)* | warm-start weights from a different run's checkpoint, then train fresh from step 0 under this run's own schedule/profiles (for fine-tuning into a new profile after a prior run's LR schedule has already decayed) |
```

Immediately after §4's existing content (before the `---` separator that starts §5), add a new subsection:

```markdown
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
```

- [ ] **Step 4: Add sections for `generate_dataset.py`, `localize.py`, and `validation_report.py`**

Replace §8 ("Generating your own labeled dataset") entirely with:

```markdown
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
```

Renumber the original §9 ("Further reading") to §11, and update its internal cross-references if any refer to old section numbers by name (check for phrases like "§8" or "section 8" in the "Further reading" text — the current content doesn't appear to reference section numbers directly, so this should be a no-op beyond the heading number itself).

- [ ] **Step 5: Verify every command in the README actually runs**

From `puthere/`, run each newly-documented command with small sample counts against the real shipped checkpoint, confirming no typos/path errors:

```bash
cd /home/nihal/Desktop/Huggingface/puthere
python generate_dataset.py --split test --num-samples 3 --output-dir /tmp/readme_check_gen
python localize.py --checkpoint checkpoints/m3_hn_r24/best.pt \
    --manifest /tmp/readme_check_gen/manifest.csv --output /tmp/readme_check_pred.csv
python scripts/validation_report.py --checkpoint checkpoints/m3_hn_r24/best.pt \
    --n-per-condition 3 --out-dir /tmp/readme_check_report
```

Expected: all three exit 0 and print their success messages: `wrote 3 samples to ...` from `generate_dataset.py`, `wrote 3 predictions to ...` from `localize.py` (3, matching the manifest's row count), and `wrote .../validation_report.json, ...` from `validation_report.py`.

- [ ] **Step 6: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "docs: document new scripts, robustness profiles, coordinate convention, multiple-matches rule" 2>&1 || true
```

---

### Task 10: Verify `requirements.txt` is dry-run-clean

**Files:**
- Verify only: `requirements.txt` (no expected content changes)

**Interfaces:** none (verification task).

- [ ] **Step 1: Confirm every import used across the codebase resolves from a listed package**

Run from `puthere/`:
```bash
grep -rhoE "^\s*(import|from)\s+[a-zA-Z0-9_.]+" --include=*.py . \
    | awk '{print $2}' | cut -d. -f1 | sort -u
```
Cross-check the output against `requirements.txt`'s packages (`numpy`, `opencv-python-headless` → imports as `cv2`, `pillow` → imports as `PIL`, `pytest`, `streamlit`, `matplotlib`, `torch`, `torchvision`) plus Python's standard library (`os`, `sys`, `csv`, `json`, `time`, `argparse`, `platform`, `subprocess`, `random`, `tempfile`, `dataclasses`, `__future__`, etc. — no action needed for these). Confirm nothing appears that isn't covered by either list. `cv2`/`PIL` are expected to show up even though they're not literally named `opencv-python-headless`/`pillow` — that's normal (import name vs. package name differ for both).

- [ ] **Step 2: Dry-run install in a clean virtualenv**

```bash
cd /home/nihal/Desktop/Huggingface/puthere
python -m venv /tmp/drift_sense_clean_env_check
source /tmp/drift_sense_clean_env_check/bin/activate
pip install -r requirements.txt
pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130
python -m pytest tests/ -q -m "not slow"
deactivate
rm -rf /tmp/drift_sense_clean_env_check
```
Expected: all installs succeed, and the fast test suite passes at whatever count Task 9 left it at (106 passed, 3 deselected). If this fails, fix `requirements.txt` (add any genuinely missing package) and re-run — but no gap is expected given Step 1's check.

- [ ] **Step 3: No commit needed**

This task only verifies existing state; skip the commit step unless Step 2 uncovered a real gap requiring a `requirements.txt` edit (in which case commit that specific fix with message `fix(deps): add <package> to requirements.txt`).

---

### Task 11: `references/CITATIONS.md`

**Files:**
- Create: `references/CITATIONS.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Write the citations file**

Create `references/CITATIONS.md` with the following content (these sources were verified during planning — real, publicly accessible URLs, not invented):

```markdown
# Public sources

Citations backing this project's synthetic-data and augmentation design
choices, per the hackathon spec's requirement to justify structures, noise,
and augmentations against credible public sources.

## DRAM 1T-1C cell structure (word lines, bit lines, capacitor storage)

- imec, "DRAM peripheral transistors technology platform."
  <https://www.imec-int.com/en/articles/technology-platform-thermally-stable-dram-peripheral-transistors>
  Describes the 1-transistor/1-capacitor DRAM cell and its access-transistor
  role, backing `src/patterns/dram.py`'s word-line/bit-line/contact/
  capacitor layout model.
- SemiAnalysis, "The Memory Wall: Past, Present, and Future of DRAM."
  <https://newsletter.semianalysis.com/p/the-memory-wall>
  Industry-level explanation of the 1T-1C array structure (word lines
  driving access-transistor gates, bit lines carrying the sensed charge)
  that this project's `cell_pitch_nm`/`word_line_*`/`bit_line_*` parameters
  model as free, non-proprietary parameters (no specific process node is
  implied — see `generate_dataset.py`'s own module docstring language,
  matching AFB's precedent).

## SEM imaging noise and degradation modeling

- "Correction of Scanning Electron Microscope Imaging Artifacts in a Novel
  Digital Image Correlation Framework," *Experimental Mechanics*
  (Springer), open access via PMC.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6541586/>
  Documents drift/charging-driven image distortion in real SEM capture --
  directly motivates `src/sem_imaging.py`'s `apply_raster_drift`,
  `add_charging_streaks`, and the `drift_jitter_px`/`shear_amplitude_px`
  parameters this project's `IMAGING_NOISE_PROFILES` randomizes.
- "Scanning Electron Microscope Image Signal-to-Noise Ratio Monitoring for
  Micro-Nanomanipulation," open access via HAL.
  <https://hal.science/hal-01051309/document>
  Establishes that SEM pixel noise is dominated by Poisson-distributed shot
  noise from the primary/secondary electron count, plus detector/amplifier
  noise -- the physical basis for this project's `dose`-driven shot-noise
  simulation and `detector_noise_sigma` parameter in `sem_imaging.py`.

## Data augmentation for scale/rotation robustness in matching tasks

- "An Efficient Deep Template Matching and In-Plane Pose Estimation Method
  via Template-Aware Dynamic Convolution," arXiv.
  <https://arxiv.org/html/2510.01678>
  Uses rotation/shear-based augmentation during training so a deep template
  matcher regresses position, rotation, and scale robustly -- the same
  training-time-augmentation strategy (rather than inference-time
  multi-hypothesis search) this project's `GEOMETRIC_PROFILES["drift"]`
  applies to reference crops.
- "Who Handles Orientation? Investigating Invariance in Feature Matching,"
  arXiv. <https://arxiv.org/html/2604.11809v1>
  Finds that rotation robustness in learned feature matchers can emerge
  from training-distribution diversity (with or without explicit rotation
  augmentation), supporting this project's design choice to bake
  scale/rotation robustness into the training distribution rather than
  adding an inference-time geometric search (which would cost runtime,
  directly graded by the spec).
```

- [ ] **Step 2: Verify every URL actually resolves**

```bash
for u in \
  "https://www.imec-int.com/en/articles/technology-platform-thermally-stable-dram-peripheral-transistors" \
  "https://newsletter.semianalysis.com/p/the-memory-wall" \
  "https://pmc.ncbi.nlm.nih.gov/articles/PMC6541586/" \
  "https://hal.science/hal-01051309/document" \
  "https://arxiv.org/html/2510.01678" \
  "https://arxiv.org/html/2604.11809v1" ; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -L "$u")
  echo "$code  $u"
done
```
Expected: every URL returns a `2xx` status (a `403` from a site that blocks scripted `curl` User-Agents but is verifiably real/browsable is acceptable too — use judgment, but a `404` means the URL is wrong and must be fixed before proceeding).

- [ ] **Step 3: Commit**

```bash
git -C /home/nihal/Desktop/Huggingface/puthere add -A
git -C /home/nihal/Desktop/Huggingface/puthere commit -m "docs: add references/CITATIONS.md with verified public sources" 2>&1 || true
```

---

## After all tasks: fine-tune training run

Not a plan task (no test cycle — it's a long-running GPU job to be launched
and monitored interactively, matching how every other training run this
session was handled). Once Tasks 1-4 are done and merged:

```bash
cd /home/nihal/Desktop/Huggingface/puthere
python scripts/train.py --run-name production_v3 \
    --init-from checkpoints/production_v2/best.pt \
    --imaging-noise-profile harsh --geometric-profile drift \
    --lambda-hn 0.5 --max-steps 10000 --steps-this-run 10000
```

Watch the validation accuracy curve the same way `production_v2`'s harsh-noise
fine-tune was monitored earlier this session; extend with `--resume` in
further 10k-step chunks (keeping `--max-steps` consistent with whatever total
budget is chosen) if accuracy is still climbing when this chunk ends. Once a
good checkpoint exists, re-run Task 8's `validation_report.py` against it for
real (non-tiny-checkpoint) numbers to go into the eventual PPTX.
