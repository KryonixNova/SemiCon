
# SemiCon
Competition
# DRAM Synthetic SEM + Reference/Search Localization Benchmark

This project has two layers:

1. A **synthetic DRAM layout + SEM-style image generator** — produces GDSII layouts of a fake memory-cell array (word lines, bit lines, contacts, capacitors, dummy border, defects) and renders them as grayscale images that look like SEM micrographs, with realistic noise/blur/process-variation.
2. A **reference/search localization benchmark** built on top of it — generates a high-res "reference" patch (10x zoom) and a low-res "search" image of the whole die, independently rotated and noised, and measures how well a SuperPoint+LightGlue+RANSAC pipeline can find the reference's true location inside the search image.

No proprietary process data is used anywhere — see the docstring at the top of `dram_layout_generator.py` for the (expired-patent / textbook) basis of every structure.

This document is a from-scratch guide: environment setup, what each file does, how to run everything, the key design decisions and why they were made, and — importantly — the current honest state of the benchmark result and what's already been ruled out as a fix.

---

## 1. Environment setup

Everything runs under a dedicated conda env. Recreate it (name is arbitrary, referred to below as `royl`):

```bash
conda create -n royl python=3.12
conda activate royl
pip install klayout opencv-python numpy torch pytest
pip install "git+https://github.com/cvg/LightGlue.git"
```

Notes:
- `lightglue` is **not** a real PyPI package under that bare name — it must be installed from the GitHub URL above.
- SuperPoint/LightGlue pretrained weights auto-download via `torch.hub` on first use (~50MB, cached at `~/.cache/torch/hub/checkpoints/`) — the first run needs network access, later runs don't.
- All commands in this doc assume the env's python is at `/home/nihal/miniconda3/envs/royl/bin/python` — substitute your own path (or just `python` inside an activated env).
- CUDA is used automatically if available (`torch.cuda.is_available()`), otherwise falls back to CPU (slower but works).

Sanity-check the install:

```bash
cd /home/nihal/Desktop
python -m pytest test_sem_render.py test_reference_search_pairs.py test_matcher.py test_localizer.py test_benchmark.py test_dram_layout.py -v
```

All 64 tests should pass.

---

## 2. File map

Only these files are part of this project (tracked in git); the working directory has other unrelated scratch files/PDFs from other work — ignore those.

| File | Role |
|---|---|
| `dram_layout_generator.py` | Core layout engine. `DRAMParams` (all tunable knobs), `ProcessVariation` (10 perturbation methods: overlay, LER, jitter, etc.), `DRAMGenerator` (builds the GDSII layout layer by layer, adds defects, writes `.gds` + metadata `.json`). No image rendering — pure geometry. |
| `sem_render.py` | Shared rasterization: turns GDS geometry into SEM-style grayscale images. Supersampled opaque painting (4x, then area-average downsample for antialiasing), circle rendering for contacts, two-tone halo+mark rendering for point defects, periodic-margin tiling, and the noise pipeline (Sobel edge emphasis, beam blur, Poisson shot noise, Gaussian read noise, per-image gain/offset jitter). |
| `generate_dram_images.py` | Standalone batch script: generates N full-array SEM images + human-QA debug images (colored by layer, ground-truth box burned in) using `dram_layout_generator.py` + `sem_render.py`. Not part of the benchmark pipeline — a earlier/simpler entry point for just looking at the generator's output. |
| `reference_search_pairs.py` | Benchmark dataset generator. For a given seed: builds one GDS layout, renders a **search image** (whole array, native resolution, own independent rotation+noise) and a **reference image** (a small patch, rendered directly from GDS at `zoom_ratio`x finer pixel pitch — genuinely higher resolution, not just a smaller crop — with its own independent rotation+noise). Returns `ReferenceSearchSample` with ground-truth `true_center_px`. |
| `matcher.py` | Thin wrapper around SuperPoint (keypoint extractor) + LightGlue (learned matcher). `SuperPointLightGlueMatcher.extract(img)` / `.match(feats_a, feats_b)`. |
| `localizer.py` | The localization algorithm. `DeepMatchLocalizer.localize(reference_img, search_img, zoom_ratio)`: downsamples the reference by `zoom_ratio` to normalize scale, matches, then runs **sequential RANSAC** with `cv2.estimateAffinePartial2D` (similarity transform — rotation + uniform scale + translation, deliberately not full homography) to handle the fact that a periodic lattice can produce multiple self-consistent-but-wrong transform hypotheses; picks whichever hypothesis's predicted center is closest to the search image's own center. |
| `benchmark.py` | CLI harness. `run_benchmark(n, tolerance_px, out_dir, ...)`: loads the matcher/localizer once, loops N samples, reports success rate + timing (mean/median/p95) + writes `benchmark_results.json` + saves a few annotated example PNGs. |
| `verify_matchability.py` | Standalone diagnostic script (not part of the reviewed pipeline). Bypasses `matcher.py`'s default match-acceptance threshold to report the **raw** best SuperPoint/LightGlue confidence per sample — used to check whether a rendering change actually helps before spending time on a full benchmark run. |
| `test_*.py` | pytest suites, one per module above (except `generate_dram_images.py`, `verify_matchability.py`, which aren't unit-tested by design). |
| `docs/superpowers/specs/*.md`, `docs/superpowers/plans/*.md` | The design docs and implementation plans this was built from, in order — see §5 below for a guided reading order and what each one decided. |

---

## 3. How to run things

### 3.1 Generate a batch of standalone SEM images (just looking at the generator)

```bash
python generate_dram_images.py --n 20 --out-dir dram_dataset --debug-images
```

Produces `n` clean SEM images (model-facing) plus, with `--debug-images`, colored-by-layer QA images with the reference box burned in. Run `--help` for the full parameter list (array size, pitches, defect rates, variation sigmas — all `DRAMParams` fields are exposed).

### 3.2 Run the localization benchmark

```bash
python benchmark.py --n 30 --tolerance-px 5 --out-dir benchmark_results
```

Writes `benchmark_results/benchmark_results.json` (per-sample records + aggregate stats) and a handful of annotated example PNGs (green cross = true center, red cross = predicted center). `--help` for all flags (`--n`, `--tolerance-px`, `--out-dir`, `--seed-start`, `--n-examples`).

### 3.3 Diagnose match quality directly (before/after a rendering change)

```bash
python verify_matchability.py --n 5
```

Prints the raw best SuperPoint/LightGlue confidence per sample plus a mean, compared against the two baselines noted in its own output. No pass/fail threshold — read the numbers.

### 3.4 Run tests

```bash
python -m pytest test_sem_render.py test_reference_search_pairs.py test_matcher.py test_localizer.py test_benchmark.py test_dram_layout.py -v
```

---

## 4. Key design decisions (and why)

These were explicit, deliberate choices made during brainstorming — don't silently revert them without understanding why:

- **Similarity transform, not homography.** `localizer.py` uses `cv2.estimateAffinePartial2D` (4 DOF: rotation + uniform scale + translation), not `cv2.findHomography` (8 DOF). The synthetic dataset only ever has translation, small rotation, and a known uniform zoom — never perspective distortion — so a similarity transform is the *correct* geometric model, not a simplification. Revisit only if perspective distortion is deliberately added to the dataset later.
- **Zoom-ratio semantics.** The reference image is rendered directly from GDS at `pixels_per_nm × zoom_ratio` — genuinely finer pixel pitch for the same physical patch — not a smaller-area crop at the same density. `localizer.py` downsamples the reference by `zoom_ratio` before matching to normalize scale against the search image.
- **Sequential RANSAC for periodicity ambiguity.** A single RANSAC pass finds the largest-consensus transform, which on a periodic lattice can easily be "off by one lattice pitch" and still self-consistent. `localizer.py._find_hypotheses` runs RANSAC repeatedly (removing each hypothesis's inliers before the next pass, up to `max_hypotheses`), then picks whichever hypothesis's predicted center is closest to the search image's geometric center.
- **`layout.find_layer()` returns `None`, not `-1`, for a never-instantiated layer.** (e.g. a zero-defect layout never creates layer `(5,0)`.) `if li >= 0:` crashes with `TypeError`; must be `if li is not None:`. Fixed once in `sem_render.py`, worth remembering if you touch KLayout layer lookups elsewhere.
- **Isotropic `pixels_per_nm`, periodic-margin tiling instead of anisotropic scaling.** A single uniform pixel scale (`min(w_ratio, h_ratio)`) keeps features undistorted; for a non-square array this leaves a margin on one axis, filled by *repeating the real periodic lattice* at its actual pitch (`_tile_margin`) rather than leaving it blank or stretching the image.
- **4x supersampling + area-average downsample** for all rendering, to avoid aliasing on thin lattice lines.
- **Contacts render as circles, not rectangles** (`CIRCLE_LAYERS = {(4,0)}` in `sem_render.py`), and **only** for the contact layer — this was a deliberate visual-style match to a real SEM reference image the project owner supplied, and is also the single most locally-distinctive landmark in the image for feature matching.
- **Defects get two-tone rendering**: particle/scratch defects get a lighter "disturbed-zone" halo with a darker mark on top (again matching a supplied reference image); CMP dishing gets a soft blended tint, not a flat overwrite (a large diffuse region, not a point defect — flattening it entirely erases the underlying lattice, which is not what real dishing looks like).
- **Defect-type allowlist matters.** `dram_layout_generator.py`'s defect list has **7** types, not 3: `particle`, `scratch`, `cmp_dishing` are point/diffuse defects worth rendering specially; `missing_contact`/`missing_capacitor` mean a feature was *deliberately omitted* (already correctly absent from the geometry — do NOT paint anything there, it would draw a bright marker exactly where something is supposed to be *missing*); `broken_wl`/`broken_bl` carry the *entire pre-break line's bbox*, not a point location (painting on that draws a full-array stripe). `sem_render.py._paint_defects` only special-renders `{"particle", "scratch"}` (via `POINT_DEFECT_TYPES`) plus `cmp_dishing` — everything else is left alone. This was a real bug caught by a final-review pass; if you extend defect rendering, re-check this allowlist against whatever `_add_defects()` actually emits.
- **Reference-patch selection stays uniform-random** over the array (`dram_layout_generator.py`, deliberately never touched by any rendering work) — not biased toward defect-containing regions, even though that would inflate apparent match quality. This is a real constraint, not an oversight: keeping it makes any benchmark number an honest measure of the *plain-lattice* case too, not just the easy defect-containing case.

---

## 5. How this was built — reading order

The full paper trail is in `docs/superpowers/`, each with a design spec (`specs/`) then an implementation plan (`plans/`), read in this order:

1. **`2026-08-06-dram-layout-generator-design.md`** — the base GDSII layout generator: geometry, process variation, defect model.
2. **`2026-08-06-dram-reference-search-localization-design.md`** — the localization benchmark: why SuperPoint+LightGlue, why a similarity transform not homography, the zoom-ratio semantics, the sequential-RANSAC periodicity fix, and the success-metric definition (Euclidean distance in search-image pixels, within a stated tolerance, over ≥30 randomized cases).
3. **`2026-08-06-dram-rendering-rework-design.md`** — the rendering rework: why (0% match rate, root-cause diagnosis below), circle contacts, two-tone defects, doubled process variation.

Each implementation plan (`plans/*.md`) was executed via Subagent-Driven Development: a fresh implementer subagent per task, a spec+quality review after each task, and a final whole-branch review at the end. All findings from those reviews (including two real bugs caught only at the final whole-branch-review stage, not per-task) are worth reading if you're auditing correctness — they're recorded in the git history's commit messages and were not just rubber-stamped.

---

## 6. Current status — read this before assuming the benchmark "works"

**As of the last commit (`ed4b975`), the benchmark's honest result is 0/30 successful localizations at default settings.** This is not a bug — it's a real, thoroughly investigated result. Full chain of evidence:

1. **Original finding:** `benchmark.py --n 30` on the first rendering style measured 0/30, all `insufficient_matches` — literally zero raw keypoint correspondences in every sample.
2. **Root cause, confirmed by direct SuperPoint/LightGlue probing** (bypassing `matcher.py`'s wrapper): best raw match confidence across the *entire* candidate grid never exceeded ~1%, vs. ~50-100% typical for a confident correct match on natural images. The original rendering was a fine, near-perfectly-periodic plaid grid where every local window looks structurally identical — the textbook "aperture problem" for local feature matching, worsened by contacts (the only naturally unique per-cell landmark) being nearly-invisible 2.7px squares.
3. **Rendering rework** (circle contacts, two-tone defects, ~2x process variation) raised mean raw confidence to **0.169** — a genuine 25x improvement, verified after a final-review pass caught and fixed 3 real bugs that had inflated an earlier (bogus) 0.071 reading (see §4's defect-allowlist note, plus a circle-boundary-clipping bug and an opaque-vs-blended dishing bug).
4. **Still 0/30 after the rework.** Investigated two further levers, both empirically tested and rejected:
   - **Loosening LightGlue's `filter_threshold`** (0.1 → 0.0) does flood the raw match pool (48-151 candidates/sample, up from 0-2), but RANSAC finds **zero genuine inliers** at the localizer's real `min_inliers=4`. Lowering `min_inliers` to its absolute floor (3) makes the localizer report "success" 15/15 — but every prediction is wildly wrong (66-600px error against a 5px tolerance): pure RANSAC overfitting to 3 coincidentally-consistent points out of noise. **Rejected** — this trades honest failure for confidently-wrong answers, which is worse. `matcher.py`/`localizer.py` are unmodified from their reviewed state.
   - **Pushing `reference_search_pairs.py`'s process-variation parameters further** (beyond the already-doubled rework defaults): tested more aggressively (contacts approaching the bit-line pitch, larger sigmas) and it made things *worse* (likely because oversized contacts start overlapping/blobbing together, destroying the one distinctive landmark that was helping). A moderate push also plateaued with no real inliers. **Not committed** — `reference_search_pairs.py` is unchanged from the reviewed rework state.
5. **Decision:** stop tuning within the current architecture. 0/30 is accepted as the honest, fully-diagnosed empirical ceiling of "SuperPoint+LightGlue on a periodic lattice with the current grid topology," not evidence of a code bug anywhere in this pipeline. All 5 localization-plan tasks plus the rendering rework passed their code reviews on their own merits — the benchmark code correctly measures a real, difficult, honestly-reported result.

**The one untried, larger-scoped lever:** a redesign of the grid geometry itself (`dram_layout_generator.py`) — e.g. asymmetric/non-periodic structure, diagonal routing, or a different pitch ratio — rather than just rendering style or process-variation tuning. This has been explicitly out of scope for every plan so far (every plan deliberately left `dram_layout_generator.py`'s topology untouched). If someone picks this up, it needs its own brainstorm → design spec → plan cycle, same as everything else here — it's a real architecture change, not a parameter tweak.

**A final whole-branch code review of the entire localization plan (all 5 tasks, base `dc833c5`, head `230e168`) was dispatched but its result was not available when this document was written** — check for a completed review or re-run one if picking this back up (`.superpowers/sdd/2026-08-06-dram-reference-search-localization/` holds the review package and per-task ledger if the workspace hasn't been cleaned up yet; if the plan's final review came back clean, that workspace should be deleted per the standard close-out process, same as was already done for the rendering-rework plan).

---

## 7. If you're picking this up fresh

1. Read this file, then skim the three design specs in reading order (§5) — they explain *why*, not just *what*.
2. Run the test suite (§1) to confirm your environment is sane.
3. Run `verify_matchability.py --n 5` to confirm the current rendering's raw confidence is still ~0.15-0.17 mean (sanity check nothing regressed).
4. If continuing the localization work: check whether the final whole-branch review mentioned in §6 completed and what it found, before assuming the localization plan is fully closed out.
5. If tackling the 0% success rate: don't re-try threshold-loosening or simple process-variation scaling — both are empirically ruled out (§6). The grid-geometry redesign is the remaining real option, and needs a fresh design spec
