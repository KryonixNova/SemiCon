"""On-the-fly training data for the localizer.

Nothing is persisted. A 10000x10000 fine canvas costs ~0.2 s and 100 MB, so
storing thousands of them is infeasible (200 GB) while regenerating them is
cheap. Every sample is therefore a pure function of its canvas seed and crop
index, which also makes val/test byte-reproducible across runs.

Splits are disjoint *seed ranges*, never random pair splits: all crops from
one canvas share a single search image, so a pair-level split would put
near-duplicates on both sides of the boundary.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src import sem_imaging
from src.localizer.config import LocalizerConfig
from src.patterns import dram as dram_mod
from src.patterns.mxn import generate_mxn_canvas
from src.pipeline import (
    FINE_CANVAS_SIZE_PX, PIXEL_SIZE_REF_NM, PIXEL_SIZE_SEARCH_NM,
    REFERENCE_SIZE_PX, SCALE_FACTOR,
)

# Ablation A1: how aperiodic the lattice is. "zero" makes the mat exactly
# periodic, which is the provably degenerate case.
JITTER_PROFILES = {
    "normal":  {"position_nm": 1.5, "width_frac": 0.10, "dist": "normal"},
    "zero":    {"position_nm": 0.0, "width_frac": 0.00, "dist": "normal"},
    "shifted": {"position_nm": 3.0, "width_frac": 0.20, "dist": "uniform"},
}

STRIP_WIDTH_NM_RANGE = (120.0, 400.0)


def split_seed_range(split: str, config: LocalizerConfig):
    return {
        "train": (config.train_seed_lo, config.train_seed_hi),
        "val": (config.val_seed_lo, config.val_seed_hi),
        "test": (config.test_seed_lo, config.test_seed_hi),
    }[split]


def _standardize(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img)).float()
    return (t - t.mean()) / t.std().clamp_min(1e-6)


def generate_canvas_bundle(canvas_seed: int, jitter_profile: str = "normal") -> dict:
    """Build one fine canvas and its single Search image. Shared by every
    crop taken from this canvas -- this is what makes crops nearly free."""
    profile = JITTER_PROFILES[jitter_profile]
    rng = np.random.default_rng(canvas_seed)

    # Patch the pattern module's jitter globals for the A1 ablation. Restored
    # in `finally` so a profile never leaks into another sample.
    saved = (dram_mod.POSITION_JITTER_NM, dram_mod.WIDTH_JITTER_FRACTION,
             dram_mod.POSITION_JITTER_DIST)
    dram_mod.POSITION_JITTER_NM = profile["position_nm"]
    dram_mod.WIDTH_JITTER_FRACTION = profile["width_frac"]
    dram_mod.POSITION_JITTER_DIST = profile.get("dist", "normal")
    try:
        m = int(rng.integers(1, 6))
        n = int(rng.integers(1, 6))
        strip_w = float(rng.uniform(*STRIP_WIDTH_NM_RANGE))
        zone = generate_mxn_canvas(FINE_CANVAS_SIZE_PX, m, n, strip_w, rng)
    finally:
        (dram_mod.POSITION_JITTER_NM, dram_mod.WIDTH_JITTER_FRACTION,
         dram_mod.POSITION_JITTER_DIST) = saved

    fine = zone["canvas"]
    search = sem_imaging.image_search(
        fine,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=float(rng.uniform(3.0, 8.0)),
        dose=float(rng.uniform(80.0, 400.0)),
        rng=rng,
        shear_amplitude_px=float(rng.uniform(0.3, 2.5)),
        drift_jitter_px=float(rng.uniform(0.1, 1.0)),
        detector_noise_sigma=float(rng.uniform(3.0, 7.0)),
    )
    return {"fine_canvas": fine, "search_img": search,
            "strip_rects": zone["strip_rects"]}


def sample_pair(bundle: dict, canvas_seed: int, crop_index: int) -> dict:
    """Cut one Reference crop from the canvas and image it. The Search image
    is reused from the bundle unchanged."""
    rng = np.random.default_rng(canvas_seed * 1_000_003 + crop_index)
    max_off = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    x0 = int(rng.integers(0, max_off + 1))
    y0 = int(rng.integers(0, max_off + 1))

    crop = bundle["fine_canvas"][y0:y0 + REFERENCE_SIZE_PX,
                                 x0:x0 + REFERENCE_SIZE_PX]
    ref_full = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=float(rng.uniform(3.0, 8.0)),
        dose=float(rng.uniform(1200.0, 3000.0)),
        rng=rng,
        detector_noise_sigma=float(rng.uniform(1.0, 3.0)),
        drift_jitter_px=float(rng.uniform(0.02, 0.2)),
    )
    # Area-average to the search scale, matching how image_search downsamples.
    ref_ds = sem_imaging.downsample_area_average(ref_full, SCALE_FACTOR)

    half = REFERENCE_SIZE_PX / SCALE_FACTOR / 2.0        # 50 px
    return {
        "reference_img": _standardize(ref_ds),
        "search_img": _standardize(bundle["search_img"]),
        "gt_x": x0 / SCALE_FACTOR + half,
        "gt_y": y0 / SCALE_FACTOR + half,
        "canvas_seed": canvas_seed,
    }


_SHUFFLE_BUFFER_SEED_BASE = 0x5eed  # arbitrary constant; only mixed with wid


class LocalizerDataset(IterableDataset):
    """Streams (reference, search, gt) samples for one canvas-disjoint split.

    Each worker takes a strided shard of the split's seed range, generates a
    canvas, and draws `crops_per_canvas` samples from it before moving on to
    the next. Samples are NOT yielded in that canvas-by-canvas order: a
    reservoir-style shuffle buffer randomizes yield order across canvases so
    that a `DataLoader`'s batches mix crops from multiple search images
    instead of collating `batch_size` crops of one byte-identical search
    image (which starves BatchNorm of real batch diversity and wastes
    forward-pass compute on redundant search-image encodes). Which samples
    each worker is responsible for -- the `wid`/`nw` seed-range striding --
    is unchanged; only the order they are yielded in changes.
    """

    def __init__(self, split: str, config: LocalizerConfig,
                 jitter_profile: str = "normal", shuffle_buffer_size: int = 256):
        self.split = split
        self.config = config
        self.jitter_profile = jitter_profile
        self.shuffle_buffer_size = shuffle_buffer_size
        self.lo, self.hi = split_seed_range(split, config)

    def _raw_items(self, wid: int, nw: int):
        for seed in range(self.lo + wid, self.hi, nw):
            bundle = generate_canvas_bundle(seed, self.jitter_profile)
            for k in range(self.config.crops_per_canvas):
                s = sample_pair(bundle, seed, k)
                yield {
                    "reference_img": s["reference_img"].unsqueeze(0),
                    "search_img": s["search_img"].unsqueeze(0),
                    "gt_x": torch.tensor(s["gt_x"], dtype=torch.float32),
                    "gt_y": torch.tensor(s["gt_y"], dtype=torch.float32),
                    "canvas_seed": torch.tensor(seed, dtype=torch.long),
                }

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        nw = 1 if info is None else info.num_workers

        # Deterministic per-worker seed: reproducible across runs (val/test
        # byte-reproducibility must survive this change) but independent
        # across workers so they don't all evict in lockstep.
        shuffle_rng = random.Random(_SHUFFLE_BUFFER_SEED_BASE + wid)
        buffer_size = max(1, self.shuffle_buffer_size)
        buffer: list = []

        for item in self._raw_items(wid, nw):
            if len(buffer) < buffer_size:
                buffer.append(item)
                continue
            j = shuffle_rng.randrange(buffer_size)
            yield buffer[j]
            buffer[j] = item

        shuffle_rng.shuffle(buffer)
        yield from buffer
