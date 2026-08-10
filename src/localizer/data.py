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

import cv2
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

# SEM imaging-artifact severity, sampled per canvas/crop. "normal" reproduces
# this pipeline's original ranges exactly (nothing changes unless a caller
# opts in). "harsh" widens every acquisition-noise and polygon-distortion
# knob src/sem_imaging.py and src/patterns/dram.py already support -- the
# same full parameter set exposed by app.py's Streamlit sliders, all of
# which the localizer's on-the-fly data pipeline previously left at their
# off/default values regardless of profile. Ranges are calibrated at roughly
# 30-50% of app.py's own slider max (its extremes are for exploring where
# the generator breaks, not for training -- going that far would destroy the
# localization signal entirely, not just make it harder).
IMAGING_NOISE_PROFILES = {
    "normal": {
        "search":    {"spot_size_nm": (3.0, 8.0), "dose": (80.0, 400.0),
                      "shear_amplitude_px": (0.3, 2.5), "drift_jitter_px": (0.1, 1.0),
                      "detector_noise_sigma": (3.0, 7.0),
                      "astigmatism_ratio": (1.0, 1.0), "vignette_strength": (0.0, 0.0),
                      "barrel_distortion_k": (0.0, 0.0), "charging_streak_prob": (0.0, 0.0),
                      "charging_streak_intensity": (0.0, 0.0), "speckle_sigma": (0.0, 0.0),
                      "salt_pepper_prob": (0.0, 0.0)},
        "reference": {"spot_size_nm": (3.0, 8.0), "dose": (1200.0, 3000.0),
                      "detector_noise_sigma": (1.0, 3.0), "drift_jitter_px": (0.02, 0.2),
                      "astigmatism_ratio": (1.0, 1.0), "vignette_strength": (0.0, 0.0),
                      "barrel_distortion_k": (0.0, 0.0), "charging_streak_prob": (0.0, 0.0),
                      "charging_streak_intensity": (0.0, 0.0), "speckle_sigma": (0.0, 0.0),
                      "salt_pepper_prob": (0.0, 0.0)},
        "pattern":   {"linewidth_bias_nm": (0.0, 0.0), "corner_rounding_px": (0.0, 0.0)},
    },
    "harsh": {
        "search":    {"spot_size_nm": (3.0, 14.0), "dose": (30.0, 400.0),
                      "shear_amplitude_px": (0.3, 5.0), "drift_jitter_px": (0.1, 2.5),
                      "detector_noise_sigma": (3.0, 14.0),
                      "astigmatism_ratio": (0.7, 1.5), "vignette_strength": (0.0, 0.35),
                      "barrel_distortion_k": (-0.08, 0.08), "charging_streak_prob": (0.0, 2.0),
                      "charging_streak_intensity": (0.0, 1.5), "speckle_sigma": (0.0, 0.3),
                      "salt_pepper_prob": (0.0, 0.015)},
        "reference": {"spot_size_nm": (3.0, 12.0), "dose": (400.0, 3000.0),
                      "detector_noise_sigma": (1.0, 6.0), "drift_jitter_px": (0.02, 0.6),
                      "astigmatism_ratio": (0.8, 1.3), "vignette_strength": (0.0, 0.2),
                      "barrel_distortion_k": (-0.05, 0.05), "charging_streak_prob": (0.0, 0.8),
                      "charging_streak_intensity": (0.0, 0.8), "speckle_sigma": (0.0, 0.15),
                      "salt_pepper_prob": (0.0, 0.008)},
        # Deterministic global CD bias / corner rounding -- one draw per
        # canvas, applied uniformly to every mat in it (not per search vs.
        # reference; both are imaged from the same underlying polygons).
        "pattern":   {"linewidth_bias_nm": (-4.0, 4.0), "corner_rounding_px": (0.0, 3.0)},
    },
}


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


def split_seed_range(split: str, config: LocalizerConfig):
    return {
        "train": (config.train_seed_lo, config.train_seed_hi),
        "val": (config.val_seed_lo, config.val_seed_hi),
        "test": (config.test_seed_lo, config.test_seed_hi),
    }[split]


def _standardize(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img)).float()
    return (t - t.mean()) / t.std().clamp_min(1e-6)


def generate_canvas_bundle(canvas_seed: int, jitter_profile: str = "normal",
                            imaging_noise_profile: str = "normal") -> dict:
    """Build one fine canvas and its single Search image. Shared by every
    crop taken from this canvas -- this is what makes crops nearly free."""
    profile = JITTER_PROFILES[jitter_profile]
    noise_profile = IMAGING_NOISE_PROFILES[imaging_noise_profile]
    noise = noise_profile["search"]
    pattern_noise = noise_profile["pattern"]
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
        # One draw per canvas, applied uniformly across every mat in it --
        # a real process's CD bias/corner rounding doesn't vary region to
        # region within a single die.
        linewidth_bias_nm = float(rng.uniform(*pattern_noise["linewidth_bias_nm"]))
        corner_rounding_px = float(rng.uniform(*pattern_noise["corner_rounding_px"]))
        zone = generate_mxn_canvas(FINE_CANVAS_SIZE_PX, m, n, strip_w, rng,
                                   linewidth_bias_nm=linewidth_bias_nm,
                                   corner_rounding_px=corner_rounding_px)
    finally:
        (dram_mod.POSITION_JITTER_NM, dram_mod.WIDTH_JITTER_FRACTION,
         dram_mod.POSITION_JITTER_DIST) = saved

    fine = zone["canvas"]
    search = sem_imaging.image_search(
        fine,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=float(rng.uniform(*noise["spot_size_nm"])),
        dose=float(rng.uniform(*noise["dose"])),
        rng=rng,
        shear_amplitude_px=float(rng.uniform(*noise["shear_amplitude_px"])),
        drift_jitter_px=float(rng.uniform(*noise["drift_jitter_px"])),
        detector_noise_sigma=float(rng.uniform(*noise["detector_noise_sigma"])),
        astigmatism_ratio=float(rng.uniform(*noise["astigmatism_ratio"])),
        vignette_strength=float(rng.uniform(*noise["vignette_strength"])),
        barrel_distortion_k=float(rng.uniform(*noise["barrel_distortion_k"])),
        charging_streak_prob=float(rng.uniform(*noise["charging_streak_prob"])),
        charging_streak_intensity=float(rng.uniform(*noise["charging_streak_intensity"])),
        speckle_sigma=float(rng.uniform(*noise["speckle_sigma"])),
        salt_pepper_prob=float(rng.uniform(*noise["salt_pepper_prob"])),
    )
    return {"fine_canvas": fine, "search_img": search,
            "strip_rects": zone["strip_rects"]}


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
        # Lets a resumed training chunk start further into the split's canvas
        # range instead of always re-walking from `lo` -- otherwise every
        # chunk of a --resume'd run would repeatedly draw from the same
        # narrow slice near `lo` rather than exploring the full range across
        # the whole run. Assumed to stay within [lo, hi) for realistic step
        # budgets (this pipeline's own real budget of 40000 steps only
        # advances a few thousand seeds into a 100000-wide train range); not
        # wrapped past `hi`, so an offset that large would exhaust a worker's
        # shard early rather than looping back to `lo`.
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
