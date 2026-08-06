"""Reference/search image pairs for the localization benchmark.

Each sample pairs a high-resolution reference render of a small die patch
(at pixels_per_nm x zoom_ratio, mimicking a 10x zoom-in capture) with a
low-resolution wide search render of the whole array (at pixels_per_nm),
each independently rotated and independently noised -- as if the two were
separate acquisitions of the same physical location.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from dram_layout_generator import DRAMParams, DRAMGenerator
from sem_render import render_sem_image, render_sem_patch

DEFAULT_PARAMS_KW = dict(
    rows=64, cols=64,
    cell_pitch_nm=80.0, cell_pitch_bl_nm=60.0,
    wl_width_nm=24.0, bl_width_nm=18.0,
    contact_size_nm=40.0, capacitor_size_nm=30.0,
    n_particles=3, n_scratches=2,
    p_broken_wl=0.02, p_broken_bl=0.02, p_cmp_dishing=0.10,
    overlay_sigma_nm=6.0, linewidth_sigma_nm=4.0, ler_amplitude_nm=3.0,
    contact_sigma_frac=0.15, jitter_sigma_nm=2.5,
)

ROTATION_RANGE_DEG = (1.0, 3.0)  # magnitude range; sign drawn independently


@dataclass
class ReferenceSearchSample:
    reference_img: np.ndarray    # uint8, grayscale
    search_img: np.ndarray       # uint8, grayscale, 1000x1000
    true_center_px: Tuple[float, float]
    zoom_ratio: float
    seed: int


def _draw_rotation_deg(rng: np.random.Generator) -> float:
    mag = rng.uniform(*ROTATION_RANGE_DEG)
    sign = rng.choice([-1.0, 1.0])
    return float(sign * mag)


def _rotate_image(img: np.ndarray, angle_deg: float) -> tuple:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


def rotate_point(point: Tuple[float, float], M: np.ndarray) -> Tuple[float, float]:
    """Apply a cv2 2x3 affine matrix to a single (x, y) point."""
    pt = np.array([[point]], dtype=np.float32)  # shape (1, 1, 2)
    out = cv2.transform(pt, M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def generate_sample(seed: int, tmp_dir: str, **params_kw) -> ReferenceSearchSample:
    """Generate one independent reference/search pair for the given seed."""
    kw = {**DEFAULT_PARAMS_KW, **params_kw}
    stem = f"pair_{seed:06d}"
    gds_path  = str(Path(tmp_dir) / f"{stem}.gds")
    json_path = str(Path(tmp_dir) / f"{stem}_meta.json")

    p = DRAMParams(**kw, seed=seed, output_gds=gds_path, output_json=json_path)
    gen = DRAMGenerator(p)
    meta = gen.generate()

    seed_seq = np.random.SeedSequence(seed)
    search_seed, ref_seed = seed_seq.spawn(2)
    rng_search = np.random.default_rng(search_seed)
    rng_ref = np.random.default_rng(ref_seed)

    # -- Search image: whole array at native resolution, independent rotation + noise --
    search_clean = render_sem_image(gds_path, meta, rng_search)
    search_angle = _draw_rotation_deg(rng_search)
    search_img, search_M = _rotate_image(search_clean, search_angle)

    true_center_px = rotate_point(tuple(meta["reference_center_px"]), search_M)

    # -- Reference image: same physical patch at zoom_ratio x finer pixel pitch --
    zoom_ratio = float(meta["zoom_ratio"])
    ref_w_px = int(round(meta["reference_image_size_px"][0] * zoom_ratio))
    ref_h_px = int(round(meta["reference_image_size_px"][1] * zoom_ratio))
    ppn_hi = meta["pixels_per_nm"] * zoom_ratio

    ref_clean = render_sem_patch(gds_path, meta["reference_bbox_nm"], ppn_hi,
                                  ref_w_px, ref_h_px, rng_ref,
                                  defect_locations=meta["defect_locations"])
    ref_angle = _draw_rotation_deg(rng_ref)
    # Rotating about the reference image's own center leaves its center point
    # fixed, so no ground-truth correction is needed for this rotation.
    reference_img, _ = _rotate_image(ref_clean, ref_angle)

    for f in (gds_path, json_path):
        Path(f).unlink(missing_ok=True)

    return ReferenceSearchSample(
        reference_img=reference_img,
        search_img=search_img,
        true_center_px=true_center_px,
        zoom_ratio=zoom_ratio,
        seed=seed,
    )
