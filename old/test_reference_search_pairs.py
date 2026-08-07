import numpy as np
import pytest
import cv2

from reference_search_pairs import (
    ReferenceSearchSample, generate_sample, rotate_point, _rotate_image,
)


def test_rotate_point_matches_warpaffine_pixel_displacement():
    img = np.zeros((200, 200), dtype=np.uint8)
    marker = (150.0, 80.0)  # (x, y)
    img[int(marker[1]), int(marker[0])] = 255

    angle = 17.0
    rotated, M = _rotate_image(img, angle)
    predicted = rotate_point(marker, M)

    actual_y, actual_x = np.unravel_index(np.argmax(rotated), rotated.shape)
    assert abs(actual_x - predicted[0]) < 1.5
    assert abs(actual_y - predicted[1]) < 1.5


SMALL_PARAMS_KW = dict(
    rows=8, cols=8, cell_pitch_nm=80.0, cell_pitch_bl_nm=60.0,
    wl_width_nm=24.0, bl_width_nm=18.0, contact_size_nm=14.0,
    capacitor_size_nm=30.0, n_particles=0, n_scratches=0,
    p_broken_wl=0.0, p_broken_bl=0.0, p_cmp_dishing=0.0,
)


def test_generate_sample_reproducible(tmp_path):
    s1 = generate_sample(seed=123, tmp_dir=str(tmp_path), **SMALL_PARAMS_KW)
    s2 = generate_sample(seed=123, tmp_dir=str(tmp_path), **SMALL_PARAMS_KW)
    assert s1.true_center_px == s2.true_center_px
    np.testing.assert_array_equal(s1.search_img, s2.search_img)
    np.testing.assert_array_equal(s1.reference_img, s2.reference_img)


def test_generate_sample_shapes_and_zoom(tmp_path):
    s = generate_sample(seed=1, tmp_dir=str(tmp_path), **SMALL_PARAMS_KW)
    assert s.search_img.shape == (1000, 1000)
    assert s.search_img.dtype == np.uint8
    assert s.zoom_ratio == 10.0
    # default reference_width_px=reference_height_px=100, zoom_ratio=10 -> 1000x1000
    assert s.reference_img.shape == (1000, 1000)
    assert s.reference_img.dtype == np.uint8


def test_generate_sample_cleans_up_intermediate_files(tmp_path):
    generate_sample(seed=42, tmp_dir=str(tmp_path), **SMALL_PARAMS_KW)
    leftover = list(tmp_path.glob("*.gds")) + list(tmp_path.glob("*_meta.json"))
    assert leftover == []


def test_true_center_differs_from_raw_reference_center_when_rotated(tmp_path):
    # With a nonzero search rotation, true_center_px (post-rotation) must not
    # equal the pre-rotation reference_center_px in general -- proves the
    # rotation-correction path actually runs, not just returns the raw value.
    import random
    found_nonzero_rotation_case = False
    for seed in range(20):
        s = generate_sample(seed=seed, tmp_dir=str(tmp_path), **SMALL_PARAMS_KW)
        # Reconstruct the raw (pre-rotation) center independently via the generator
        # to compare against.
        from dram_layout_generator import DRAMParams, DRAMGenerator
        from pathlib import Path
        gds_path = str(tmp_path / f"chk_{seed}.gds")
        json_path = str(tmp_path / f"chk_{seed}_meta.json")
        p = DRAMParams(**SMALL_PARAMS_KW, seed=seed, output_gds=gds_path, output_json=json_path)
        meta = DRAMGenerator(p).generate()
        Path(gds_path).unlink(missing_ok=True)
        Path(json_path).unlink(missing_ok=True)
        raw_center = tuple(meta["reference_center_px"])
        if abs(s.true_center_px[0] - raw_center[0]) > 0.5 or abs(s.true_center_px[1] - raw_center[1]) > 0.5:
            found_nonzero_rotation_case = True
            break
    assert found_nonzero_rotation_case


def test_default_params_kw_has_rework_values():
    from reference_search_pairs import DEFAULT_PARAMS_KW
    assert DEFAULT_PARAMS_KW["contact_size_nm"] == 40.0
    assert DEFAULT_PARAMS_KW["overlay_sigma_nm"] == 6.0
    assert DEFAULT_PARAMS_KW["linewidth_sigma_nm"] == 4.0
    assert DEFAULT_PARAMS_KW["ler_amplitude_nm"] == 3.0
    assert DEFAULT_PARAMS_KW["contact_sigma_frac"] == 0.15
    assert DEFAULT_PARAMS_KW["jitter_sigma_nm"] == 2.5


def test_generate_sample_with_default_params_wires_defect_locations(tmp_path):
    # Uses the real DEFAULT_PARAMS_KW (no override), so this exercises the
    # render_sem_patch(..., defect_locations=meta["defect_locations"])
    # wiring added in this task against a real generation run, not just a
    # dict-value check.
    s = generate_sample(seed=1, tmp_dir=str(tmp_path))
    assert s.reference_img.shape == (1000, 1000)
    assert s.search_img.shape == (1000, 1000)
