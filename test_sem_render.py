import hashlib
import numpy as np
import pytest

from dram_layout_generator import DRAMParams, DRAMGenerator


SMALL_PARAMS_KW = dict(
    rows=8, cols=8, cell_pitch_nm=80.0, cell_pitch_bl_nm=60.0,
    wl_width_nm=24.0, bl_width_nm=18.0, contact_size_nm=14.0,
    capacitor_size_nm=30.0, n_particles=1, n_scratches=1,
    p_broken_wl=0.02, p_broken_bl=0.02, p_cmp_dishing=0.10,
)


def test_render_sem_image_output_unchanged_by_extraction(tmp_path):
    from sem_render import render_sem_image

    gds_path = str(tmp_path / "t.gds")
    json_path = str(tmp_path / "t.json")
    p = DRAMParams(**SMALL_PARAMS_KW, seed=99, output_gds=gds_path, output_json=json_path)
    meta = DRAMGenerator(p).generate()

    rng = np.random.default_rng(99)
    img = render_sem_image(gds_path, meta, rng)

    assert img.shape == (1000, 1000)
    assert img.dtype == np.uint8
    # Known-good digest captured after the rendering-rework (Task 1: circle contacts
    # and two-tone defect rendering). Regenerate this digest only if render_sem_image's
    # actual output is intentionally changed, never to make a broken refactor pass.
    digest = hashlib.sha256(img.tobytes()).hexdigest()
    assert digest == "6ba3acd09c0a68a9bfdaebbff23a48adf88501ae34c04169581ece3d15ed60ad"


def test_render_sem_patch_fills_exact_canvas_size(tmp_path):
    from sem_render import render_sem_patch

    gds_path = str(tmp_path / "t.gds")
    json_path = str(tmp_path / "t.json")
    p = DRAMParams(**SMALL_PARAMS_KW, seed=5, output_gds=gds_path, output_json=json_path)
    meta = DRAMGenerator(p).generate()

    zoom_ratio = float(meta["zoom_ratio"])
    ppn_hi = meta["pixels_per_nm"] * zoom_ratio
    w_px = int(round(meta["reference_image_size_px"][0] * zoom_ratio))
    h_px = int(round(meta["reference_image_size_px"][1] * zoom_ratio))

    rng = np.random.default_rng(5)
    img = render_sem_patch(gds_path, meta["reference_bbox_nm"], ppn_hi, w_px, h_px, rng)

    assert img.shape == (h_px, w_px)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255


def test_contact_layer_renders_as_circle_not_rectangle(tmp_path):
    from sem_render import _supersampled_masks_for_bbox, CIRCLE_LAYERS
    import klayout.db as db

    # Minimal synthetic GDS with one square contact shape, bypassing
    # DRAMGenerator entirely -- an isolated check of the rendering
    # primitive, not the whole generator pipeline.
    gds_path = str(tmp_path / "contact.gds")
    layout = db.Layout()
    layout.dbu = 0.001  # 1 DBU = 1 nm
    top = layout.create_cell("TOP")
    li = layout.layer(4, 0)
    top.shapes(li).insert(db.Box(80, 80, 120, 120))  # 40nm x 40nm square
    layout.write(gds_path)

    ppn_ss = 4.0
    masks = _supersampled_masks_for_bbox(gds_path, [0.0, 0.0, 200.0, 200.0],
                                          ppn_ss, [(4, 0)], 800, 800)
    mask = masks[(4, 0)]
    filled_px = int((mask > 0.5).sum())

    bbox_side_px = 40 * ppn_ss
    rect_area = bbox_side_px * bbox_side_px
    circle_area = np.pi * (bbox_side_px / 2) ** 2

    assert (4, 0) in CIRCLE_LAYERS
    assert abs(filled_px - circle_area) < 0.1 * circle_area
    assert filled_px < 0.85 * rect_area


def test_paint_defects_produces_halo_and_mark_bands():
    from sem_render import (
        _paint_defects, SEM_BG, DEFECT_HALO_INTENSITY, DEFECT_MARK_INTENSITY,
    )

    ppn_ss = 4.0
    origin_bbox_nm = [0.0, 0.0, 200.0, 200.0]
    ss_size = int(200 * ppn_ss)
    intensity = np.full((ss_size, ss_size), SEM_BG, dtype=np.float32)

    defect = {"type": "scratch", "bbox_nm": [90.0, 90.0, 110.0, 110.0]}
    _paint_defects(intensity, [defect], origin_bbox_nm, ppn_ss)

    assert intensity[10, 10] == SEM_BG              # untouched background
    assert intensity[320, 320] == DEFECT_HALO_INTENSITY  # halo band
    assert intensity[400, 400] == DEFECT_MARK_INTENSITY  # mark band


def test_paint_defects_cmp_dishing_uses_soft_tint():
    from sem_render import _paint_defects, SEM_BG, DISHING_INTENSITY

    ppn_ss = 4.0
    origin_bbox_nm = [0.0, 0.0, 200.0, 200.0]
    ss_size = int(200 * ppn_ss)
    intensity = np.full((ss_size, ss_size), SEM_BG, dtype=np.float32)

    defect = {"type": "cmp_dishing", "bbox_nm": [90.0, 90.0, 110.0, 110.0]}
    _paint_defects(intensity, [defect], origin_bbox_nm, ppn_ss)

    assert intensity[400, 400] == 0.5 * SEM_BG + 0.5 * DISHING_INTENSITY
    assert intensity[10, 10] == SEM_BG


def test_render_sem_patch_defect_locations_changes_output(tmp_path):
    from dram_layout_generator import DRAMParams, DRAMGenerator
    from sem_render import render_sem_patch

    gds_path = str(tmp_path / "t.gds")
    json_path = str(tmp_path / "t.json")
    p = DRAMParams(rows=8, cols=8, cell_pitch_nm=80.0, cell_pitch_bl_nm=60.0,
                   wl_width_nm=24.0, bl_width_nm=18.0, contact_size_nm=40.0,
                   capacitor_size_nm=30.0, n_particles=3, n_scratches=2,
                   p_broken_wl=0.0, p_broken_bl=0.0, p_cmp_dishing=0.0,
                   seed=7, output_gds=gds_path, output_json=json_path)
    meta = DRAMGenerator(p).generate()

    # Use the whole array as the "patch" bbox so any particle/scratch defect
    # (there are 3+2 of them at these params) is guaranteed to be inside it.
    bbox_nm = meta["search_bbox_nm"]
    ppn = meta["pixels_per_nm"]

    rng_a = np.random.default_rng(7)
    img_with = render_sem_patch(gds_path, bbox_nm, ppn, 400, 400, rng_a,
                                 defect_locations=meta["defect_locations"])

    rng_b = np.random.default_rng(7)
    img_without = render_sem_patch(gds_path, bbox_nm, ppn, 400, 400, rng_b,
                                    defect_locations=None)

    assert not np.array_equal(img_with, img_without)
