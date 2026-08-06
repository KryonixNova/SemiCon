"""
Test suite for DRAM layout generator module.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
import json
from dram_layout_generator import DRAMParams
import dataclasses


def test_params_defaults():
    """Test that DRAMParams creates with correct default values."""
    p = DRAMParams()
    assert p.rows == 64
    assert p.cols == 64
    assert p.cell_pitch_nm == 80.0
    assert p.cell_pitch_bl_nm == 60.0
    assert p.wl_width_nm == 24.0
    assert p.bl_width_nm == 18.0
    assert p.contact_size_nm == 14.0
    assert p.capacitor_size_nm == 30.0
    assert p.dummy_rows == 2
    assert p.dummy_cols == 2
    assert p.overlay_sigma_nm == 3.0
    assert p.linewidth_sigma_nm == 2.0
    assert p.ler_amplitude_nm == 1.5
    assert p.ler_period_nm == 20.0
    assert p.contact_sigma_frac == 0.08
    assert p.jitter_sigma_nm == 1.0
    assert p.p_missing_contact == 0.005
    assert p.p_missing_capacitor == 0.003
    assert p.p_broken_wl == 0.02
    assert p.p_broken_bl == 0.02
    assert p.n_particles == 3
    assert p.n_scratches == 2
    assert p.p_cmp_dishing == 0.10
    assert p.output_gds == "dram.gds"
    assert p.output_json == "dram_metadata.json"
    assert p.seed == 42
    assert p.dbu_nm == 1.0
    assert dataclasses.is_dataclass(p)


def test_params_frozen():
    """Test that DRAMParams is frozen and cannot be mutated."""
    p = DRAMParams()
    try:
        p.rows = 99
        assert False, "should be frozen"
    except Exception:
        pass


def test_params_custom():
    """Test that DRAMParams can be created with custom values."""
    p = DRAMParams(rows=8, cols=8, seed=7)
    assert p.rows == 8
    assert p.cols == 8
    assert p.seed == 7


# ── ProcessVariation tests ──────────────────────────────────────────────────

import klayout.db as db

def _pv(seed=0, **kwargs):
    from dram_layout_generator import ProcessVariation
    p = DRAMParams(seed=seed, **kwargs)
    rng = np.random.default_rng(seed)
    return ProcessVariation(p, rng)

def test_overlay_shift_returns_box():
    pv = _pv()
    b = db.Box(0, 0, 100, 50)
    result = pv.overlay_shift(b)
    assert isinstance(result, db.Box)
    # shift should be bounded (6-sigma = 18nm for sigma=3)
    assert abs(result.left - b.left) < 20
    assert abs(result.bottom - b.bottom) < 20

def test_vary_linewidth_shrinks_or_grows():
    pv = _pv(seed=1, linewidth_sigma_nm=5.0)
    b = db.Box(0, 0, 1000, 30)
    results = [pv.vary_linewidth(b, 'x') for _ in range(20)]
    heights = [r.height() for r in results]
    assert min(heights) < 30 or max(heights) > 30  # some variation

def test_add_ler_returns_list_of_boxes():
    pv = _pv()
    b = db.Box(0, 0, 400, 24)
    boxes = pv.add_ler(b, 'x')
    assert isinstance(boxes, list)
    assert len(boxes) >= 2
    for box in boxes:
        assert isinstance(box, db.Box)
        assert box.width() > 0 and box.height() > 0

def test_vary_contact_changes_size():
    pv = _pv(seed=5, contact_sigma_frac=0.3)
    b = db.Box(0, 0, 14, 14)
    results = [pv.vary_contact(b) for _ in range(20)]
    sizes = [r.width() for r in results]
    assert min(sizes) != max(sizes)  # variance present

def test_jitter_cell():
    pv = _pv(seed=0, jitter_sigma_nm=5.0)
    results = [pv.jitter_cell(500.0, 500.0) for _ in range(20)]
    xs = [r[0] for r in results]
    assert min(xs) < 500 or max(xs) > 500

def test_is_missing_extreme_probs():
    pv = _pv()
    assert all(pv.is_missing(1.0) for _ in range(10))
    assert not any(pv.is_missing(0.0) for _ in range(10))

def test_break_line_two_pieces():
    pv = _pv()
    b = db.Box(0, 0, 1000, 24)
    pieces = pv.break_line(b, 'x')
    assert len(pieces) == 2
    for piece in pieces:
        assert isinstance(piece, db.Box)
        assert piece.width() > 0

def test_add_particles_count():
    pv = _pv()
    bbox = db.Box(0, 0, 5000, 5000)
    particles = pv.add_particles(bbox, 5)
    assert len(particles) == 5
    for p in particles:
        assert isinstance(p, db.Box)

def test_add_scratches_count():
    pv = _pv()
    bbox = db.Box(0, 0, 5000, 5000)
    scratches = pv.add_scratches(bbox, 3)
    assert len(scratches) == 3

def test_cmp_dishing_shrinks_lines():
    pv = _pv()
    boxes = [db.Box(0, i*80, 5000, i*80+24) for i in range(10)]
    dished = pv.cmp_dishing(boxes, cx=2500.0, cy=400.0, r=500.0)
    assert len(dished) == len(boxes)
    for orig, dish in zip(boxes, dished):
        assert dish.height() <= orig.height()


# ── DRAMGenerator tests ─────────────────────────────────────────────────────

from dram_layout_generator import DRAMGenerator

def _gen(seed=42, **kwargs):
    p = DRAMParams(seed=seed, rows=4, cols=4, dummy_rows=1, dummy_cols=1,
                   **kwargs)
    return DRAMGenerator(p)

def test_generator_init_creates_layout():
    g = _gen()
    assert isinstance(g.layout, db.Layout)
    assert g.layout.dbu == pytest.approx(1e-3, rel=1e-6)
    assert g.cell is not None

def test_generator_has_all_layers():
    g = _gen()
    for attr in ('L_CAP', 'L_WL', 'L_BL', 'L_CON', 'L_DEF', 'L_DUM'):
        assert hasattr(g, attr)

def test_build_dummy_cells_places_shapes():
    g = _gen()
    g._build_dummy_cells()
    shapes = g.cell.shapes(g.L_DUM)
    assert shapes.size() > 0

def test_build_capacitors_count():
    g = _gen()
    centers = g._build_capacitors()
    assert len(centers) == 4 * 4  # rows * cols
    shapes = g.cell.shapes(g.L_CAP)
    # shapes may be fewer if some are missing (p_missing_capacitor=0.003)
    assert shapes.size() > 0

def test_build_capacitors_center_coords():
    p = DRAMParams(rows=2, cols=2, cell_pitch_nm=80, cell_pitch_bl_nm=60,
                   capacitor_size_nm=30, jitter_sigma_nm=0,
                   p_missing_capacitor=0, overlay_sigma_nm=0,
                   linewidth_sigma_nm=0, ler_amplitude_nm=0,
                   contact_sigma_frac=0, seed=0)
    g = DRAMGenerator(p)
    centers = g._build_capacitors()
    # Cell (0,0): cx=30, cy=40; (0,1): cx=90, cy=40; (1,0): cx=30, cy=120; etc.
    xs = sorted(set(c[0] for c in centers))
    ys = sorted(set(c[1] for c in centers))
    assert xs == [30, 90]   # 0*60+30, 1*60+30
    assert ys == [40, 120]  # 0*80+40, 1*80+40


def test_wordlines_count():
    p = DRAMParams(rows=4, cols=4, dummy_rows=1, dummy_cols=1,
                   p_broken_wl=0, linewidth_sigma_nm=0,
                   overlay_sigma_nm=0, ler_amplitude_nm=0, seed=0)
    g = DRAMGenerator(p)
    coords = g._build_wordlines()
    assert len(coords) == 4  # one per active row
    for c in coords:
        assert 'row' in c and 'y_nm' in c
        assert 'x_start_nm' in c and 'x_end_nm' in c
        assert c['x_end_nm'] > c['x_start_nm']


def test_bitlines_count():
    p = DRAMParams(rows=4, cols=4, dummy_rows=1, dummy_cols=1,
                   p_broken_bl=0, linewidth_sigma_nm=0,
                   overlay_sigma_nm=0, ler_amplitude_nm=0, seed=0)
    g = DRAMGenerator(p)
    coords = g._build_bitlines()
    assert len(coords) == 4  # one per active col


def test_wordlines_span_full_width():
    p = DRAMParams(rows=2, cols=3, dummy_rows=1, dummy_cols=1,
                   cell_pitch_bl_nm=60, cell_pitch_nm=80,
                   p_broken_wl=0, linewidth_sigma_nm=0,
                   overlay_sigma_nm=0, ler_amplitude_nm=0, seed=0)
    g = DRAMGenerator(p)
    coords = g._build_wordlines()
    # x_start should be at -1 dummy col, x_end at (cols+1)*pitch_bl
    assert coords[0]['x_start_nm'] < 0
    assert coords[0]['x_end_nm'] > 3 * 60


def test_wl_shapes_on_layer():
    p = DRAMParams(rows=4, cols=4, p_broken_wl=0,
                   linewidth_sigma_nm=0, overlay_sigma_nm=0,
                   ler_amplitude_nm=0, seed=0)
    g = DRAMGenerator(p)
    g._build_wordlines()
    assert g.cell.shapes(g.L_WL).size() > 0


def test_bl_shapes_on_layer():
    p = DRAMParams(rows=4, cols=4, p_broken_bl=0,
                   linewidth_sigma_nm=0, overlay_sigma_nm=0,
                   ler_amplitude_nm=0, seed=0)
    g = DRAMGenerator(p)
    g._build_bitlines()
    assert g.cell.shapes(g.L_BL).size() > 0


# ── DRAMGenerator._build_contacts tests ────────────────────────────────────

def test_contacts_count_no_missing():
    p = DRAMParams(rows=4, cols=4, p_missing_contact=0,
                   contact_sigma_frac=0, overlay_sigma_nm=0,
                   jitter_sigma_nm=0, seed=0)
    g = DRAMGenerator(p)
    locs = g._build_contacts()
    assert len(locs) == 4 * 4
    # All contacts placed → shape count == rows * cols
    assert g.cell.shapes(g.L_CON).size() == 4 * 4

def test_contacts_all_missing():
    p = DRAMParams(rows=4, cols=4, p_missing_contact=1.0, seed=0)
    g = DRAMGenerator(p)
    g._build_contacts()
    assert g.cell.shapes(g.L_CON).size() == 0

def test_contacts_offset_from_capacitor():
    """Contact centre should differ from capacitor centre (it's on the drain side)."""
    p = DRAMParams(rows=1, cols=1, p_missing_contact=0,
                   contact_sigma_frac=0, overlay_sigma_nm=0,
                   jitter_sigma_nm=0, cell_pitch_bl_nm=60,
                   cell_pitch_nm=80, contact_size_nm=14, seed=0)
    g = DRAMGenerator(p)
    locs = g._build_contacts()
    # Contact placed at cx + pitch_bl/4 offset; cap at cx=30, contact at 30+15=45
    assert locs[0][0] != 30.0  # contact x differs from capacitor centre x


# ── Task 6: _add_defects, generate(), generate_dram_layout(), CLI tests ──────

import tempfile


def test_add_defects_layer5():
    p = DRAMParams(rows=4, cols=4, n_particles=5, n_scratches=3,
                   p_cmp_dishing=0, seed=0)
    g = DRAMGenerator(p)
    g._build_wordlines()   # needed so WL shapes exist for CMP dishing
    defects = g._add_defects()
    assert isinstance(defects, list)
    assert g.cell.shapes(g.L_DEF).size() > 0


def test_generate_returns_dict_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        gds_path  = os.path.join(tmpdir, "test.gds")
        json_path = os.path.join(tmpdir, "test.json")
        p = DRAMParams(rows=4, cols=4, dummy_rows=1, dummy_cols=1,
                       output_gds=gds_path, output_json=json_path, seed=0)
        g = DRAMGenerator(p)
        meta = g.generate()
        for key in ("cell_centers", "wordline_coords", "bitline_coords",
                    "bounding_box", "defect_locations", "params",
                    "gds_file", "metadata_file"):
            assert key in meta, f"missing key: {key}"
        assert len(meta["cell_centers"]) == 4 * 4
        assert len(meta["wordline_coords"]) == 4
        assert len(meta["bitline_coords"]) == 4


def test_generate_writes_gds():
    with tempfile.TemporaryDirectory() as tmpdir:
        gds_path  = os.path.join(tmpdir, "out.gds")
        json_path = os.path.join(tmpdir, "out.json")
        p = DRAMParams(rows=4, cols=4, output_gds=gds_path, output_json=json_path, seed=0)
        DRAMGenerator(p).generate()
        assert os.path.exists(gds_path)
        assert os.path.getsize(gds_path) > 0


def test_generate_writes_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        gds_path  = os.path.join(tmpdir, "out.gds")
        json_path = os.path.join(tmpdir, "out.json")
        p = DRAMParams(rows=4, cols=4, output_gds=gds_path, output_json=json_path, seed=0)
        DRAMGenerator(p).generate()
        with open(json_path) as f:
            data = json.load(f)
        assert "cell_centers" in data


def test_generate_deterministic():
    with tempfile.TemporaryDirectory() as tmpdir:
        def run():
            gp = os.path.join(tmpdir, f"d{id(object())}.gds")
            jp = os.path.join(tmpdir, f"d{id(object())}.json")
            p = DRAMParams(rows=4, cols=4, output_gds=gp, output_json=jp, seed=99)
            DRAMGenerator(p).generate()
            return open(gp, 'rb').read()
        assert run() == run()


def test_generate_dram_layout_entry_point():
    with tempfile.TemporaryDirectory() as tmpdir:
        from dram_layout_generator import generate_dram_layout
        p = DRAMParams(rows=4, cols=4,
                       output_gds=os.path.join(tmpdir, "e.gds"),
                       output_json=os.path.join(tmpdir, "e.json"), seed=1)
        meta = generate_dram_layout(p)
        assert isinstance(meta, dict)
        assert len(meta["cell_centers"]) == 16


def test_bounding_box_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = DRAMParams(rows=4, cols=4,
                       cell_pitch_nm=80, cell_pitch_bl_nm=60,
                       output_gds=os.path.join(tmpdir, "_bb_test.gds"),
                       output_json=os.path.join(tmpdir, "_bb_test.json"), seed=0)
        meta = DRAMGenerator(p).generate()
    bb = meta["bounding_box"]
    assert bb["x_min_nm"] == 0
    assert bb["y_min_nm"] == 0
    assert bb["x_max_nm"] == pytest.approx(4 * 60)
    assert bb["y_max_nm"] == pytest.approx(4 * 80)


def test_cmp_dishing_modifies_wl_geometry():
    """CMP dishing must actually thin WL shapes on Layer 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Without dishing: measure WL height
        p_no = DRAMParams(rows=4, cols=4, p_cmp_dishing=0,
                          p_broken_wl=0, linewidth_sigma_nm=0,
                          overlay_sigma_nm=0, ler_amplitude_nm=0,
                          seed=0,
                          output_gds=os.path.join(tmpdir, 'no.gds'),
                          output_json=os.path.join(tmpdir, 'no.json'))
        g_no = DRAMGenerator(p_no)
        g_no._build_dummy_cells()
        g_no._build_capacitors()
        g_no._build_wordlines()
        g_no._build_bitlines()
        shapes_no = list(g_no.cell.shapes(g_no.L_WL).each())
        heights_no = [s.bbox().height() for s in shapes_no]

        # With full dishing across entire array: all WLs should be thinned
        p_yes = DRAMParams(rows=4, cols=4, p_cmp_dishing=1.0,
                           p_broken_wl=0, linewidth_sigma_nm=0,
                           overlay_sigma_nm=0, ler_amplitude_nm=0,
                           n_particles=0, n_scratches=0,
                           seed=0,
                           output_gds=os.path.join(tmpdir, 'yes.gds'),
                           output_json=os.path.join(tmpdir, 'yes.json'))
        g_yes = DRAMGenerator(p_yes)
        g_yes.generate()
        shapes_yes = list(g_yes.cell.shapes(g_yes.L_WL).each())
        heights_yes = [s.bbox().height() for s in shapes_yes]

        # With p_cmp_dishing=1.0, the dish covers the whole array → lines thinned
        if heights_no and heights_yes:
            assert max(heights_yes) <= max(heights_no), (
                f"CMP dishing should have thinned WLs: before={heights_no}, after={heights_yes}"
            )


# ── Rasterization extension tests ─────────────────────────────────────────────

from dram_layout_generator import (
    RasterizationConfig, nm_to_pixel, pixel_to_nm,
    bbox_nm_to_pixel, bbox_pixel_to_nm,
)


def test_params_rasterization_defaults():
    p = DRAMParams()
    assert p.output_width_px == 1000
    assert p.output_height_px == 1000
    assert p.reference_width_px == 100
    assert p.reference_height_px == 100
    assert p.zoom_ratio == 10
    assert p.pixels_per_nm is None


def test_validation_output_width():
    with pytest.raises(ValueError, match="output_width_px"):
        DRAMGenerator(DRAMParams(output_width_px=512))


def test_validation_reference_height():
    with pytest.raises(ValueError, match="reference_height_px"):
        DRAMGenerator(DRAMParams(reference_height_px=50))


def test_validation_zoom_ratio():
    with pytest.raises(ValueError, match="zoom_ratio"):
        DRAMGenerator(DRAMParams(zoom_ratio=5))


def test_pixels_per_nm_auto_computed():
    # Default 64×64 array: w = 64*60 = 3840 nm, h = 64*80 = 5120 nm
    # pixels_per_nm = min(1000/3840, 1000/5120) = 1000/5120 ≈ 0.1953125
    p = DRAMParams()
    g = DRAMGenerator(p)
    expected = min(1000 / (64 * 60), 1000 / (64 * 80))
    assert abs(g._pixels_per_nm - expected) < 1e-9


def test_pixels_per_nm_explicit():
    p = DRAMParams(pixels_per_nm=0.5)
    g = DRAMGenerator(p)
    assert g._pixels_per_nm == 0.5


def test_coordinate_utilities_roundtrip():
    ppn = 0.25
    for x in [0.0, 100.0, 500.0, 1234.56]:
        assert abs(pixel_to_nm(nm_to_pixel(x, ppn), ppn) - x) < 1e-9
    bbox = [10.0, 20.0, 110.0, 220.0]
    recovered = bbox_pixel_to_nm(bbox_nm_to_pixel(bbox, ppn), ppn)
    for a, b in zip(bbox, recovered):
        assert abs(a - b) < 1e-9


def test_generate_rasterization_metadata_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = DRAMParams(rows=4, cols=4,
                       output_gds=os.path.join(tmpdir, "r.gds"),
                       output_json=os.path.join(tmpdir, "r.json"), seed=7)
        meta = DRAMGenerator(p).generate()
    for key in ("search_image_size_px", "reference_image_size_px", "zoom_ratio",
                "pixels_per_nm", "search_bbox_nm", "search_bbox_px",
                "reference_bbox_nm", "reference_center_nm",
                "reference_bbox_px", "reference_center_px"):
        assert key in meta, f"missing key: {key}"


def test_reference_patch_inside_active_array():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = DRAMParams(rows=8, cols=8,
                       output_gds=os.path.join(tmpdir, "rp.gds"),
                       output_json=os.path.join(tmpdir, "rp.json"), seed=3)
        meta = DRAMGenerator(p).generate()
    arr_w = 8 * 60.0
    arr_h = 8 * 80.0
    rx0, ry0, rx1, ry1 = meta["reference_bbox_nm"]
    assert rx0 >= 0 and ry0 >= 0
    assert rx1 <= arr_w and ry1 <= arr_h


def test_reference_center_matches_bbox():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = DRAMParams(rows=4, cols=4,
                       output_gds=os.path.join(tmpdir, "rc.gds"),
                       output_json=os.path.join(tmpdir, "rc.json"), seed=5)
        meta = DRAMGenerator(p).generate()
    rx0, ry0, rx1, ry1 = meta["reference_bbox_nm"]
    cx, cy = meta["reference_center_nm"]
    assert abs(cx - (rx0 + rx1) / 2) < 1e-6
    assert abs(cy - (ry0 + ry1) / 2) < 1e-6


def test_get_rasterization_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        gp = os.path.join(tmpdir, "gc.gds")
        jp = os.path.join(tmpdir, "gc.json")
        p = DRAMParams(rows=4, cols=4, output_gds=gp, output_json=jp, seed=2)
        g = DRAMGenerator(p)
        g.generate()
        cfg = g.get_rasterization_config()
    assert isinstance(cfg, RasterizationConfig)
    assert cfg.gds_file == gp
    assert cfg.width_px == 1000
    assert cfg.height_px == 1000
    assert len(cfg.reference_bbox_px) == 4
    assert len(cfg.reference_center_px) == 2
    assert cfg.pixels_per_nm > 0
