"""Tests for dram_localization_pipeline.py's multi-region die layout feature."""
import dataclasses

import klayout.db as db
import numpy as np
import pytest

import dram_localization_pipeline as m


def test_dram_params_die_defaults_are_passthrough():
    p = m.DRAMParams()
    assert p.die_block_rows == 1
    assert p.die_block_cols == 1
    assert p.die_block_width_nm == 3840.0
    assert p.die_block_height_nm == 5120.0
    assert p.die_street_width_nm == 40.0
    assert p.die_block_variation_frac == 0.4


def test_layer_street_constant():
    assert m.LAYER_STREET == (7, 0)


def test_randomize_block_params_zero_frac_is_identity():
    base = m.DRAMParams()
    rng = np.random.default_rng(1)
    out = m._randomize_block_params(base, rng, 0.0)
    assert out == base


def test_randomize_block_params_bounds():
    base = m.DRAMParams()
    rng = np.random.default_rng(7)
    frac = 0.4
    out = m._randomize_block_params(base, rng, frac)

    geometry_fields = [
        "cell_pitch_nm", "cell_pitch_bl_nm", "wl_width_nm", "bl_width_nm",
        "contact_size_nm", "capacitor_size_nm", "overlay_sigma_nm",
        "linewidth_sigma_nm", "ler_amplitude_nm", "ler_period_nm",
        "contact_sigma_frac", "jitter_sigma_nm",
    ]
    for field in geometry_fields:
        base_val = getattr(base, field)
        out_val = getattr(out, field)
        assert base_val * (1 - frac) - 1e-9 <= out_val <= base_val * (1 + frac) + 1e-9, field

    prob_fields = ["p_missing_contact", "p_missing_capacitor", "p_broken_wl",
                   "p_broken_bl", "p_cmp_dishing"]
    for field in prob_fields:
        out_val = getattr(out, field)
        assert 0.0 <= out_val <= 1.0, field

    for field in ["n_particles", "n_scratches"]:
        out_val = getattr(out, field)
        assert out_val >= 0, field

    # rows/cols/dummy/output/seed/path fields are untouched by this function
    assert out.rows == base.rows
    assert out.cols == base.cols
    assert out.dummy_rows == base.dummy_rows
    assert out.output_width_px == base.output_width_px
    assert out.seed == base.seed


def test_randomize_block_params_deterministic_for_fixed_seed():
    base = m.DRAMParams()
    out_a = m._randomize_block_params(base, np.random.default_rng(42), 0.4)
    out_b = m._randomize_block_params(base, np.random.default_rng(42), 0.4)
    assert out_a == out_b


def _shapes_by_layer(gds_path):
    """Decode a GDS file's box shapes per layer, sorted, for content
    comparison. NOT raw-byte comparison: KLayout's GDS writer embeds a
    wall-clock last-modified timestamp in the file header, so two separate
    write() calls with identical geometry never produce identical bytes --
    confirmed empirically before writing this plan. Comparing decoded shape
    content is what "identical output" actually means here."""
    layout = db.Layout()
    layout.read(gds_path)
    top = layout.top_cell()
    result = {}
    for ln, dt in [m.LAYER_CAPACITOR, m.LAYER_WORDLINE, m.LAYER_BITLINE,
                   m.LAYER_CONTACT, m.LAYER_DEFECT, m.LAYER_DUMMY]:
        li = layout.find_layer(ln, dt)
        boxes = []
        if li is not None:
            for shape in top.shapes(li).each():
                b = shape.bbox()
                boxes.append((b.left, b.bottom, b.right, b.top))
        result[(ln, dt)] = sorted(boxes)
    return result


def test_die_generator_1x1_is_exact_passthrough(tmp_path):
    gds_a = str(tmp_path / "a.gds")
    json_a = str(tmp_path / "a.json")
    gds_b = str(tmp_path / "b.gds")
    json_b = str(tmp_path / "b.json")

    p_direct = m.DRAMParams(rows=8, cols=8, seed=5, output_gds=gds_a, output_json=json_a)
    meta_direct = m.DRAMGenerator(p_direct).generate()

    p_die = m.DRAMParams(rows=8, cols=8, seed=5, output_gds=gds_b, output_json=json_b)
    meta_die = m.DieGenerator(p_die).generate()

    assert _shapes_by_layer(gds_a) == _shapes_by_layer(gds_b)

    # "params"/"gds_file"/"metadata_file" legitimately differ: gds_a/gds_b
    # use different tmp paths so the GDS content can be compared separately
    # above without one write clobbering the other. Every other key must
    # match exactly.
    path_dependent_keys = {"params", "gds_file", "metadata_file"}
    for key in meta_direct:
        if key in path_dependent_keys:
            continue
        assert meta_die[key] == meta_direct[key], key
    assert meta_die["blocks"] == [{
        "block_row": 0, "block_col": 0,
        "bbox_nm": [meta_direct["bounding_box"]["x_min_nm"],
                    meta_direct["bounding_box"]["y_min_nm"],
                    meta_direct["bounding_box"]["x_max_nm"],
                    meta_direct["bounding_box"]["y_max_nm"]],
        "params": dataclasses.asdict(p_die),
    }]


def test_die_generator_2x2_block_bboxes_dont_overlap_and_are_separated_by_street(tmp_path):
    p = m.DRAMParams(
        die_block_rows=2, die_block_cols=2,
        die_block_width_nm=800.0, die_block_height_nm=800.0,
        die_street_width_nm=30.0,
        output_gds=str(tmp_path / "die.gds"),
        output_json=str(tmp_path / "die.json"),
        seed=3,
    )
    meta = m.DieGenerator(p).generate()

    blocks = {(b["block_row"], b["block_col"]): b["bbox_nm"] for b in meta["blocks"]}
    assert set(blocks.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    x0_00, y0_00, x1_00, y1_00 = blocks[(0, 0)]
    x0_01, y0_01, x1_01, y1_01 = blocks[(0, 1)]
    x0_10, y0_10, x1_10, y1_10 = blocks[(1, 0)]

    assert x1_00 <= x0_01  # (0,0) doesn't overlap (0,1) horizontally
    assert abs((x0_01 - x1_00) - p.die_street_width_nm) < 1e-6
    assert y1_00 <= y0_10  # (0,0) doesn't overlap (1,0) vertically
    assert abs((y0_10 - y1_00) - p.die_street_width_nm) < 1e-6

    for (r, c), (x0, y0, x1, y1) in blocks.items():
        assert x1 - x0 == pytest.approx(p.die_block_width_nm)
        assert y1 - y0 == pytest.approx(p.die_block_height_nm)

    total_w = 2 * p.die_block_width_nm + p.die_street_width_nm
    total_h = 2 * p.die_block_height_nm + p.die_street_width_nm
    assert meta["search_bbox_nm"] == [0.0, 0.0, total_w, total_h]


def test_die_generator_2x2_defect_locations_stay_within_die_bounds(tmp_path):
    p = m.DRAMParams(
        die_block_rows=2, die_block_cols=2,
        die_block_width_nm=800.0, die_block_height_nm=800.0,
        die_street_width_nm=30.0,
        output_gds=str(tmp_path / "die.gds"),
        output_json=str(tmp_path / "die.json"),
        seed=9,
    )
    meta = m.DieGenerator(p).generate()
    x0, y0, x1, y1 = meta["search_bbox_nm"]
    # Dummy-cell fringe extends dummy_rows/cols * cell_pitch beyond a block's
    # nominal footprint, and defects are placed within that full fringe (by
    # design, same as the single-array generator). With the default
    # dummy_rows=dummy_cols=2 and die_block_variation_frac=0.4, pitch can be
    # jittered up to 1.4x base (80nm/60nm), so the fringe can reach roughly
    # 2 * 80 * 1.4 = 224nm (y) / 2 * 60 * 1.4 = 168nm (x). Use a comfortable
    # margin above that worst case.
    margin = 300
    for d in meta["defect_locations"]:
        bx0, by0, bx1, by1 = d["bbox_nm"]
        assert x0 - margin <= bx0 and bx1 <= x1 + margin
        assert y0 - margin <= by0 and by1 <= y1 + margin


def test_street_layer_renders_distinct_intensity(tmp_path):
    p = m.DRAMParams(
        die_block_rows=1, die_block_cols=2,
        die_block_width_nm=800.0, die_block_height_nm=800.0,
        die_street_width_nm=60.0,
        output_gds=str(tmp_path / "die.gds"),
        output_json=str(tmp_path / "die.json"),
        seed=11,
    )
    meta = m.DieGenerator(p).generate()
    rng = np.random.default_rng(0)
    img = m.render_sem_image(p.output_gds, meta, rng)

    # Sample a column in the middle of the street gap between the two blocks,
    # away from the top/bottom dummy-cell margin, and confirm its intensity
    # sits at the configured street value rather than background or lattice.
    street_x_nm = p.die_block_width_nm + p.die_street_width_nm / 2
    col_px = int(round(street_x_nm * meta["pixels_per_nm"]))
    row_px = img.shape[0] // 2
    sample = img[row_px - 3:row_px + 3, col_px - 3:col_px + 3]
    assert abs(int(sample.mean()) - m.SEM_INTENSITY[m.LAYER_STREET]) < 25


def test_generate_sample_uses_die_generator(tmp_path, monkeypatch):
    calls = []
    real_die_generator = m.DieGenerator

    class SpyDieGenerator(real_die_generator):
        def __init__(self, params):
            calls.append(params)
            super().__init__(params)

    monkeypatch.setattr(m, "DieGenerator", SpyDieGenerator)
    m.generate_sample(seed=123, tmp_dir=str(tmp_path),
                       die_block_rows=2, die_block_cols=2,
                       die_block_width_nm=800.0, die_block_height_nm=800.0)
    assert len(calls) == 1
    assert calls[0].die_block_rows == 2
    assert calls[0].die_block_cols == 2


def test_generate_sample_default_single_block_unchanged(tmp_path):
    # die_block_rows=die_block_cols=1 (no override) must still work exactly
    # as it did before this feature existed.
    sample = m.generate_sample(seed=1, tmp_dir=str(tmp_path))
    assert sample.search_img.shape == (1000, 1000)
    assert sample.reference_img.shape[:2] == (1000, 1000)  # 100px * zoom_ratio(10)
    assert 0 <= sample.true_center_px[0] <= 1000
    assert 0 <= sample.true_center_px[1] <= 1000


def test_place_training_defect_inserts_square_box_of_expected_size():
    layout = db.Layout()
    layout.dbu = 0.001
    cell = layout.create_cell("T")
    layer = layout.layer(5, 0)
    rng = np.random.default_rng(0)
    bbox = db.Box(0, 0, 10000, 10000)

    defect = m._place_training_defect(cell, layer, pixels_per_nm=0.2, rng=rng,
                                       bbox=bbox, size_px=10)

    assert defect["type"] == "training_scratch"
    x0, y0, x1, y1 = defect["bbox_nm"]
    side_nm = 10 / 0.2  # 50nm
    assert (x1 - x0) == pytest.approx(side_nm, abs=1)
    assert (y1 - y0) == pytest.approx(side_nm, abs=1)
    assert bbox.left <= x0 and x1 <= bbox.right
    assert bbox.bottom <= y0 and y1 <= bbox.top

    shapes = list(cell.shapes(layer).each())
    assert len(shapes) == 1


def test_dram_generator_single_training_defect_replaces_particles_and_scratches(tmp_path):
    p = m.DRAMParams(
        n_particles=3, n_scratches=2,
        single_training_defect=True,
        output_gds=str(tmp_path / "d.gds"),
        output_json=str(tmp_path / "d.json"),
        seed=5,
    )
    meta = m.DRAMGenerator(p).generate()
    types = [d["type"] for d in meta["defect_locations"]]
    assert types.count("particle") == 0
    assert types.count("scratch") == 0
    assert types.count("training_scratch") == 1
    # Only the training_scratch should be present; all other defect types should be suppressed
    assert len(meta["defect_locations"]) == 1


def test_die_generator_2x2_single_training_defect_places_exactly_one(tmp_path):
    p = m.DRAMParams(
        die_block_rows=2, die_block_cols=2,
        die_block_width_nm=800.0, die_block_height_nm=800.0,
        die_street_width_nm=30.0,
        single_training_defect=True,
        n_particles=3, n_scratches=2,
        output_gds=str(tmp_path / "die.gds"),
        output_json=str(tmp_path / "die.json"),
        seed=17,
    )
    meta = m.DieGenerator(p).generate()
    types = [d["type"] for d in meta["defect_locations"]]
    assert types.count("training_scratch") == 1
    assert types.count("particle") == 0
    assert types.count("scratch") == 0
    # Exactly one defect total across all 4 blocks -- no CMP dishing,
    # missing capacitors/contacts, or broken lines snuck in from any block
    # (default probabilities for those are all nonzero, so this is a real
    # regression guard, not a vacuous one).
    assert len(meta["defect_locations"]) == 1


def test_paint_defects_training_scratch_draws_diagonal_line():
    ss_h, ss_w = 200, 200
    intensity = np.full((ss_h, ss_w), 150.0, dtype=np.float32)
    bbox_nm = [0.0, 0.0, 100.0, 100.0]
    defect = {"type": "training_scratch", "bbox_nm": bbox_nm}
    ppn_ss = 2.0  # 100nm * 2px/nm = 200px, matches the canvas exactly

    m._paint_defects(intensity, [defect], bbox_nm, ppn_ss)

    # Points along the top-left -> bottom-right diagonal are darkened.
    diag_vals = [intensity[i, i] for i in (40, 100, 160)]
    assert all(v < 150.0 for v in diag_vals)
    # A point well off the diagonal (near the bottom-left corner) is untouched.
    assert intensity[180, 20] == pytest.approx(150.0)


def test_generate_sample_single_training_defect_populates_tile_and_coords(tmp_path):
    sample = m.generate_sample(seed=55, tmp_dir=str(tmp_path),
                                single_training_defect=True)
    assert sample.training_defect_center_px is not None
    assert sample.training_defect_bbox_px is not None
    row, col = sample.training_defect_tile
    assert 0 <= row <= 9
    assert 0 <= col <= 9
    cx, cy = sample.training_defect_center_px
    assert 0 <= cx <= 1000
    assert 0 <= cy <= 1000


def test_generate_sample_default_has_no_training_defect(tmp_path):
    sample = m.generate_sample(seed=1, tmp_dir=str(tmp_path))
    assert sample.training_defect_center_px is None
    assert sample.training_defect_bbox_px is None
    assert sample.training_defect_tile is None


def test_generate_sample_single_training_defect_multiblock_die_in_frame(tmp_path):
    sample = m.generate_sample(seed=99, tmp_dir=str(tmp_path),
                                single_training_defect=True,
                                die_block_rows=1, die_block_cols=3,
                                die_block_width_nm=800.0, die_block_height_nm=800.0)
    assert sample.training_defect_center_px is not None
    cx, cy = sample.training_defect_center_px
    assert 0 <= cx <= 1000
    assert 0 <= cy <= 1000
    row, col = sample.training_defect_tile
    assert 0 <= row <= 9
    assert 0 <= col <= 9


def test_generate_sample_single_training_defect_stays_in_frame_across_seeds(tmp_path):
    for seed in range(25):
        sample = m.generate_sample(seed=seed, tmp_dir=str(tmp_path),
                                    single_training_defect=True)
        cx, cy = sample.training_defect_center_px
        assert 0 <= cx <= 1000, f"seed {seed}: cx={cx} out of frame"
        assert 0 <= cy <= 1000, f"seed {seed}: cy={cy} out of frame"


def test_generate_sample_single_training_defect_reaches_full_tile_range(tmp_path):
    cols_seen = set()
    for seed in range(40):
        sample = m.generate_sample(seed=seed, tmp_dir=str(tmp_path),
                                    single_training_defect=True)
        _, col = sample.training_defect_tile
        cols_seen.add(col)
    # Before the canvas-bbox fix, placement was bounded by the array's own
    # footprint, which (at the default aspect ratio) made tile columns 8-9
    # structurally unreachable. Confirm the fix actually reaches them.
    assert cols_seen & {8, 9}, f"tile columns 8-9 never reached across 40 seeds: {cols_seen}"
