import numpy as np

from src.patterns.mxn import fixed_count_spans, generate_mxn_canvas


def test_fixed_count_spans_single_mat_fills_canvas():
    spans = fixed_count_spans(1000, 1, 100.0)
    assert spans == [(True, 0, 1000)]


def test_fixed_count_spans_alternates_and_covers():
    spans = fixed_count_spans(1000, 3, 100.0)
    assert [is_mat for is_mat, _, _ in spans] == [True, False, True, False, True]
    assert spans[0][1] == 0 and spans[-1][2] == 1000
    for (_, _, end), (_, start, _) in zip(spans, spans[1:]):
        assert end == start                      # no gaps, no overlap


def test_generate_mxn_canvas_shape_and_region_count():
    rng = np.random.default_rng(0)
    out = generate_mxn_canvas(1000, 2, 3, 120.0, rng)
    assert out["canvas"].shape == (1000, 1000)
    assert out["canvas"].dtype == np.uint8
    assert len(out["mat_rects"]) == 2 * 3
    assert len(out["mat_feature_sizes_nm"]) == 2 * 3
    assert out["strip_rects"]                     # separators exist


def test_generate_mxn_canvas_restricts_to_given_preset_names():
    from src.presets import get_preset

    allowed = {"dram_loose", "dram_wide", "dram_legacy"}
    allowed_feature_sizes = {get_preset(n)["feature_size_nm"] for n in allowed}

    rng = np.random.default_rng(0)
    out = generate_mxn_canvas(1000, 3, 3, 120.0, rng, preset_names=sorted(allowed))
    assert set(out["mat_feature_sizes_nm"]) <= allowed_feature_sizes


def test_generate_mxn_canvas_preset_names_none_is_a_noop():
    a = generate_mxn_canvas(500, 2, 2, 120.0, np.random.default_rng(3))["canvas"]
    b = generate_mxn_canvas(500, 2, 2, 120.0, np.random.default_rng(3), preset_names=None)["canvas"]
    assert np.array_equal(a, b)


def test_generate_mxn_canvas_is_deterministic_for_a_seed():
    a = generate_mxn_canvas(500, 2, 2, 120.0, np.random.default_rng(7))["canvas"]
    b = generate_mxn_canvas(500, 2, 2, 120.0, np.random.default_rng(7))["canvas"]
    assert np.array_equal(a, b)


def test_jitter_globals_are_read_at_call_time():
    """Ablation A1 patches these globals; late-bound defaults would break it."""
    from src.patterns import dram as dram_mod

    saved = (dram_mod.POSITION_JITTER_NM, dram_mod.WIDTH_JITTER_FRACTION)
    try:
        dram_mod.POSITION_JITTER_NM = 0.0
        dram_mod.WIDTH_JITTER_FRACTION = 0.0
        a = dram_mod.generate_dram_canvas(
            400, {"kind": "dram", "feature_size_nm": 32, "word_line_pitch_nm": 64,
                  "word_line_width_nm": 32, "bit_line_pitch_nm": 96,
                  "bit_line_width_nm": 32, "contact_diameter_nm": 32},
            0.0, np.random.default_rng(0))
        dram_mod.POSITION_JITTER_NM, dram_mod.WIDTH_JITTER_FRACTION = 1.5, 0.10
        b = dram_mod.generate_dram_canvas(
            400, {"kind": "dram", "feature_size_nm": 32, "word_line_pitch_nm": 64,
                  "word_line_width_nm": 32, "bit_line_pitch_nm": 96,
                  "bit_line_width_nm": 32, "contact_diameter_nm": 32},
            0.0, np.random.default_rng(0))
    finally:
        dram_mod.POSITION_JITTER_NM, dram_mod.WIDTH_JITTER_FRACTION = saved

    assert not np.array_equal(a, b), \
        "patching WIDTH_JITTER_FRACTION had no effect -- default is late-bound"


def test_position_jitter_dist_uniform_is_actually_bounded():
    # Regression for the A1 "shifted" profile overclaim: JITTER_PROFILES had
    # a "dist": "uniform" key that was never read anywhere, so "shifted"
    # only ever changed jitter magnitude, never distribution family. Now
    # that POSITION_JITTER_DIST is threaded through, "uniform" must produce
    # values with bounded support (unlike "normal", which has unbounded
    # tails) -- a statistical, not just a "some values differ", check.
    from src.patterns import dram as dram_mod

    rng = np.random.default_rng(0)
    saved = dram_mod.POSITION_JITTER_DIST
    try:
        dram_mod.POSITION_JITTER_DIST = "uniform"
        half_width = dram_mod.POSITION_JITTER_NM * (3.0 ** 0.5)
        uniform_draws = np.array([dram_mod._position_jitter(rng) for _ in range(500)])
        assert np.all(np.abs(uniform_draws) <= half_width + 1e-9), \
            "uniform draws exceeded their bounded support"

        dram_mod.POSITION_JITTER_DIST = "normal"
        normal_draws = np.array([dram_mod._position_jitter(rng) for _ in range(500)])
        assert np.any(np.abs(normal_draws) > half_width), (
            "expected some normal-distributed draws to exceed the uniform "
            "family's bound (its tails are unbounded) -- dist switch may "
            "not be taking effect"
        )
    finally:
        dram_mod.POSITION_JITTER_DIST = saved


def test_shifted_profile_uses_uniform_position_jitter_via_generate_canvas_bundle():
    """End-to-end: the A1 'shifted' profile must actually route through
    POSITION_JITTER_DIST="uniform" during canvas generation, not just carry
    an unused "dist" key (the original bug) or merely restore the global
    correctly afterwards (necessary but not sufficient)."""
    from src.patterns import dram as dram_mod
    from src.localizer.data import JITTER_PROFILES, generate_canvas_bundle

    assert JITTER_PROFILES["shifted"]["dist"] == "uniform"
    assert JITTER_PROFILES["normal"]["dist"] == "normal"

    seen_dists = []
    real_position_jitter = dram_mod._position_jitter

    def spy(rng):
        seen_dists.append(dram_mod.POSITION_JITTER_DIST)
        return real_position_jitter(rng)

    dram_mod._position_jitter = spy
    try:
        generate_canvas_bundle(0, jitter_profile="shifted")
    finally:
        dram_mod._position_jitter = real_position_jitter

    assert seen_dists, "generate_canvas_bundle never called _position_jitter"
    assert all(d == "uniform" for d in seen_dists), (
        "the 'shifted' profile did not run with POSITION_JITTER_DIST="
        "'uniform' during canvas generation"
    )


import pytest
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.data import (
    LocalizerDataset, generate_canvas_bundle, sample_pair, split_seed_range,
)


def test_split_seed_ranges_are_disjoint():
    cfg = LocalizerConfig()
    tr = set(range(*split_seed_range("train", cfg)))
    va = set(range(*split_seed_range("val", cfg)))
    te = set(range(*split_seed_range("test", cfg)))
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_sample_pair_shapes_and_ground_truth_in_range():
    bundle = generate_canvas_bundle(0)
    s = sample_pair(bundle, 0, 0)
    assert s["reference_img"].shape == (100, 100)
    assert s["search_img"].shape == (1000, 1000)
    assert s["reference_img"].dtype == torch.float32
    assert 50.0 <= s["gt_x"] <= 950.0
    assert 50.0 <= s["gt_y"] <= 950.0


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


def test_same_seed_reproduces_identical_sample():
    a = sample_pair(generate_canvas_bundle(3), 3, 5)
    b = sample_pair(generate_canvas_bundle(3), 3, 5)
    assert torch.equal(a["reference_img"], b["reference_img"])
    assert a["gt_x"] == b["gt_x"] and a["gt_y"] == b["gt_y"]


def test_different_crop_indices_give_different_locations():
    bundle = generate_canvas_bundle(3)
    locs = {(sample_pair(bundle, 3, i)["gt_x"], sample_pair(bundle, 3, i)["gt_y"])
            for i in range(10)}
    assert len(locs) > 1


def test_standardisation_is_applied():
    s = sample_pair(generate_canvas_bundle(1), 1, 0)
    for key in ("reference_img", "search_img"):
        assert abs(float(s[key].mean())) < 1e-4
        assert abs(float(s[key].std()) - 1.0) < 1e-2


def test_generate_canvas_bundle_threads_dram_presets_to_generate_mxn_canvas():
    from src.localizer import data as data_mod

    allowed = ["dram_loose", "dram_wide", "dram_legacy"]
    seen = []
    real_generate_mxn_canvas = data_mod.generate_mxn_canvas

    def spy(*args, **kwargs):
        seen.append(kwargs.get("preset_names"))
        return real_generate_mxn_canvas(*args, **kwargs)

    data_mod.generate_mxn_canvas = spy
    try:
        data_mod.generate_canvas_bundle(0, dram_presets=allowed)
    finally:
        data_mod.generate_mxn_canvas = real_generate_mxn_canvas

    assert seen == [allowed]


def test_generate_canvas_bundle_dram_presets_none_is_a_noop():
    a = generate_canvas_bundle(11)["search_img"]
    b = generate_canvas_bundle(11, dram_presets=None)["search_img"]
    assert np.array_equal(a, b)


def test_zero_jitter_profile_produces_a_different_canvas():
    a = generate_canvas_bundle(11, jitter_profile="normal")["search_img"]
    b = generate_canvas_bundle(11, jitter_profile="zero")["search_img"]
    assert not torch.equal(torch.as_tensor(a), torch.as_tensor(b))


def test_dataset_geometric_profile_changes_reference_content():
    cfg = LocalizerConfig(crops_per_canvas=1, val_seed_lo=0, val_seed_hi=1)
    normal_item = next(iter(LocalizerDataset(
        "val", cfg, geometric_profile="normal", shuffle_buffer_size=1)))
    drift_item = next(iter(LocalizerDataset(
        "val", cfg, geometric_profile="drift", shuffle_buffer_size=1)))
    assert not torch.equal(normal_item["reference_img"], drift_item["reference_img"])


def test_dataset_threads_dram_presets_to_generate_canvas_bundle():
    from src.localizer import data as data_mod

    allowed = ["dram_loose", "dram_wide", "dram_legacy"]
    seen = []
    real_generate_canvas_bundle = data_mod.generate_canvas_bundle

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dram_presets"))
        return real_generate_canvas_bundle(*args, **kwargs)

    data_mod.generate_canvas_bundle = spy
    try:
        cfg = LocalizerConfig(crops_per_canvas=1, val_seed_lo=0, val_seed_hi=1)
        list(LocalizerDataset("val", cfg, dram_presets=allowed, shuffle_buffer_size=1))
    finally:
        data_mod.generate_canvas_bundle = real_generate_canvas_bundle

    assert seen == [allowed]


def test_dataset_yields_batchable_items():
    cfg = LocalizerConfig(crops_per_canvas=2)
    # A small shuffle_buffer_size here: this test only cares about item
    # schema/shapes, not shuffling (see
    # test_iter_shuffle_buffer_mixes_samples_from_multiple_canvases for
    # that), and the default buffer_size=256 would otherwise force ~128
    # canvases (~0.95s each) to be generated before the first item is
    # yielded, at this config's crops_per_canvas=2.
    it = iter(LocalizerDataset("val", cfg, shuffle_buffer_size=1))
    item = next(it)
    assert set(item) == {"reference_img", "search_img", "gt_x", "gt_y", "canvas_seed"}
    assert item["reference_img"].shape == (1, 100, 100)
    assert item["search_img"].shape == (1, 1000, 1000)


def test_iter_shuffle_buffer_mixes_samples_from_multiple_canvases():
    # Regression for "no real batch diversity": pre-fix, __iter__ yielded all
    # `crops_per_canvas` crops of one canvas consecutively before moving on,
    # so a DataLoader batch of consecutive items was always 8 crops of one
    # byte-identical search image. Use a seed range wide enough to span
    # several canvases and a shuffle buffer bigger than one canvas's crop
    # count, so a strictly canvas-by-canvas order would still show up as
    # long unbroken runs of one canvas_seed -- the shuffle must break those
    # runs up.
    cfg = LocalizerConfig(crops_per_canvas=5,
                          val_seed_lo=0, val_seed_hi=6)
    ds = LocalizerDataset("val", cfg, shuffle_buffer_size=8)
    seeds = [int(item["canvas_seed"]) for item in ds]

    n_canvases = 6
    assert len(seeds) == n_canvases * cfg.crops_per_canvas

    # A perfectly sequential (unshuffled) stream changes canvas_seed exactly
    # `n_canvases - 1` times (once at each canvas boundary). The shuffle
    # buffer should produce many more transitions than that.
    n_transitions = sum(1 for a, b in zip(seeds, seeds[1:]) if a != b)
    assert n_transitions > n_canvases - 1, (
        f"expected shuffled interleaving (>{n_canvases - 1} transitions), "
        f"got {n_transitions} -- looks canvas-sequential, not shuffled"
    )

    # Within a small early window (smaller than one canvas's crop count),
    # more than one distinct canvas_seed should already appear.
    window = seeds[:cfg.crops_per_canvas]
    assert len(set(window)) > 1, (
        "first window of samples came from a single canvas_seed -- "
        "shuffle buffer is not mixing canvases"
    )

    # Sharding contract preserved: the full set of seeds visited by this
    # single (unsharded, wid=0/nw=1) iterator is still every seed in range.
    assert set(seeds) == set(range(6))


def test_dataset_skips_a_canvas_whose_generation_raises_instead_of_crashing(monkeypatch):
    """Defense-in-depth: a rare numeric edge case in the imaging/canvas
    pipeline (seen in practice under harsh+drift synthetic validation)
    must not kill the whole DataLoader worker -- it should drop that one
    canvas's samples and continue with the next seed."""
    import src.localizer.data as data_mod

    real_generate_canvas_bundle = data_mod.generate_canvas_bundle

    def flaky_generate_canvas_bundle(seed, *args, **kwargs):
        if seed == 2:
            raise ValueError("simulated canvas-generation failure")
        return real_generate_canvas_bundle(seed, *args, **kwargs)

    monkeypatch.setattr(data_mod, "generate_canvas_bundle", flaky_generate_canvas_bundle)

    cfg = LocalizerConfig(crops_per_canvas=2, val_seed_lo=0, val_seed_hi=4)
    ds = LocalizerDataset("val", cfg, shuffle_buffer_size=1)
    with pytest.warns(RuntimeWarning, match="skipping canvas seed=2"):
        seeds = [int(item["canvas_seed"]) for item in ds]

    # Every seed except the flaky one still produced its samples.
    assert set(seeds) == {0, 1, 3}
    assert seeds.count(0) == cfg.crops_per_canvas
    assert seeds.count(1) == cfg.crops_per_canvas
    assert seeds.count(3) == cfg.crops_per_canvas
    assert 2 not in seeds


def test_dataset_skips_a_crop_whose_sampling_raises_instead_of_crashing(monkeypatch):
    """Same defense-in-depth, but for a failure in sample_pair (a single
    crop within an otherwise-good canvas) rather than canvas generation."""
    import src.localizer.data as data_mod

    real_sample_pair = data_mod.sample_pair

    def flaky_sample_pair(bundle, canvas_seed, crop_index, *args, **kwargs):
        if canvas_seed == 1 and crop_index == 1:
            raise ValueError("simulated crop-sampling failure")
        return real_sample_pair(bundle, canvas_seed, crop_index, *args, **kwargs)

    monkeypatch.setattr(data_mod, "sample_pair", flaky_sample_pair)

    cfg = LocalizerConfig(crops_per_canvas=3, val_seed_lo=0, val_seed_hi=3)
    ds = LocalizerDataset("val", cfg, shuffle_buffer_size=1)
    with pytest.warns(RuntimeWarning, match="skipping crop seed=1 k=1"):
        seeds = [int(item["canvas_seed"]) for item in ds]

    # Canvas 1 lost exactly one of its three crops; canvases 0 and 2 are
    # untouched.
    assert seeds.count(0) == cfg.crops_per_canvas
    assert seeds.count(1) == cfg.crops_per_canvas - 1
    assert seeds.count(2) == cfg.crops_per_canvas


def test_dataset_throttles_skip_warnings_past_the_limit(monkeypatch):
    """A failure rate higher than the "rare" case this was designed for
    must not flood a real training run's log with one warning per skip --
    past LocalizerDataset._SKIP_WARN_LIMIT, individual warnings stop and a
    single suppression notice fires instead."""
    import src.localizer.data as data_mod

    def always_fails(seed, *args, **kwargs):
        raise ValueError("simulated total canvas-generation failure")

    monkeypatch.setattr(data_mod, "generate_canvas_bundle", always_fails)

    limit = LocalizerDataset._SKIP_WARN_LIMIT
    cfg = LocalizerConfig(crops_per_canvas=1, val_seed_lo=0, val_seed_hi=limit + 5)
    ds = LocalizerDataset("val", cfg, shuffle_buffer_size=1)

    with pytest.warns(RuntimeWarning) as record:
        items = list(ds)

    assert items == []  # every canvas failed, nothing was ever yielded

    skip_messages = [str(w.message) for w in record if "skipping canvas" in str(w.message)]
    suppression_messages = [str(w.message) for w in record if "suppressing further" in str(w.message)]

    assert len(skip_messages) == limit
    assert len(suppression_messages) == 1
