import numpy as np

from src.sem_imaging import add_shot_noise, apply_barrel_distortion


def test_add_shot_noise_survives_nan_and_inf_input():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    out = add_shot_noise(img, dose=np.nan, rng=rng)
    assert out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float64)).all()


def test_add_shot_noise_survives_inf_and_implausibly_large_dose():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    for dose in (np.inf, 1e300):
        out = add_shot_noise(img, dose=dose, rng=rng)
        assert out.dtype == np.uint8
        assert np.isfinite(out.astype(np.float64)).all()


def test_add_shot_noise_survives_zero_and_negative_dose():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    for dose in (0.0, -5.0):
        out = add_shot_noise(img, dose=dose, rng=rng)
        assert out.dtype == np.uint8
        assert np.isfinite(out.astype(np.float64)).all()


def test_add_shot_noise_normal_case_unaffected():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    out_a = add_shot_noise(img, dose=400.0, rng=rng_a)
    out_b = add_shot_noise(img, dose=400.0, rng=rng_b)
    np.testing.assert_array_equal(out_a, out_b)


def test_apply_barrel_distortion_survives_extreme_k_without_crashing():
    """Hardening test for this function in isolation, NOT a confirmed
    regression test for a specific production crash: a training run once
    crashed with a float32 overflow inside this function under real
    harsh+drift sampling, but that overflow was not reproducible from any
    k value actually reachable by a real imaging-noise profile (harsh caps
    at +-0.08) -- the true trigger for that crash is still unconfirmed,
    and unblocking training used a separate defense-in-depth fix in
    LocalizerDataset (see test_data.py's
    test_dataset_skips_a_canvas_whose_generation_raises_instead_of_crashing).
    This test only pins the property that actually matters here: for any
    k, however extreme, this function must never raise and must always
    return a finite, valid uint8 image -- it must not itself become a
    second, independent source of worker crashes.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    for k in (1e10, -1e10, 1e20, 1e37, -1e37, 1e300):
        out = apply_barrel_distortion(img.copy(), k)
        assert out.dtype == np.uint8
        assert out.shape == img.shape
        assert np.isfinite(out.astype(np.float64)).all()


def test_apply_barrel_distortion_normal_case_nearly_unaffected():
    """float64 (this fix) vs the original float32 computation must agree
    almost exactly for realistic k -- this isn't byte-identical (float64
    rounds slightly differently than float32 before the final cast back to
    float32 for cv2.remap), but the difference must be at most a couple of
    intensity levels, not a behavior change.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    for k in (-0.08, -0.02, 0.02, 0.08):
        old = _old_float32_barrel_distortion(img.copy(), k)
        new = apply_barrel_distortion(img.copy(), k)
        assert new.dtype == np.uint8
        assert new.shape == img.shape
        max_diff = np.abs(old.astype(int) - new.astype(int)).max()
        assert max_diff <= 2, f"k={k}: max pixel diff {max_diff} exceeds precision tolerance"


def _old_float32_barrel_distortion(img, k):
    """The pre-fix implementation, kept only in this test to pin how much
    the fix is allowed to change normal-case output by."""
    import cv2
    if k == 0.0:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx ** 2 + ny ** 2
    factor = 1.0 + k * r2
    map_x = (nx * factor) * cx + cx
    map_y = (ny * factor) * cy + cy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
