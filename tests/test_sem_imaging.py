import numpy as np

from src.sem_imaging import add_shot_noise


def test_add_shot_noise_survives_nan_and_inf_input():
    img = np.full((8, 8), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    out = add_shot_noise(img, dose=np.nan, rng=rng)
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
