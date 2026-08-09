import numpy as np
import pytest

from baseline_solution.grid_baseline import analytic_ceiling, grid_predict


def test_analytic_ceiling_matches_spec_values():
    assert analytic_ceiling(5.0) == pytest.approx(0.0079, abs=1e-4)
    assert analytic_ceiling(10.0) == pytest.approx(0.0314, abs=1e-4)
    assert analytic_ceiling(50.0) == pytest.approx(0.7854, abs=1e-4)


def test_ceiling_is_a_probability():
    assert 0.0 <= analytic_ceiling(200.0) <= 1.0


def test_grid_predict_returns_a_cell_centre():
    rng = np.random.default_rng(0)
    search = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)
    reference = search[300:400, 300:400]
    out = grid_predict(reference, search)
    # Cell centres are at 50, 150, 250, ... by construction.
    assert (out["x"] - 50) % 100 == 0
    assert (out["y"] - 50) % 100 == 0


def test_grid_predict_finds_an_aligned_patch():
    rng = np.random.default_rng(1)
    search = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)
    reference = search[500:600, 200:300]      # exactly cell (row 5, col 2)
    out = grid_predict(reference, search)
    assert (out["x"], out["y"]) == (250.0, 550.0)
