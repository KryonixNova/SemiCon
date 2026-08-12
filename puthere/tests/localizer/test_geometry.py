import numpy as np
import pytest

from src.localizer.geometry import (
    CORR, HALF, REF_FEAT, SEARCH_FEAT, STRIDE,
    cell_to_pixel, pixel_to_cell, pixel_to_cell_offset,
)


def test_derived_constants():
    assert REF_FEAT == 25
    assert SEARCH_FEAT == 250
    assert CORR == SEARCH_FEAT - REF_FEAT + 1 == 226


def test_grid_exactly_covers_ground_truth_range():
    # Generator crops the fine canvas at x0 in [0, 9000]; gt centre = x0/10 + 50.
    assert cell_to_pixel(0) == 50.0
    assert cell_to_pixel(CORR - 1) == 950.0


def test_round_trip_residual_within_half_stride():
    for x in np.linspace(50.0, 950.0, 2001):
        c, delta = pixel_to_cell_offset(x)
        assert 0 <= c <= CORR - 1
        assert abs(delta) <= STRIDE / 2 + 1e-9
        assert cell_to_pixel(c) + delta == pytest.approx(x)


def test_offset_target_range_is_symmetric():
    deltas = [pixel_to_cell_offset(x)[1] for x in np.linspace(50.0, 950.0, 2001)]
    assert min(deltas) >= -STRIDE / 2 - 1e-9
    assert max(deltas) <= STRIDE / 2 + 1e-9


def test_align_offset_shifts_mapping():
    assert cell_to_pixel(10, align_offset=1.5) == cell_to_pixel(10) + 1.5
    assert pixel_to_cell(cell_to_pixel(10, 1.5), align_offset=1.5) == 10


def test_pixel_to_cell_clamps_out_of_range():
    assert pixel_to_cell(-1000.0) == 0
    assert pixel_to_cell(1e6) == CORR - 1
