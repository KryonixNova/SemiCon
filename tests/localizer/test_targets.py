import torch

from src.localizer.geometry import CORR, STRIDE, cell_to_pixel
from src.localizer.targets import build_targets


def test_shapes_and_peak_value():
    t = build_targets(torch.tensor([500.0]), torch.tensor([500.0]))
    assert t["heatmap"].shape == (1, CORR, CORR)
    assert t["offset"].shape == (1, 2, CORR, CORR)
    assert torch.isclose(t["heatmap"].max(), torch.tensor(1.0), atol=1e-6)


def test_peak_lands_on_the_nearest_cell():
    x = torch.tensor([cell_to_pixel(30)])
    y = torch.tensor([cell_to_pixel(70)])
    t = build_targets(x, y)
    assert tuple(t["peak_cell"][0].tolist()) == (30, 70)
    flat = int(torch.argmax(t["heatmap"][0]))
    assert divmod(flat, CORR) == (70, 30)          # (row, col)


def test_offset_target_is_within_half_a_stride():
    xs = torch.linspace(50.0, 950.0, 101)
    t = build_targets(xs, xs)
    b = torch.arange(len(xs))
    cols, rows = t["peak_cell"][:, 0], t["peak_cell"][:, 1]
    dx = t["offset"][b, 0, rows, cols]
    dy = t["offset"][b, 1, rows, cols]
    assert dx.abs().max() <= STRIDE / 2 + 1e-4
    assert dy.abs().max() <= STRIDE / 2 + 1e-4


def test_offset_reconstructs_the_true_coordinate():
    x = torch.tensor([437.3])
    t = build_targets(x, x)
    col, row = t["peak_cell"][0].tolist()
    recovered = cell_to_pixel(col) + float(t["offset"][0, 0, row, col])
    assert abs(recovered - 437.3) < 1e-3


def test_larger_sigma_spreads_more_mass():
    narrow = build_targets(torch.tensor([500.0]), torch.tensor([500.0]), sigma_cells=1.0)
    wide = build_targets(torch.tensor([500.0]), torch.tensor([500.0]), sigma_cells=3.0)
    assert wide["heatmap"].sum() > narrow["heatmap"].sum()
