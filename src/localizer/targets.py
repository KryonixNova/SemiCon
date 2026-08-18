"""Training targets: a soft Gaussian heatmap plus a sub-cell offset field.

Hard 0/1 cell labels are wrong for this problem. Ground-truth centres are
uniformly distributed within a cell (measured: zero of 100 boxes align to a
100 px grid), so a hard label puts a decision discontinuity exactly where
half the samples live. The Gaussian is peaked at 1.0 on the nearest cell,
which is what the CenterNet focal loss expects as its positive location.

The offset field stores the residual in PIXELS, in [-STRIDE/2, +STRIDE/2].
"""

from __future__ import annotations

import torch

from src.localizer.geometry import CORR, HALF, STRIDE


def build_targets(gt_x: torch.Tensor, gt_y: torch.Tensor,
                  sigma_cells: float = 2.0, align_offset: float = 0.0) -> dict:
    gt_x = torch.as_tensor(gt_x, dtype=torch.float32).reshape(-1)
    gt_y = torch.as_tensor(gt_y, dtype=torch.float32).reshape(-1)
    b = gt_x.shape[0]
    dev = gt_x.device

    # Continuous cell coordinates, then the nearest integer cell.
    cf_x = (gt_x - HALF - align_offset) / STRIDE
    cf_y = (gt_y - HALF - align_offset) / STRIDE
    col = cf_x.round().clamp(0, CORR - 1).long()
    row = cf_y.round().clamp(0, CORR - 1).long()

    grid = torch.arange(CORR, dtype=torch.float32, device=dev)
    d_col = (grid.view(1, 1, CORR) - cf_x.view(b, 1, 1)) ** 2
    d_row = (grid.view(1, CORR, 1) - cf_y.view(b, 1, 1)) ** 2
    heatmap = torch.exp(-(d_col + d_row) / (2.0 * sigma_cells ** 2))
    # Force an exact 1.0 at the assigned cell: the focal loss treats
    # target == 1 as the sole positive.
    heatmap[torch.arange(b, device=dev), row, col] = 1.0

    offset = torch.zeros(b, 2, CORR, CORR, dtype=torch.float32, device=dev)
    res_x = gt_x - (STRIDE * col.float() + HALF + align_offset)
    res_y = gt_y - (STRIDE * row.float() + HALF + align_offset)
    idx = torch.arange(b, device=dev)
    offset[idx, 0, row, col] = res_x
    offset[idx, 1, row, col] = res_y

    return {"heatmap": heatmap, "offset": offset,
            "peak_cell": torch.stack([col, row], dim=1)}
