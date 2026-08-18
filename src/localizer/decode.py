"""Heatmap -> (x, y, confidence).

Two rules encode design decisions made deliberately:

1. The closest-to-image-centre tie-break applies ONLY here, at decode time,
   and only among peaks that are genuinely tied (within peak_tie_ratio of the
   maximum). It never enters any loss: ground-truth centres are uniformly
   distributed, so a learned centre prior would be actively harmful. When
   several tied cells share the exact same minimum centre-distance (e.g. the
   four corners of a square grid are all equidistant from centre), a
   secondary key breaks that remaining tie by preferring the higher `peaks`
   value, so the choice is never left to arbitrary flattened-index order.

2. confidence is the peak-to-second-peak MARGIN, not the peak height. The
   ZNCC baseline scored 0.902 on average across a set containing 35%
   failures, so absolute score does not separate right from wrong here.

Note: confidence can be negative. This happens specifically when the
tie-break above overrides raw peak ranking -- it chooses the centre-closer
of two near-tied peaks even when that peak is not the single highest-scoring
one on the map. In that case the "runner-up" used for the margin can score
higher than the chosen peak, making chosen - runner_up < 0. This is not a
bug: it is the honest signal that decode deferred to the tie-break rather
than to raw ranking, i.e. "the model was ambiguous here." Downstream
consumers only need confidence's relative ordering across predictions, so a
negative value is fine to rank on -- treat it as low confidence, not as an
error to guard against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.localizer.geometry import CORR, HALF, STRIDE


def decode(heatmap: torch.Tensor, offset: torch.Tensor,
           peak_tie_ratio: float = 0.95, nms_kernel: int = 3,
           align_offset: float = 0.0) -> dict:
    b = heatmap.shape[0]
    dev = heatmap.device

    # 3x3 non-maximum suppression: keep only local maxima.
    pooled = F.max_pool2d(heatmap.unsqueeze(1), nms_kernel,
                          stride=1, padding=nms_kernel // 2).squeeze(1)
    peaks = heatmap * (heatmap >= pooled - 1e-9).float()

    flat = peaks.reshape(b, -1)
    best = flat.max(dim=1).values

    # Distance of every cell from the map centre, used to break ties.
    grid = torch.arange(CORR, dtype=torch.float32, device=dev)
    mid = (CORR - 1) / 2.0
    dist = ((grid.view(1, CORR, 1) - mid) ** 2 +
            (grid.view(1, 1, CORR) - mid) ** 2).sqrt().expand(b, CORR, CORR)

    tied = peaks >= (best.view(b, 1, 1) * peak_tie_ratio)
    tied &= peaks > 0
    # Among tied peaks pick minimum centre-distance; others get +inf.
    masked = torch.where(tied, dist, torch.full_like(dist, float("inf")))
    masked_flat = masked.reshape(b, -1)
    min_dist = masked_flat.min(dim=1, keepdim=True).values
    # Secondary tie-break: among cells at (or within float tolerance of) the
    # minimum centre-distance, prefer the one with the strongest actual
    # signal instead of falling back to positional/flattened-index order.
    dist_candidates = masked_flat <= (min_dist + 1e-4)
    peaks_flat = peaks.reshape(b, -1)
    tie_break_peaks = torch.where(
        dist_candidates, peaks_flat, torch.full_like(peaks_flat, float("-inf")))
    chosen = tie_break_peaks.argmax(dim=1)
    row, col = chosen // CORR, chosen % CORR

    idx = torch.arange(b, device=dev)
    dx = offset[idx, 0, row, col]
    dy = offset[idx, 1, row, col]

    x = STRIDE * col.float() + HALF + align_offset + dx
    y = STRIDE * row.float() + HALF + align_offset + dy

    # Margin: chosen peak minus the strongest peak outside a small exclusion
    # window around it.
    excl = ((grid.view(1, CORR, 1) - row.view(b, 1, 1).float()) ** 2 +
            (grid.view(1, 1, CORR) - col.view(b, 1, 1).float()) ** 2) > 9.0
    runner_up = torch.where(excl, peaks, torch.zeros_like(peaks))
    runner_up = runner_up.reshape(b, -1).max(dim=1).values
    confidence = peaks[idx, row, col] - runner_up

    return {"x": x, "y": y, "confidence": confidence}
