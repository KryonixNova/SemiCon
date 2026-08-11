"""m x n multi-region DRAM die composition.

Unlike src/patterns/zones.py's generate_zone_canvas -- which always yields a
square k x k grid because it derives both axes from one mat_size_nm -- this
picks the row and column counts independently, giving true m x n layouts.

Collapse is always disabled here (collapse_threshold_nm=0.0): this module
backs the no-defect DRAM dataset and the localizer's training data.
"""

from __future__ import annotations

import numpy as np

from src.patterns.dram import generate_dram_canvas
from src.patterns.zones import _strip_routing_texture
from src.presets import get_preset, presets_for_kind


def fixed_count_spans(size_px: int, count: int, strip_width_nm: float):
    """Alternating [mat, strip, mat, ...] spans with exactly `count` mats."""
    if count == 1:
        return [(True, 0, size_px)]
    mat_size = (size_px - (count - 1) * strip_width_nm) / count
    spans, pos = [], 0.0
    for k in range(count):
        mat_end = pos + mat_size
        spans.append((True, int(round(pos)), int(round(mat_end))))
        pos = mat_end
        if k < count - 1:
            strip_end = pos + strip_width_nm
            spans.append((False, int(round(pos)), int(round(strip_end))))
            pos = strip_end
    return spans


def generate_mxn_canvas(
    size_px: int, m: int, n: int, strip_width_nm: float, rng: np.random.Generator,
    linewidth_bias_nm: float = 0.0, corner_rounding_px: float = 0.0,
    preset_names: list[str] | None = None,
) -> dict:
    """Tile an m (rows) x n (cols) grid of independent DRAM mats, each with
    its own randomly chosen preset, separated by routing-strip material.

    `linewidth_bias_nm`/`corner_rounding_px` are passed straight through to
    every mat's `generate_dram_canvas` call -- the same CD-bias/corner-
    rounding distortion the single-architecture pipeline (src/pipeline.py)
    already supports, just applied uniformly across every region of this
    m x n grid.

    `preset_names`, if given, restricts every mat's preset draw to that
    subset of DRAM_PRESET_NAMES (e.g. training on only the presets a prior
    checkpoint tested weak on) instead of the full six-preset pool. `None`
    (the default) reproduces the original all-presets behavior exactly.
    """
    presets = ([get_preset(n) for n in preset_names] if preset_names
               else presets_for_kind("dram"))
    canvas = _strip_routing_texture(size_px, rng)

    row_spans = fixed_count_spans(size_px, m, strip_width_nm)
    col_spans = fixed_count_spans(size_px, n, strip_width_nm)

    mat_rects, strip_rects, mat_features = [], [], []
    for row_is_mat, y0, y1 in row_spans:
        for col_is_mat, x0, x1 in col_spans:
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                mat_h, mat_w = y1 - y0, x1 - x0
                preset = presets[int(rng.integers(0, len(presets)))]
                child_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
                mat_canvas = generate_dram_canvas(
                    max(mat_h, mat_w), preset, 0.0, child_rng,
                    linewidth_bias_nm=linewidth_bias_nm,
                    corner_rounding_px=corner_rounding_px,
                )
                canvas[y0:y1, x0:x1] = mat_canvas[:mat_h, :mat_w]
                mat_rects.append((x0, y0, mat_w, mat_h))
                mat_features.append(preset["feature_size_nm"])
            elif not (row_is_mat and col_is_mat):
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))

    return {
        "canvas": canvas,
        "mat_rects": mat_rects,
        "strip_rects": strip_rects,
        "mat_feature_sizes_nm": mat_features,
    }
