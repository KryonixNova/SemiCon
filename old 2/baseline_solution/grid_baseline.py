#!/usr/bin/env python3
"""B1 -- the original 10x10 grid-classification baseline.

Splits the Search image into 100 non-overlapping 100x100 cells, scores the
downsampled Reference against each with ZNCC, and predicts the centre of the
best-scoring cell.

Implemented so its measured accuracy can be checked against `analytic_ceiling`,
which bounds what ANY 10x10 grid classifier can achieve -- even a perfect one.
Because the ground-truth centre is uniform within the chosen cell while the
prediction sits at the cell centre, accuracy cannot exceed the ratio of the
tolerance disc to the cell area. At 5 px that ceiling is 0.79%.
"""

import math

import cv2
import numpy as np

from src.localizer.geometry import REF_DS_PX, SCALE


def analytic_ceiling(tol_px: float, cell_px: int = 100) -> float:
    """Upper bound on acc@tol for a perfect 10x10 grid classifier."""
    return min(1.0, math.pi * tol_px ** 2 / cell_px ** 2)


def grid_predict(reference: np.ndarray, search: np.ndarray, grid: int = 10) -> dict:
    """Score the reference against each grid cell; return the best cell centre."""
    if reference.shape[0] != REF_DS_PX:
        reference = cv2.resize(reference, (REF_DS_PX, REF_DS_PX),
                               interpolation=cv2.INTER_AREA)
    cell = search.shape[0] // grid
    tpl = cv2.resize(reference, (cell, cell), interpolation=cv2.INTER_AREA)
    tpl = tpl.astype(np.float32)
    tpl = (tpl - tpl.mean()) / max(float(tpl.std()), 1e-6)

    best = {"score": -np.inf, "x": cell / 2.0, "y": cell / 2.0, "cell": (0, 0)}
    for r in range(grid):
        for c in range(grid):
            patch = search[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            patch = patch.astype(np.float32)
            patch = (patch - patch.mean()) / max(float(patch.std()), 1e-6)
            score = float((patch * tpl).mean())        # ZNCC over the cell
            if score > best["score"]:
                best = {"score": score,
                        "x": c * cell + cell / 2.0,
                        "y": r * cell + cell / 2.0,
                        "cell": (r, c)}
    return best
