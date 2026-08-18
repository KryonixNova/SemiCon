"""Evaluation metrics.

localization_error lives here and nowhere else: it needs ground truth and is
therefore an evaluation quantity, not a model output. The model emits only
predicted_x, predicted_y, confidence.

The PR/AP convention matches baseline_solution/evaluate.py so numbers are
directly comparable: every sample has exactly one true match, so total
positives = N regardless of the acceptance threshold.
"""

from __future__ import annotations

import numpy as np


def localization_error(pred_x, pred_y, gt_x, gt_y) -> np.ndarray:
    return np.hypot(np.asarray(pred_x, dtype=float) - np.asarray(gt_x, dtype=float),
                    np.asarray(pred_y, dtype=float) - np.asarray(gt_y, dtype=float))


def accuracy_at(errors, tol: float) -> float:
    return float((np.asarray(errors) <= tol).mean())


def pr_curve(scores, corrects, n_total: int):
    order = np.argsort(-np.asarray(scores, dtype=float))
    c = np.asarray(corrects, dtype=bool)[order]
    tp = np.cumsum(c)
    fp = np.cumsum(~c)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_total, 1)
    return (np.concatenate([[1.0], precision]),
            np.concatenate([[0.0], recall]))


def average_precision(precision, recall) -> float:
    order = np.argsort(recall)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(np.asarray(precision)[order], np.asarray(recall)[order]))


def summarize(pred_x, pred_y, gt_x, gt_y, scores, tolerances=(5.0, 10.0, 50.0)) -> dict:
    errors = localization_error(pred_x, pred_y, gt_x, gt_y)
    out = {f"acc@{int(t)}px": accuracy_at(errors, t) for t in tolerances}
    out["median_error_px"] = float(np.median(errors))
    out["mean_error_px"] = float(errors.mean())
    out["n"] = int(len(errors))
    primary = max(tolerances) if 50.0 not in tolerances else 50.0
    p, r = pr_curve(scores, errors <= primary, len(errors))
    out["ap"] = average_precision(p, r)
    return out
