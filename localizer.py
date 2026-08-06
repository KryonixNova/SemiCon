"""Zoom-normalized, sequential-RANSAC similarity-transform localization.

The dataset only ever applies translation, small independent rotation, and
a known uniform zoom -- never perspective distortion -- so a similarity
transform (rotation + uniform scale + translation, cv2.estimateAffinePartial2D)
is the correct geometric model. Full homography is out of scope unless
perspective distortion is introduced to the dataset later.
"""
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LocalizationResult:
    predicted_center_px: Optional[Tuple[float, float]]
    confidence: float
    elapsed_ms: float
    num_matches: int
    num_inliers: int
    success: bool
    failure_reason: Optional[str]


class DeepMatchLocalizer:
    def __init__(self, matcher, min_matches: int = 8, min_inliers: int = 4,
                 max_hypotheses: int = 3, ransac_reproj_threshold: float = 3.0):
        self.matcher = matcher
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.max_hypotheses = max_hypotheses
        self.ransac_reproj_threshold = ransac_reproj_threshold

    def localize(self, reference_img: np.ndarray, search_img: np.ndarray,
                 zoom_ratio: float) -> LocalizationResult:
        t0 = time.perf_counter()

        ref_h, ref_w = reference_img.shape[:2]
        low_w = max(1, round(ref_w / zoom_ratio))
        low_h = max(1, round(ref_h / zoom_ratio))
        ref_lowres = cv2.resize(reference_img, (low_w, low_h), interpolation=cv2.INTER_AREA)

        feats_ref = self.matcher.extract(ref_lowres)
        feats_search = self.matcher.extract(search_img)
        matches = self.matcher.match(feats_ref, feats_search)
        num_matches = len(matches.scores)

        if num_matches < self.min_matches:
            return LocalizationResult(None, 0.0, _elapsed_ms(t0), num_matches, 0,
                                       False, "insufficient_matches")

        search_h, search_w = search_img.shape[:2]
        search_center = (search_w / 2.0, search_h / 2.0)
        ref_center = np.array([low_w / 2.0, low_h / 2.0], dtype=np.float64)

        hypotheses = self._find_hypotheses(matches, ref_center)

        if not hypotheses:
            return LocalizationResult(None, 0.0, _elapsed_ms(t0), num_matches, 0,
                                       False, "ransac_failed")

        best = min(hypotheses, key=lambda h: _dist(h["predicted_center"], search_center))

        inlier_ratio = best["num_inliers"] / num_matches
        confidence = inlier_ratio * best["mean_score"]
        if not (0.5 <= best["scale"] <= 2.0):
            confidence *= 0.5

        return LocalizationResult(
            predicted_center_px=best["predicted_center"],
            confidence=float(confidence),
            elapsed_ms=_elapsed_ms(t0),
            num_matches=num_matches,
            num_inliers=best["num_inliers"],
            success=True,
            failure_reason=None,
        )

    def _find_hypotheses(self, matches, ref_center: np.ndarray) -> list:
        pts_a, pts_b, scores = matches.kpts_a, matches.kpts_b, matches.scores
        remaining = np.arange(len(scores))
        hypotheses = []

        for _ in range(self.max_hypotheses):
            if len(remaining) < self.min_inliers:
                break
            M, inlier_mask = cv2.estimateAffinePartial2D(
                pts_a[remaining], pts_b[remaining],
                method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_threshold)
            if M is None:
                break
            inlier_mask = inlier_mask.ravel().astype(bool)
            n_inliers = int(inlier_mask.sum())
            if n_inliers < self.min_inliers:
                break

            inlier_idx = remaining[inlier_mask]
            predicted = M @ np.array([ref_center[0], ref_center[1], 1.0])
            hypotheses.append({
                "predicted_center": (float(predicted[0]), float(predicted[1])),
                "num_inliers": n_inliers,
                "mean_score": float(scores[inlier_idx].mean()),
                "scale": float(np.hypot(M[0, 0], M[1, 0])),
            })
            remaining = remaining[~inlier_mask]

        return hypotheses


def _dist(a: tuple, b: tuple) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
