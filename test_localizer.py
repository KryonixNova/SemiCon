import numpy as np
import pytest

from matcher import MatchResult
from localizer import DeepMatchLocalizer, LocalizationResult


class _StubMatcher:
    """Duck-typed stand-in for SuperPointLightGlueMatcher in unit tests --
    DeepMatchLocalizer only calls .extract() and .match() on its matcher,
    so a stub returning a fixed MatchResult avoids loading real models."""
    def __init__(self, match_result: MatchResult):
        self._result = match_result

    def extract(self, img):
        return img  # unused by the stub's match(); passthrough is enough

    def match(self, feats_a, feats_b):
        return self._result


def _make_match_result(pairs, score=0.9):
    kpts_a = np.array([p[0] for p in pairs], dtype=np.float32)
    kpts_b = np.array([p[1] for p in pairs], dtype=np.float32)
    scores = np.full(len(pairs), score, dtype=np.float32)
    return MatchResult(kpts_a=kpts_a, kpts_b=kpts_b, scores=scores)


def test_insufficient_matches_fails_gracefully():
    result = _make_match_result([((0, 0), (0, 0))] * 3)  # fewer than min_matches
    localizer = DeepMatchLocalizer(_StubMatcher(result), min_matches=8)
    out = localizer.localize(np.zeros((100, 100), np.uint8),
                              np.zeros((1000, 1000), np.uint8), zoom_ratio=10.0)
    assert isinstance(out, LocalizationResult)
    assert out.success is False
    assert out.failure_reason == "insufficient_matches"
    assert out.predicted_center_px is None
    assert out.confidence == 0.0


def test_ransac_failed_when_no_consistent_model():
    rng = np.random.default_rng(0)
    # Random, geometrically inconsistent correspondences -- no valid similarity fits.
    pairs = [((float(rng.uniform(0, 100)), float(rng.uniform(0, 100))),
              (float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))))
             for _ in range(10)]
    result = _make_match_result(pairs)
    localizer = DeepMatchLocalizer(_StubMatcher(result), min_matches=8, min_inliers=6)
    out = localizer.localize(np.zeros((100, 100), np.uint8),
                              np.zeros((1000, 1000), np.uint8), zoom_ratio=10.0)
    assert out.success is False
    assert out.failure_reason == "ransac_failed"


def test_closest_to_center_disambiguation():
    # Two internally-consistent similarity transforms (pure translations),
    # simulating two lattice-period-offset match clusters -- the concrete
    # form of the periodic-pattern ambiguity. Cluster A's predicted center
    # lands far from the search image center; cluster B's lands on it.
    offsets_xy = [(10, 10), (20, 10), (10, 20), (30, 30), (5, 40), (40, 5)]
    cluster_a = [((x, y), (x + 50, y + 50)) for x, y in offsets_xy]
    cluster_b = [((x, y), (x + 450, y + 450)) for x, y in offsets_xy]
    result = _make_match_result(cluster_a + cluster_b)

    localizer = DeepMatchLocalizer(_StubMatcher(result), min_matches=8, min_inliers=4,
                                    max_hypotheses=3, ransac_reproj_threshold=1.0)
    out = localizer.localize(np.zeros((100, 100), np.uint8),
                              np.zeros((1000, 1000), np.uint8), zoom_ratio=1.0)

    assert out.success is True
    # ref_center for a 100x100 reference at zoom_ratio=1.0 -> (50, 50).
    # cluster B predicted center = (50+450, 50+450) = (500, 500), the search center.
    px, py = out.predicted_center_px
    assert abs(px - 500) < 5
    assert abs(py - 500) < 5


def test_successful_localization_reports_matches_and_inliers():
    offsets_xy = [(10, 10), (20, 10), (10, 20), (30, 30), (5, 40), (40, 5)]
    pairs = [((x, y), (x + 450, y + 450)) for x, y in offsets_xy]
    result = _make_match_result(pairs, score=0.8)
    localizer = DeepMatchLocalizer(_StubMatcher(result), min_matches=4, min_inliers=4)
    out = localizer.localize(np.zeros((100, 100), np.uint8),
                              np.zeros((1000, 1000), np.uint8), zoom_ratio=1.0)
    assert out.success is True
    assert out.num_matches == 6
    assert out.num_inliers == 6
    assert out.confidence == pytest.approx(1.0 * 0.8, abs=1e-3)
    assert out.elapsed_ms >= 0.0
