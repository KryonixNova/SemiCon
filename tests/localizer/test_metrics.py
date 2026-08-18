import numpy as np

from src.localizer.metrics import (
    accuracy_at, average_precision, localization_error, pr_curve, summarize,
)


def test_localization_error_is_euclidean():
    e = localization_error(np.array([3.0]), np.array([4.0]),
                           np.array([0.0]), np.array([0.0]))
    assert e[0] == 5.0


def test_accuracy_at_counts_inclusive():
    errors = np.array([1.0, 5.0, 5.001, 100.0])
    assert accuracy_at(errors, 5.0) == 0.5


def test_perfect_ranking_gives_ap_one():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    corrects = np.array([True, True, True, True])
    p, r = pr_curve(scores, corrects, 4)
    assert average_precision(p, r) > 0.99


def test_ap_penalises_confident_mistakes():
    good = pr_curve(np.array([0.9, 0.1]), np.array([True, False]), 2)
    bad = pr_curve(np.array([0.9, 0.1]), np.array([False, True]), 2)
    assert average_precision(*good) > average_precision(*bad)


def test_summarize_reports_the_spec_tolerances():
    n = 10
    out = summarize(np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                    np.ones(n))
    for key in ("acc@1px", "acc@3px", "acc@5px", "median_error_px",
                "mean_error_px", "ap"):
        assert key in out
    assert out["acc@5px"] == 1.0
    assert out["median_error_px"] == 0.0


def test_default_tolerances_are_1_3_5px():
    out = summarize(np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.ones(5))
    acc_keys = {k for k in out if k.startswith("acc@")}
    assert acc_keys == {"acc@1px", "acc@3px", "acc@5px"}


def test_summarize_reproduces_a_known_bimodal_split():
    # 6 perfect, 4 catastrophic -- mirrors the measured ZNCC failure shape.
    # Explicit tolerances here (not the default) because this test is about
    # the split behavior at a tight AND a loose threshold, not about what
    # the default tuple happens to be.
    err_x = np.array([0.0] * 6 + [200.0] * 4)
    out = summarize(err_x, np.zeros(10), np.zeros(10), np.zeros(10), np.ones(10),
                    tolerances=(5.0, 50.0))
    assert out["acc@5px"] == 0.6
    assert out["acc@50px"] == 0.6
    assert out["median_error_px"] == 0.0
    assert out["mean_error_px"] == 80.0


def test_primary_tolerance_overrides_ap_threshold():
    # errors 2.0px and 4.0px. Default primary (=max of tolerances, 5.0) calls
    # both "correct" -> AP is perfect. Forcing primary_tolerance=3.0 makes the
    # 4.0px prediction "incorrect" -> AP must drop.
    px, py = [2.0, 4.0], [0.0, 0.0]
    gx, gy = [0.0, 0.0], [0.0, 0.0]
    scores = [0.9, 0.8]
    loose = summarize(px, py, gx, gy, scores, tolerances=(1.0, 3.0, 5.0))
    tight = summarize(px, py, gx, gy, scores, tolerances=(1.0, 3.0, 5.0),
                      primary_tolerance=3.0)
    assert loose["ap"] > tight["ap"]
