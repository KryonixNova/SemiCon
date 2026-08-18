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
    for key in ("acc@5px", "acc@10px", "acc@50px", "median_error_px",
                "mean_error_px", "ap"):
        assert key in out
    assert out["acc@5px"] == 1.0
    assert out["median_error_px"] == 0.0


def test_summarize_reproduces_a_known_bimodal_split():
    # 6 perfect, 4 catastrophic -- mirrors the measured ZNCC failure shape.
    err_x = np.array([0.0] * 6 + [200.0] * 4)
    out = summarize(err_x, np.zeros(10), np.zeros(10), np.zeros(10), np.ones(10))
    assert out["acc@5px"] == 0.6
    assert out["acc@50px"] == 0.6
    assert out["median_error_px"] == 0.0
    assert out["mean_error_px"] == 80.0
