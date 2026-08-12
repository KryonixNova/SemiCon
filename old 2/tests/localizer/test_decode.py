import torch

from src.localizer.decode import decode
from src.localizer.geometry import CORR, cell_to_pixel
from src.localizer.targets import build_targets


def _blank():
    return torch.zeros(1, CORR, CORR), torch.zeros(1, 2, CORR, CORR)


def test_single_peak_decodes_to_its_pixel_centre():
    hm, off = _blank()
    hm[0, 70, 30] = 1.0
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(30)
    assert float(out["y"][0]) == cell_to_pixel(70)


def test_offset_is_added_to_the_cell_centre():
    hm, off = _blank()
    hm[0, 70, 30] = 1.0
    off[0, 0, 70, 30] = 1.75
    off[0, 1, 70, 30] = -1.25
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(30) + 1.75
    assert float(out["y"][0]) == cell_to_pixel(70) - 1.25


def test_tied_peaks_resolve_to_the_one_nearest_image_centre():
    hm, off = _blank()
    hm[0, 113, 113] = 1.0        # centre of the map
    hm[0, 5, 5] = 1.0            # equally strong, far corner
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(113)
    assert float(out["y"][0]) == cell_to_pixel(113)


def test_a_clearly_stronger_far_peak_still_wins():
    hm, off = _blank()
    hm[0, 113, 113] = 0.30       # below the tie ratio
    hm[0, 5, 5] = 1.00
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(5)


def test_confidence_is_the_margin_not_the_peak_height():
    hm_amb, off = _blank()
    hm_amb[0, 40, 40] = 0.9
    hm_amb[0, 150, 150] = 0.88   # ambiguous: high peak, tiny margin

    hm_clear, _ = _blank()
    hm_clear[0, 40, 40] = 0.9
    hm_clear[0, 150, 150] = 0.05  # unambiguous: same peak, large margin

    assert float(decode(hm_clear, off)["confidence"][0]) > \
           float(decode(hm_amb, off)["confidence"][0])


def test_decode_recovers_the_target_it_was_built_from():
    t = build_targets(torch.tensor([437.3]), torch.tensor([612.9]))
    out = decode(t["heatmap"], t["offset"])
    assert abs(float(out["x"][0]) - 437.3) < 1e-2
    assert abs(float(out["y"][0]) - 612.9) < 1e-2


def test_batch_is_handled():
    t = build_targets(torch.tensor([200.0, 800.0]), torch.tensor([300.0, 700.0]))
    out = decode(t["heatmap"], t["offset"])
    assert out["x"].shape == (2,)
    assert abs(float(out["x"][0]) - 200.0) < 1e-2
    assert abs(float(out["x"][1]) - 800.0) < 1e-2


def test_exact_distance_tie_resolves_to_the_higher_peak():
    # Two cells symmetric about the grid centre (~112.5, 112.5) are at the
    # EXACT same centre-distance, so the primary "closest to centre"
    # tie-break can't distinguish them. Before the fix, argmin fell through
    # to flattened row-major order and always picked the first one
    # regardless of its score. The secondary key must instead prefer the
    # cell with the higher peak value.
    hm, off = _blank()
    hm[0, 100, 100] = 0.99   # symmetric about (112.5, 112.5), weaker peak
    hm[0, 125, 125] = 1.00   # symmetric about (112.5, 112.5), stronger peak
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(125)
    assert float(out["y"][0]) == cell_to_pixel(125)


def test_exact_distance_tie_does_not_default_to_first_index():
    # Same exact-distance tie as above, but with the stronger peak placed
    # FIRST in row-major order, to make sure the fix isn't accidentally just
    # picking "the second candidate" -- it must genuinely track peak height.
    hm, off = _blank()
    hm[0, 100, 100] = 1.00   # stronger peak, comes first in flattened order
    hm[0, 125, 125] = 0.99   # weaker peak
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(100)
    assert float(out["y"][0]) == cell_to_pixel(100)


def test_confidence_can_go_negative_when_tie_break_overrules_raw_score():
    # Two near-tied peaks (within peak_tie_ratio): (150, 150) is closer to
    # the 226x226 grid's centre (~112.5, 112.5) than (40, 40) is, so the
    # centre tie-break picks (150, 150) even though its raw score (0.88) is
    # lower than the other peak's (0.9). The margin is then computed against
    # that higher-scoring runner-up, so confidence must come out negative --
    # this is the deliberate "the model was ambiguous here" signal, not a
    # bug, and it must not silently regress.
    hm, off = _blank()
    hm[0, 40, 40] = 0.9
    hm[0, 150, 150] = 0.88
    out = decode(hm, off)
    assert float(out["x"][0]) == cell_to_pixel(150)
    assert float(out["y"][0]) == cell_to_pixel(150)
    assert float(out["confidence"][0]) < 0
