import cv2
import numpy as np
import pytest

from matcher import SuperPointLightGlueMatcher, MatchResult
from reference_search_pairs import generate_sample


SMALL_PARAMS_KW = dict(
    rows=16, cols=16, cell_pitch_nm=80.0, cell_pitch_bl_nm=60.0,
    wl_width_nm=24.0, bl_width_nm=18.0, contact_size_nm=14.0,
    capacitor_size_nm=30.0, n_particles=0, n_scratches=0,
    p_broken_wl=0.0, p_broken_bl=0.0, p_cmp_dishing=0.0,
)


def test_matcher_finds_correspondences_between_same_layout_renders(tmp_path):
    sample = generate_sample(seed=7, tmp_dir=str(tmp_path), **SMALL_PARAMS_KW)

    m = SuperPointLightGlueMatcher()

    ref_h, ref_w = sample.reference_img.shape[:2]
    low_w = round(ref_w / sample.zoom_ratio)
    low_h = round(ref_h / sample.zoom_ratio)
    ref_lowres = cv2.resize(sample.reference_img, (low_w, low_h), interpolation=cv2.INTER_AREA)

    feats_a = m.extract(ref_lowres)
    feats_b = m.extract(sample.search_img)
    result = m.match(feats_a, feats_b)

    assert isinstance(result, MatchResult)
    assert result.kpts_a.shape == result.kpts_b.shape
    assert result.kpts_a.shape[1] == 2
    assert result.scores.shape[0] == result.kpts_a.shape[0]
    assert len(result.scores) > 0
    assert result.kpts_a.dtype == np.float32
