import numpy as np
import pytest


@pytest.mark.slow
def test_predict_pair_returns_xyz(tmp_path, tiny_checkpoint):
    import cv2

    from src.localizer.inference import load_model, predict_pair

    ckpt_path, device = tiny_checkpoint
    rng = np.random.default_rng(0)
    ref_path = tmp_path / "ref.png"
    search_path = tmp_path / "search.png"
    cv2.imwrite(str(ref_path), rng.integers(0, 255, (100, 100), dtype=np.uint8))
    cv2.imwrite(str(search_path), rng.integers(0, 255, (1000, 1000), dtype=np.uint8))

    model = load_model(ckpt_path, device)
    result = predict_pair(model, str(ref_path), str(search_path), device)

    assert set(result) == {"x", "y", "confidence"}
    assert isinstance(result["x"], float) and isinstance(result["y"], float)
    assert isinstance(result["confidence"], float)


@pytest.mark.slow
def test_predict_pair_rejects_wrong_search_size(tmp_path, tiny_checkpoint):
    import cv2

    from src.localizer.inference import load_model, predict_pair

    ckpt_path, device = tiny_checkpoint
    rng = np.random.default_rng(0)
    ref_path = tmp_path / "ref.png"
    bad_search_path = tmp_path / "bad_search.png"
    cv2.imwrite(str(ref_path), rng.integers(0, 255, (100, 100), dtype=np.uint8))
    cv2.imwrite(str(bad_search_path), rng.integers(0, 255, (500, 500), dtype=np.uint8))

    model = load_model(ckpt_path, device)
    with pytest.raises(ValueError, match="must be"):
        predict_pair(model, str(ref_path), str(bad_search_path), device)


def test_load_standardized_resizes_reference_to_target_px(tmp_path):
    import cv2

    from src.localizer.inference import load_standardized

    rng = np.random.default_rng(0)
    path = tmp_path / "big_ref.png"
    cv2.imwrite(str(path), rng.integers(0, 255, (1000, 1000), dtype=np.uint8))

    t = load_standardized(str(path), target_px=100)
    assert t.shape == (1, 1, 100, 100)
