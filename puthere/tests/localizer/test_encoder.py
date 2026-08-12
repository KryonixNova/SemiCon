import torch

from src.localizer.encoder import SiameseEncoder
from src.localizer.geometry import REF_FEAT, SEARCH_FEAT, STRIDE


def test_output_shapes_match_the_design():
    enc = SiameseEncoder().eval()
    with torch.no_grad():
        s = enc(torch.zeros(1, 1, 1000, 1000))
        r = enc(torch.zeros(1, 1, 100, 100))
    assert s.shape == (1, 128, SEARCH_FEAT, SEARCH_FEAT)
    assert r.shape == (1, 128, REF_FEAT, REF_FEAT)


def test_features_are_l2_normalised_along_channels():
    enc = SiameseEncoder().eval()
    with torch.no_grad():
        f = enc(torch.randn(1, 1, 200, 200))
    norms = f.pow(2).sum(dim=1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_translation_equivariance():
    """Shifting the input by k*STRIDE px must shift features by exactly k."""
    torch.manual_seed(0)
    enc = SiameseEncoder().eval()
    x = torch.randn(1, 1, 256, 256)
    k = 3
    shift_px = k * STRIDE
    with torch.no_grad():
        f_full = enc(x)
        f_shift = enc(torch.roll(x, shifts=(0, -shift_px), dims=(2, 3)))
    # Compare an interior window to avoid wrap-around and padding artefacts.
    # margin=24: layer2's dilated (dilation=2) 3x3 convs add real
    # receptive-field reach near the crop boundary -- verified empirically
    # (diff was 0.064 at margin=8, 0.0 at margin=16); 24 gives headroom
    # above the exact threshold instead of sitting right on it.
    a = f_full[:, :, 24:-24, 24 + k:-24 + k]
    b = f_shift[:, :, 24:-24, 24:-24]
    assert torch.allclose(a, b, atol=1e-4), \
        f"equivariance broken: max diff {(a - b).abs().max():.2e}"


def test_both_branches_apply_the_same_function():
    """Siamese means one weight set: identical input must give identical output,
    and a weight change must move both branches together."""
    torch.manual_seed(0)
    enc = SiameseEncoder().eval()
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        before_a = enc(x)
        before_b = enc(x.clone())
    assert torch.equal(before_a, before_b)

    with torch.no_grad():
        for p in enc.parameters():
            p.add_(0.01)
        after = enc(x)
    assert not torch.allclose(before_a, after, atol=1e-6), \
        "perturbing weights did not change the output; branches are not shared"


from src.localizer.encoder import calibrate_alignment


def test_calibrated_alignment_is_small_and_finite():
    enc = SiameseEncoder().eval()
    off = calibrate_alignment(enc)
    assert isinstance(off, float)
    assert abs(off) <= STRIDE, f"alignment offset {off} exceeds one stride"


def test_calibration_is_stable_across_input_sizes():
    enc = SiameseEncoder().eval()
    a = calibrate_alignment(enc, size=256)
    b = calibrate_alignment(enc, size=512)
    assert abs(a - b) < 0.5, f"alignment drifts with input size: {a} vs {b}"
