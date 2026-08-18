import pytest
import torch

from src.localizer.correlation import dense_correlation


def test_output_shape_is_valid_mode():
    s = torch.randn(2, 8, 250, 250)
    r = torch.randn(2, 8, 25, 25)
    assert dense_correlation(s, r).shape == (2, 8, 226, 226)


def test_channels_are_preserved_not_collapsed():
    out = dense_correlation(torch.randn(1, 128, 50, 50), torch.randn(1, 128, 25, 25))
    assert out.shape[1] == 128


def test_peak_is_at_the_embedded_location():
    torch.manual_seed(0)
    s = torch.zeros(1, 4, 40, 40)
    r = torch.randn(1, 4, 8, 8)
    s[:, :, 12:20, 5:13] = r                       # plant the template
    out = dense_correlation(s, r).sum(dim=1)       # collapse only for the check
    peak = torch.argmax(out.flatten())
    row, col = divmod(int(peak), out.shape[-1])
    assert (row, col) == (12, 5)


def test_batch_items_are_independent():
    s = torch.randn(2, 3, 30, 30)
    r = torch.randn(2, 3, 10, 10)
    both = dense_correlation(s, r)
    first = dense_correlation(s[:1], r[:1])
    assert torch.allclose(both[:1], first, atol=1e-5)


def test_oversized_reference_raises():
    # A reference feature map bigger than the search feature map along
    # either spatial axis makes conv2d's valid-mode correlation undefined
    # (negative output size). Catch it here with a clear message instead of
    # letting it fail deep inside decode.py with an opaque tensor-size error.
    s = torch.randn(1, 4, 20, 20)
    r_too_tall = torch.randn(1, 4, 25, 10)
    with pytest.raises(AssertionError, match="oversized reference"):
        dense_correlation(s, r_too_tall)

    r_too_wide = torch.randn(1, 4, 10, 25)
    with pytest.raises(AssertionError, match="oversized reference"):
        dense_correlation(s, r_too_wide)
