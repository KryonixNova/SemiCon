"""Context / disambiguation head -- the core contribution.

A DRAM mat is a periodic lattice, so a template window and its lattice
translate are pixel-identical and the correlation surface has exactly-tied
maxima. No local receptive field can break that tie; the required information
lives in the global arrangement of peaks and in where the lattice terminates
at mat boundaries.

This head therefore needs a receptive field spanning the entire 226x226
correlation map. Dilations (1,2,4,8,16,32,64) reach RF 255 in seven layers;
the trailing dilation-1 layers add 4 more and suppress the gridding artefacts
that pure exponential dilation stacks produce. Total RF = 259 >= 226.

Resolution is preserved throughout (all strides 1, no pooling) because the
offset head reads full-resolution features.

All dilated convs use `padding_mode="replicate"` rather than PyTorch's
zero-padding default. With zero-padding, a border cell's receptive field
structurally contains more synthetic zeros than an interior cell's -- a
content-independent statistical difference present from initialization.
Under focal loss's heavy negative-cell pressure this is a cheap,
content-independent shortcut for gradient descent to exploit (observed in
real M2 training: predictions collapsed to a fixed corner regardless of
input). `replicate` padding removes the zero/non-zero discontinuity at the
border so cells there aren't statistically distinguishable from interior
cells based on padding alone. (`reflect` was considered and rejected: it
requires `padding < input_size`, which the dilation=64 layer's padding=64
violates for inputs like the 64x64 case this module is tested against.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 1, 1)


def theoretical_rf(dilations, kernel: int = 3) -> int:
    """RF of a stack of stride-1 dilated convs: 1 + sum (k-1)*dilation."""
    return 1 + sum((kernel - 1) * d for d in dilations)


class _DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=dilation,
                              dilation=dilation, padding_mode="replicate",
                              bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))    # residual


class ContextHead(nn.Module):
    def __init__(self, in_channels: int = 128, mid_channels: int = 64,
                 dilations=DEFAULT_DILATIONS):
        super().__init__()
        self.dilations = tuple(dilations)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_DilatedBlock(mid_channels, d) for d in self.dilations]
        )
        self.heatmap_head = nn.Conv2d(mid_channels, 1, 1)
        self.offset_head = nn.Conv2d(mid_channels, 2, 1)
        # Focal loss expects most cells to start as confident negatives.
        nn.init.constant_(self.heatmap_head.bias, -4.0)

    @property
    def receptive_field(self) -> int:
        return theoretical_rf(self.dilations)

    def forward(self, corr: torch.Tensor):
        f = self.blocks(self.reduce(corr))
        return self.heatmap_head(f).squeeze(1), self.offset_head(f)


def effective_receptive_field(head: "ContextHead", size: int = 226) -> float:
    """Gradient-based effective RF, in cells.

    Backpropagates from the single centre output unit and returns the width of
    the smallest centred square containing 95% of input-gradient magnitude.

    The probe input must NOT be all-zero. `reduce` is conv(bias=False) ->
    GroupNorm -> ReLU with no skip connection bypassing the ReLU, and every
    `_DilatedBlock` after it reproduces exactly 0 for an exactly-0 input. An
    all-zero input therefore drives every activation in the stack to exactly
    0, and ReLU's backward rule (`grad_input = grad_output * (input > 0)`) is
    exactly 0 at input == 0 -- so the gradient is identically zero everywhere,
    the 95%-threshold loop below never fires, and this function silently
    falls through to `return float(size)`, i.e. always reports the saturated
    fallback regardless of the network's true receptive field. Verified: a
    torch.zeros(...) probe gives 226.0 (== size) unconditionally; a small
    random probe gives a real, non-saturated measurement (~191 for the
    default architecture).
    """
    head = head.eval()
    torch.manual_seed(0)
    x = torch.randn(1, head.reduce[0].in_channels, size, size) * 0.01
    x.requires_grad_(True)
    hm, _ = head(x)
    hm[0, size // 2, size // 2].backward()

    g = x.grad.abs().sum(dim=(0, 1))
    total = g.sum().clamp_min(1e-12)
    mid = size // 2
    for half in range(1, size // 2 + 1):
        window = g[max(mid - half, 0):mid + half + 1,
                   max(mid - half, 0):mid + half + 1]
        if float(window.sum() / total) >= 0.95:
            return float(2 * half + 1)
    return float(size)
