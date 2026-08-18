"""Siamese feature encoder.

ResNet-18 stem + layer1 + layer2, with layer2's stride replaced by dilation
to give output stride 4 and 128 channels. layer3/layer4 are dropped: the task
needs spatial resolution, not the class-level abstraction those stages build.

Stride 4 means one feature cell = 4 search px = 40 nm, which preserves the
64-240 nm DRAM pitch range; stride 8 would begin aliasing the dense presets.

Hard constraint: the network must be translation-equivariant. No global
pooling, no positional encoding. Weights are shared across both branches by
construction -- it is one module called twice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SiameseEncoder(nn.Module):
    def __init__(self, out_channels: int = 128):
        super().__init__()
        m = resnet18(weights=None)
        # Grayscale input: replace the 3-channel stem.
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.body = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2
        )
        # Destride layer2 (index 5) -> overall stride 4 instead of 8, using
        # dilation so the receptive field is preserved.
        for mod in self.body[5].modules():
            if isinstance(mod, nn.Conv2d):
                if mod.stride == (2, 2):
                    mod.stride = (1, 1)
                if mod.kernel_size == (3, 3):
                    mod.dilation, mod.padding = (2, 2), (2, 2)
        # Also destride the downsample shortcut, or the residual add breaks.
        for mod in self.body[5].modules():
            if isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
                mod.stride = (1, 1)
        assert out_channels == 128, "layer2 of ResNet-18 emits 128 channels"
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.body(x), p=2.0, dim=1)


@torch.no_grad()
def calibrate_alignment(encoder: "SiameseEncoder", size: int = 256) -> float:
    """Measure the constant offset between feature-cell index and input pixel.

    Feeds a single-pixel impulse at a known input location, finds the centroid
    of the resulting feature response, and returns
    `impulse_px - STRIDE * response_centroid_cell`.

    Returns a value in Search-image pixels, to be passed as `align_offset` to
    geometry.cell_to_pixel / pixel_to_cell.
    """
    from src.localizer.geometry import STRIDE

    was_training = encoder.training
    encoder.eval()
    centre = size // 2

    base = torch.zeros(1, 1, size, size, device=next(encoder.parameters()).device)
    impulse = base.clone()
    impulse[0, 0, centre, centre] = 1.0

    # Difference isolates the impulse's contribution from padding/bias effects.
    resp = (encoder.body(impulse) - encoder.body(base)).abs().sum(dim=1)[0]

    weights = resp / resp.sum().clamp_min(1e-12)
    idx = torch.arange(resp.shape[-1], dtype=torch.float32, device=resp.device)
    cx = float((weights.sum(dim=0) * idx).sum())
    cy = float((weights.sum(dim=1) * idx).sum())

    if was_training:
        encoder.train()
    return float(centre - STRIDE * (cx + cy) / 2.0)
