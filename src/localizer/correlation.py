"""Dense cross-correlation between reference and search feature maps.

One grouped-convolution call evaluates every candidate offset -- all 51,076
of them at stride-4 granularity -- replacing the 100-patch grid comparison
with the exact quantity it was approximating, more cheaply.

Correlation is kept CHANNEL-WISE (depthwise), not collapsed to a scalar
similarity. A scalar score discards precisely the information needed to
distinguish a true match from a periodic twin, since both correlate ~0.9.
The context head consumes the full channel volume.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dense_correlation(search_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
    """Batched depthwise cross-correlation in valid mode.

    Folds the batch into the channel dimension so a single grouped conv2d
    handles every (sample, channel) pair independently.
    """
    b, c, hs, ws = search_feat.shape
    br, cr, hr, wr = ref_feat.shape
    assert (b, c) == (br, cr), "search and reference must share batch and channels"
    assert hr <= hs and wr <= ws, (
        f"reference feature map ({hr}x{wr}) must not exceed the search "
        f"feature map ({hs}x{ws}); got an oversized reference"
    )

    out = F.conv2d(
        search_feat.reshape(1, b * c, hs, ws),
        ref_feat.reshape(b * c, 1, hr, wr),
        groups=b * c,
    )
    return out.reshape(b, c, hs - hr + 1, ws - wr + 1)
