"""End-to-end localizer.

reference (1,100,100) + search (1,1000,1000)
  -> shared encoder            -> (128,25,25) and (128,250,250)
  -> dense depthwise xcorr     -> (128,226,226)
  -> context head              -> heatmap (226,226) + offset (2,226,226)
  -> decode                    -> x, y, confidence

use_context=False builds the B2 ablation used at milestone M1: the channel
dimension is summed away and the offset field is zero, so the model reduces
to a learned-feature correlation with argmax decoding. Keeping it in the same
class guarantees M1 and M2 differ by exactly one component.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.localizer.config import LocalizerConfig
from src.localizer.context_head import ContextHead
from src.localizer.correlation import dense_correlation
from src.localizer.decode import decode
from src.localizer.encoder import SiameseEncoder, calibrate_alignment


class DriftSenseLocalizer(nn.Module):
    def __init__(self, config: LocalizerConfig | None = None,
                 use_context: bool = True):
        super().__init__()
        self.config = config or LocalizerConfig()
        self.use_context = use_context
        self.encoder = SiameseEncoder(out_channels=self.config.corr_channels)
        self.context = (
            ContextHead(in_channels=self.config.corr_channels,
                        mid_channels=self.config.context_channels,
                        dilations=self.config.context_dilations)
            if use_context else None
        )
        # B2 ablation: a small trainable affine recalibrates the raw
        # channel-summed correlation into a numerically sane logit range,
        # mirroring the context head's -4.0 bias-init trick (Task 12). The
        # raw sum of 128 depthwise-correlated, L2-normalised-as-a-whole-
        # vector channels lands in the hundreds by construction, which
        # saturates sigmoid to exactly 0 or 1 in float32 and kills all
        # gradient before it reaches the encoder. Scale starts at
        # 1/corr_channels and bias at -4.0 so training starts in the
        # "confidently negative, not saturated" regime with real gradient;
        # both stay trainable so the model can recalibrate as the encoder's
        # feature magnitudes evolve during training.
        self.b2_scale = nn.Parameter(torch.tensor(1.0 / self.config.corr_channels))
        self.b2_bias = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer("align_offset", torch.zeros(()))

    def calibrate(self) -> float:
        """Measure and store the encoder's cell/pixel alignment offset."""
        off = calibrate_alignment(self.encoder)
        self.align_offset.fill_(off)
        return off

    def forward(self, reference: torch.Tensor, search: torch.Tensor):
        corr = dense_correlation(self.encoder(search), self.encoder(reference))
        if self.context is not None:
            return self.context(corr)
        # B2 ablation: collapse channels, no learned offset.
        heatmap = corr.sum(dim=1) * self.b2_scale + self.b2_bias
        offset = torch.zeros(corr.shape[0], 2, *heatmap.shape[-2:],
                             device=corr.device, dtype=corr.dtype)
        return heatmap, offset

    @torch.no_grad()
    def predict(self, reference: torch.Tensor, search: torch.Tensor) -> dict:
        logits, offset = self(reference, search)
        # Both the context and B2 paths produce raw sigmoid logits -- that is
        # the entire point of B2's b2_scale/b2_bias calibration -- so there
        # is no branching here: every mode goes through sigmoid before
        # decode. (A prior version skipped sigmoid for B2, which happened to
        # work only by accident, back when the uncalibrated logits were
        # always positive (~540-563) by construction. Once B2 was calibrated
        # to a proper logit range, b2_bias starts at -4.0 and is driven more
        # negative by the focal loss as training progresses -- exactly as
        # intended. Un-sigmoided, that makes every raw logit negative,
        # which fails decode.py's `peaks > 0` gate on every single cell and
        # collapses every prediction to the (x=50, y=50) fallback regardless
        # of input. Sigmoid must always be applied.)
        heatmap = torch.sigmoid(logits)
        return decode(heatmap, offset,
                      peak_tie_ratio=self.config.peak_tie_ratio,
                      nms_kernel=self.config.nms_kernel,
                      align_offset=float(self.align_offset))
