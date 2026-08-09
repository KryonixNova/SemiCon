"""Integration regression for the corner-bias/collapse bug family.

Four separate corner-bias/collapse bugs were found on this branch over its
development (see model.py's `predict()` docstring and test_model.py's
`test_b2_predict_does_not_collapse_once_bias_goes_negative`, and
context_head.py / test_context_head.py's replicate-pad corner-collapse
fix). Every one of them was only ever caught by a human actually running
training or inference and noticing an odd number -- never by a shape-only
unit test, since a collapsed model still produces tensors of the right
shape. This runs a handful of real optimizer steps through the actual
encoder -> correlation -> context head -> loss stack (not mocked) on random
tensors and checks, mechanically, the things a human previously had to
notice by eye:
  (a) the loss stays finite throughout (also regression-tests the fp16-AMP
      NaN bug in focal_heatmap_loss, see test_losses.py),
  (b) the loss trends downward / does not diverge over a few steps,
  (c) decoded predictions for genuinely different random inputs in the same
      batch are not all identical (the collapse signature itself).
"""

from __future__ import annotations

import pytest
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.losses import focal_heatmap_loss, offset_loss
from src.localizer.model import DriftSenseLocalizer
from src.localizer.targets import build_targets


@pytest.mark.slow
def test_a_few_optimizer_steps_stay_finite_and_do_not_collapse():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = LocalizerConfig()
    model = DriftSenseLocalizer(cfg).to(device)
    align = model.calibrate()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    b, n_steps = 4, 8
    losses = []
    for step in range(n_steps):
        ref = torch.randn(b, 1, 100, 100, device=device)
        search = torch.randn(b, 1, 1000, 1000, device=device)
        gt_x = torch.empty(b, device=device).uniform_(60.0, 940.0)
        gt_y = torch.empty(b, device=device).uniform_(60.0, 940.0)
        tgt = build_targets(gt_x, gt_y, sigma_cells=cfg.heatmap_sigma_cells,
                            align_offset=align)
        hm_t = tgt["heatmap"].to(device)
        off_t = tgt["offset"].to(device)
        peak = tgt["peak_cell"].to(device)

        logits, offset = model(ref, search)
        loss = focal_heatmap_loss(logits, hm_t, cfg.focal_alpha, cfg.focal_beta)
        loss = loss + cfg.lambda_offset * offset_loss(offset, off_t, peak)

        # (a) finite throughout -- also regression-tests the fp16-AMP NaN
        # bug in focal_heatmap_loss (test_losses.py covers that directly;
        # this covers it end-to-end through the real model).
        assert torch.isfinite(loss), f"non-finite loss at step {step}: {loss}"

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())

    # (b) trends downward / does not diverge. A handful of steps on random,
    # unrelated-per-step tensors is noisy, so compare the mean of the first
    # half against the second half rather than requiring a strict
    # step-by-step decrease.
    first_half = sum(losses[:n_steps // 2]) / (n_steps // 2)
    second_half = sum(losses[n_steps // 2:]) / (n_steps // 2)
    assert second_half <= first_half * 1.5, (
        f"loss diverged over {n_steps} steps: {losses}"
    )

    # (c) predictions for genuinely different inputs must not all collapse
    # to one identical point -- the exact signature of the four corner-bias
    # bugs found on this branch.
    model.eval()
    with torch.no_grad():
        ref = torch.randn(b, 1, 100, 100, device=device)
        search = torch.randn(b, 1, 1000, 1000, device=device)
        out = model.predict(ref, search)

    xs, ys = out["x"], out["y"]
    all_identical = all(
        torch.allclose(xs[i], xs[0]) and torch.allclose(ys[i], ys[0])
        for i in range(1, b)
    )
    assert not all_identical, (
        "predictions collapsed to an identical (x, y) for different inputs"
    )
