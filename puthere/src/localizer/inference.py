"""Shared single-pair inference: checkpoint loading + prediction, used by
both scripts/predict.py's CLI and localize.py's batch mode so the two never
diverge in how a reference/search pair is turned into a prediction.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from src.localizer.config import LocalizerConfig
from src.localizer.geometry import REF_DS_PX, SEARCH_PX
from src.localizer.model import DriftSenseLocalizer


def load_standardized(path: str, target_px: int | None = None) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not read {path}")
    if target_px is not None and img.shape[0] != target_px:
        img = cv2.resize(img, (target_px, target_px), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(img)).float()
    t = (t - t.mean()) / t.std().clamp_min(1e-6)
    return t.unsqueeze(0).unsqueeze(0)


def load_model(checkpoint_path: str, device: str) -> DriftSenseLocalizer:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = DriftSenseLocalizer(LocalizerConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.align_offset.fill_(ckpt["align_offset"])
    model.eval()
    return model


def predict_pair(model: DriftSenseLocalizer, reference_path: str, search_path: str,
                  device: str) -> dict:
    """Returns {"x": float, "y": float, "confidence": float}.

    The reference is deliberately force-resized to REF_DS_PX regardless of
    its input size -- that's intentional (matches how the model was
    trained: reference crops are always downsampled to REF_DS_PX). It
    implicitly assumes the reference is already at the correct nm/px scale
    *before* this resize; a reference captured at a genuinely different
    physical scale will be silently misinterpreted (resized to the right
    pixel count without ever being rescaled to the right physical extent).

    Unlike the reference, the search image is NOT auto-resized: a wrong
    size means the physical scale is likely wrong too, so this raises
    ValueError rather than silently resizing.
    """
    ref = load_standardized(reference_path, REF_DS_PX).to(device)
    search = load_standardized(search_path).to(device)
    if search.shape[-2:] != (SEARCH_PX, SEARCH_PX):
        raise ValueError(
            f"search image must be {SEARCH_PX}x{SEARCH_PX} px, got "
            f"{search.shape[-2]}x{search.shape[-1]} px ({search_path}). "
            f"Unlike the reference, the search image is not auto-resized: "
            f"a wrong size means the physical scale is likely wrong too."
        )
    with torch.no_grad():
        out = model.predict(ref, search)
    return {"x": float(out["x"][0]), "y": float(out["y"][0]),
            "confidence": float(out["confidence"][0])}
