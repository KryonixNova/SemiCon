"""SuperPoint + LightGlue feature extraction and matching, GPU-resident.

Install: pip install "git+https://github.com/cvg/LightGlue.git"
Weights download automatically (via torch.hub) on first use and are cached
in ~/.cache/torch/hub/checkpoints/ afterward -- first construction of
SuperPointLightGlueMatcher requires network access; subsequent runs do not.
"""
from dataclasses import dataclass

import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd


@dataclass
class MatchResult:
    kpts_a: np.ndarray   # (N, 2) matched keypoint xy coords in image a, float32
    kpts_b: np.ndarray   # (N, 2) matched keypoint xy coords in image b, float32
    scores: np.ndarray   # (N,) LightGlue confidence per match, float32


class SuperPointLightGlueMatcher:
    def __init__(self, device: str = None, max_num_keypoints: int = 2048):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    @torch.no_grad()
    def extract(self, img: np.ndarray) -> dict:
        """SuperPoint keypoints + descriptors for one grayscale uint8 image."""
        tensor = numpy_image_to_torch(img).to(self.device)
        return self.extractor.extract(tensor)

    @torch.no_grad()
    def match(self, feats_a: dict, feats_b: dict) -> MatchResult:
        """LightGlue correspondences between two extracted feature sets."""
        result = self.matcher({"image0": feats_a, "image1": feats_b})
        feats_a, feats_b, result = [rbd(x) for x in (feats_a, feats_b, result)]
        idx = result["matches"]
        kpts_a = feats_a["keypoints"][idx[:, 0]].detach().cpu().numpy().astype(np.float32)
        kpts_b = feats_b["keypoints"][idx[:, 1]].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        return MatchResult(kpts_a=kpts_a, kpts_b=kpts_b, scores=scores)
