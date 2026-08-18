"""Every tunable quantity from the design spec, in one place.

Nothing here is a magic number buried in code: each field is either fixed
by an argued constraint (encoder_stride) or is swept/calibrated by an
experiment described in the spec.
"""

from dataclasses import dataclass, asdict


@dataclass
class LocalizerConfig:
    # --- architecture (fixed by the aliasing argument in the spec) ---
    encoder_stride: int = 4
    corr_channels: int = 128
    context_channels: int = 64
    context_dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 1, 1)

    # --- targets ---
    heatmap_sigma_cells: float = 2.0        # swept over {1, 2, 3}

    # --- losses ---
    lambda_offset: float = 1.0
    lambda_hard_negative: float = 0.5       # swept {0, 0.25, 0.5, 1.0}; 0 = M2 ablation
    hard_negative_radius_cells: int = 24    # swept {6, 12, 24, 48}; 24 selected by Task 17's sweep
    focal_alpha: float = 2.0
    focal_beta: float = 4.0

    # --- decoding ---
    peak_tie_ratio: float = 0.98            # calibrated on val, sweep [0.80, 1.00]
    nms_kernel: int = 3

    # --- optimisation ---
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_steps: int = 40000

    # --- data (canvas-disjoint split seed ranges) ---
    train_seed_lo: int = 0
    train_seed_hi: int = 100_000
    val_seed_lo: int = 100_000
    val_seed_hi: int = 100_500
    test_seed_lo: int = 200_000
    test_seed_hi: int = 200_500
    crops_per_canvas: int = 100

    def as_dict(self) -> dict:
        return asdict(self)
