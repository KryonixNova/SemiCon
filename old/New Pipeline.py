"""DRAM SEM-style localization pipeline -- single-file build.

Everything needed to generate synthetic DRAM SEM image pairs and benchmark
SuperPoint+LightGlue localization against them, combined into one file:

  1. GDS layout generation      (was dram_layout_generator.py)
  2. SEM-style rasterization    (was sem_render.py)
  3. Reference/search sampling  (was reference_search_pairs.py)
  4. SuperPoint+LightGlue match (was matcher.py)
  5. RANSAC localization        (was localizer.py)
  6. Benchmark CLI              (was benchmark.py)

The original modular files still exist alongside this one and are what the
test suite (test_*.py) imports against -- this file is a standalone,
dependency-ordered concatenation for anyone who wants a single script.
Run it directly: `python dram_localization_pipeline.py --n 30`.

Why each layout structure is not proprietary:
  Capacitor nodes  - 1T-1C concept, Dennard (1968), expired patent
  Word lines       - Sze & Ng (2006) Physics of Semiconductor Devices Sec.6
  Bit lines        - Razavi (2002) Design of Analog CMOS ICs Sec.12
  Contacts         - Jaeger (2002) Intro to Microelectronic Fabrication Ch.6
  Defects          - Postek & Vladar (2011) SEM metrology literature
  Dummy cells      - JEDEC JESD79 array margin rules

Cell geometry is fully abstracted: cell_pitch_nm is a free parameter with
no F^2 constraint. No real process node is implied.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import klayout.db as db
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd


# ═════════════════════════════════════════════════════════════════════════
# 1. GDS layout generation (dram_layout_generator.py)
# ═════════════════════════════════════════════════════════════════════════

# ── GDS layer constants ────────────────────────────────────────────────────
LAYER_CAPACITOR = (1, 0)   # storage node rectangles
LAYER_WORDLINE = (2, 0)    # horizontal WL stripes
LAYER_BITLINE = (3, 0)     # vertical BL stripes
LAYER_CONTACT = (4, 0)     # small square contacts
LAYER_DEFECT = (5, 0)      # particles, scratches, dishing
LAYER_DUMMY = (6, 0)       # boundary dummy fills
LAYER_STREET = (7, 0)     # gap/scribe-line rectangles between die blocks


@dataclass(frozen=True)
class DRAMParams:
    """All generator parameters. Frozen -- never mutated after construction."""

    # ── Array geometry ──────────────────────────────────────────────────────
    rows: int = 64
    cols: int = 64
    cell_pitch_nm: float = 80.0       # WL-direction pitch (nm)
    cell_pitch_bl_nm: float = 60.0    # BL-direction pitch (nm)
    wl_width_nm: float = 24.0         # word-line stripe width
    bl_width_nm: float = 18.0         # bit-line stripe width
    contact_size_nm: float = 14.0     # square contact side length
    capacitor_size_nm: float = 30.0   # square storage-node side length
    dummy_rows: int = 2               # dummy boundary rows per side
    dummy_cols: int = 2               # dummy boundary cols per side

    # ── Process variation sigmas ────────────────────────────────────────────
    overlay_sigma_nm: float = 3.0     # layer-to-layer misalignment sigma
    linewidth_sigma_nm: float = 2.0   # CD variation sigma (half applied per edge)
    ler_amplitude_nm: float = 1.5     # line-edge roughness amplitude
    ler_period_nm: float = 20.0       # LER spatial correlation length
    contact_sigma_frac: float = 0.08  # fractional sigma of contact CD
    jitter_sigma_nm: float = 1.0      # cell-placement jitter sigma

    # ── Defect probabilities ────────────────────────────────────────────────
    p_missing_contact: float = 0.005  # Bernoulli prob per contact
    p_missing_capacitor: float = 0.003
    p_broken_wl: float = 0.02         # Bernoulli prob per word line
    p_broken_bl: float = 0.02         # Bernoulli prob per bit line
    n_particles: int = 3              # number of particle defects in array
    n_scratches: int = 2              # number of scratch defects
    p_cmp_dishing: float = 0.10       # fraction of array radius affected

    # ── Output ─────────────────────────────────────────────────────────────
    output_gds: str = "dram.gds"
    output_json: str = "dram_metadata.json"
    seed: int = 42
    dbu_nm: float = 1.0               # 1 DBU = 1 nm (layout.dbu = dbu_nm*1e-3 um)

    # ── Rasterization / localization benchmark ──────────────────────────────
    output_width_px: int = 1000       # search image width in pixels
    output_height_px: int = 1000      # search image height in pixels
    reference_width_px: int = 100     # reference crop width in pixels
    reference_height_px: int = 100    # reference crop height in pixels
    zoom_ratio: int = 10              # output_width_px / reference_width_px
    pixels_per_nm: Optional[float] = None
    # If None, auto-computed: min(output_width_px/array_w_nm,
    #                             output_height_px/array_h_nm)

    # ── Multi-region die layout ──────────────────────────────────────────────
    # die_block_rows=die_block_cols=1 (the default) is an exact passthrough:
    # DieGenerator behaves identically to constructing DRAMGenerator directly.
    die_block_rows: int = 1              # blocks stacked vertically
    die_block_cols: int = 1              # blocks side-by-side
    die_block_width_nm: float = 3840.0   # per-block physical footprint (= 64*60nm, today's array width)
    die_block_height_nm: float = 5120.0  # per-block physical footprint (= 64*80nm, today's array height)
    die_street_width_nm: float = 40.0    # gap rendered between adjacent blocks
    die_block_variation_frac: float = 0.4   # +/- bound on per-block parameter jitter

    # ── Single training defect ────────────────────────────────────────────
    # Opt-in: when True, exactly one small diagonal "training_scratch"
    # defect is placed on the die (n_particles/n_scratches are forced to 0
    # so it's the only defect present), for generating labeled
    # defect-detection training data. False (the default) is an exact
    # passthrough -- output is unaffected.
    single_training_defect: bool = False
    training_defect_size_px: int = 10   # rendered footprint, in search-image px


@dataclass(frozen=True)
class RasterizationConfig:
    """
    Everything a rasterizer needs to convert the GDS to a 1000x1000 image.
    Returned by DRAMGenerator.get_rasterization_config(). No rasterization
    is performed here -- this is a pure parameter bundle.
    """
    gds_file: str
    pixels_per_nm: float
    origin_nm: Tuple[float, float]        # (x, y) nm coords of image pixel (0,0)
    width_px: int                          # 1000
    height_px: int                         # 1000
    layers: List[Tuple[int, int]]          # GDS (layer, datatype) pairs
    reference_bbox_px: List[int]           # [x0, y0, x1, y1] ground-truth crop
    reference_center_px: List[float]       # [cx, cy] ground-truth label


# ── Coordinate conversion utilities ───────────────────────────────────────────

def nm_to_pixel(x_nm: float, pixels_per_nm: float) -> float:
    """Convert a scalar nm coordinate to pixel coordinate."""
    return x_nm * pixels_per_nm


def pixel_to_nm(x_px: float, pixels_per_nm: float) -> float:
    """Convert a scalar pixel coordinate to nm."""
    return x_px / pixels_per_nm


def bbox_nm_to_pixel(bbox_nm: List[float], pixels_per_nm: float) -> List[float]:
    """Convert [x0, y0, x1, y1] bbox from nm to pixel coords."""
    return [v * pixels_per_nm for v in bbox_nm]


def bbox_pixel_to_nm(bbox_px: List[float], pixels_per_nm: float) -> List[float]:
    """Convert [x0, y0, x1, y1] bbox from pixel to nm coords."""
    return [v / pixels_per_nm for v in bbox_px]


class ProcessVariation:
    """
    Pure geometric perturbation methods -- no KLayout cell/layout calls.
    Each method takes db.Box input(s) and returns perturbed db.Box output(s).
    All randomness comes from the seeded rng; results are deterministic.
    """

    def __init__(self, params: DRAMParams, rng: np.random.Generator) -> None:
        self.p = params
        self.rng = rng

    def overlay_shift(self, box: db.Box) -> db.Box:
        """Shift box by Gaussian(0, overlay_sigma_nm) in x and y."""
        dx = int(round(self.rng.normal(0, self.p.overlay_sigma_nm)))
        dy = int(round(self.rng.normal(0, self.p.overlay_sigma_nm)))
        return box.moved(dx, dy)

    def vary_linewidth(self, box: db.Box, axis: str) -> db.Box:
        """
        Perturb line CD by Gaussian(0, linewidth_sigma_nm).
        axis='x': line is horizontal -- vary height (y dimension).
        axis='y': line is vertical   -- vary width  (x dimension).
        """
        delta = int(round(self.rng.normal(0, self.p.linewidth_sigma_nm / 2)))
        if axis == 'x':
            new_bot = box.bottom + delta
            new_top = max(new_bot + 1, box.top - delta)
            return db.Box(box.left, new_bot, box.right, new_top)
        else:
            new_left = box.left + delta
            new_right = max(new_left + 1, box.right - delta)
            return db.Box(new_left, box.bottom, new_right, box.top)

    def add_ler(self, box: db.Box, axis: str,
                n_seg: Optional[int] = None) -> List[db.Box]:
        """
        Approximate line-edge roughness by slicing the box into n_seg
        sub-rectangles and perturbing each edge independently.
        axis='x': horizontal line -- perturb top/bottom edges per segment.
        axis='y': vertical line   -- perturb left/right edges per segment.
        """
        amp = max(1, int(round(self.p.ler_amplitude_nm)))
        if axis == 'x':
            total = box.right - box.left
            n = n_seg or max(4, int(total / max(1, self.p.ler_period_nm)))
            seg = max(1, total // n)
            boxes: List[db.Box] = []
            for k in range(n):
                x0 = box.left + k * seg
                x1 = min(box.right, x0 + seg)
                dt = int(round(self.rng.normal(0, amp)))
                db_ = int(round(self.rng.normal(0, amp)))
                y0 = box.bottom + db_
                y1 = max(y0 + 1, box.top + dt)
                boxes.append(db.Box(x0, y0, x1, y1))
            return boxes
        else:
            total = box.top - box.bottom
            n = n_seg or max(4, int(total / max(1, self.p.ler_period_nm)))
            seg = max(1, total // n)
            boxes = []
            for k in range(n):
                y0 = box.bottom + k * seg
                y1 = min(box.top, y0 + seg)
                dl = int(round(self.rng.normal(0, amp)))
                dr = int(round(self.rng.normal(0, amp)))
                x0 = box.left + dl
                x1 = max(x0 + 1, box.right + dr)
                boxes.append(db.Box(x0, y0, x1, y1))
            return boxes

    def vary_contact(self, box: db.Box) -> db.Box:
        """Scale contact CD by Gaussian(1, contact_sigma_frac), keep centred."""
        cx = (box.left + box.right) // 2
        cy = (box.bottom + box.top) // 2
        side = box.width()
        scale = float(self.rng.normal(1.0, self.p.contact_sigma_frac))
        new_half = max(1, int(round(side * scale / 2)))
        return db.Box(cx - new_half, cy - new_half,
                      cx + new_half, cy + new_half)

    def jitter_cell(self, cx: float, cy: float) -> Tuple[float, float]:
        """Add Gaussian(0, jitter_sigma_nm) displacement to cell centre."""
        dx = float(self.rng.normal(0, self.p.jitter_sigma_nm))
        dy = float(self.rng.normal(0, self.p.jitter_sigma_nm))
        return cx + dx, cy + dy

    def is_missing(self, prob: float) -> bool:
        """Return True with probability prob (Bernoulli draw)."""
        return bool(self.rng.random() < prob)

    def break_line(self, box: db.Box, axis: str) -> List[db.Box]:
        """
        Remove a random sub-segment from the line, returning two pieces.
        axis='x': horizontal -- remove a x-range sub-segment.
        axis='y': vertical   -- remove a y-range sub-segment.
        """
        if axis == 'x':
            span = box.right - box.left
            gap_start = int(self.rng.integers(span // 4, span // 2))
            gap_end   = int(self.rng.integers(span // 2, 3 * span // 4))
            piece_a = db.Box(box.left,          box.bottom,
                             box.left + gap_start, box.top)
            piece_b = db.Box(box.left + gap_end, box.bottom,
                             box.right,            box.top)
        else:
            span = box.top - box.bottom
            gap_start = int(self.rng.integers(span // 4, span // 2))
            gap_end   = int(self.rng.integers(span // 2, 3 * span // 4))
            piece_a = db.Box(box.left, box.bottom,
                             box.right, box.bottom + gap_start)
            piece_b = db.Box(box.left, box.bottom + gap_end,
                             box.right, box.top)
        return [p for p in [piece_a, piece_b] if p.width() > 0 and p.height() > 0]

    def add_particles(self, bbox: db.Box, n: int) -> List[db.Box]:
        """Place n random rectangular particle defects inside bbox."""
        particles = []
        for _ in range(n):
            side_x = int(self.rng.integers(5, 41))
            side_y = int(self.rng.integers(5, 41))
            x0 = int(self.rng.integers(bbox.left, max(bbox.left + 1,
                                                       bbox.right - side_x)))
            y0 = int(self.rng.integers(bbox.bottom, max(bbox.bottom + 1,
                                                         bbox.top - side_y)))
            particles.append(db.Box(x0, y0, x0 + side_x, y0 + side_y))
        return particles

    def add_scratches(self, bbox: db.Box, n: int) -> List[db.Box]:
        """Place n thin elongated scratch defects (axis-aligned) inside bbox."""
        scratches = []
        for _ in range(n):
            if self.rng.random() < 0.5:  # horizontal scratch
                length = int(self.rng.integers(50, 201))
                width  = int(self.rng.integers(2, 6))
                x0 = int(self.rng.integers(bbox.left,
                                           max(bbox.left + 1, bbox.right - length)))
                y0 = int(self.rng.integers(bbox.bottom,
                                           max(bbox.bottom + 1, bbox.top - width)))
                scratches.append(db.Box(x0, y0, x0 + length, y0 + width))
            else:  # vertical scratch
                length = int(self.rng.integers(50, 201))
                width  = int(self.rng.integers(2, 6))
                x0 = int(self.rng.integers(bbox.left,
                                           max(bbox.left + 1, bbox.right - width)))
                y0 = int(self.rng.integers(bbox.bottom,
                                           max(bbox.bottom + 1, bbox.top - length)))
                scratches.append(db.Box(x0, y0, x0 + width, y0 + length))
        return scratches

    def cmp_dishing(self, boxes: List[db.Box],
                    cx: float, cy: float, r: float) -> List[db.Box]:
        """
        Shrink line width for boxes whose centre falls within radius r of
        dish centre (cx, cy). Models CMP dishing -- a polishing artefact
        that thins lines in a circular region.
        """
        result = []
        for box in boxes:
            bcx = (box.left + box.right) / 2
            bcy = (box.bottom + box.top) / 2
            if (bcx - cx) ** 2 + (bcy - cy) ** 2 < r ** 2:
                # shrink by up to 20% proportional to distance from centre
                dist = ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5
                factor = 1.0 - 0.2 * (1.0 - dist / r)
                if box.width() >= box.height():  # horizontal line
                    new_h = max(1, int(round(box.height() * factor)))
                    mid_y = (box.bottom + box.top) // 2
                    result.append(db.Box(box.left, mid_y - new_h // 2,
                                         box.right, mid_y - new_h // 2 + new_h))
                else:  # vertical line
                    new_w = max(1, int(round(box.width() * factor)))
                    mid_x = (box.left + box.right) // 2
                    result.append(db.Box(mid_x - new_w // 2, box.bottom,
                                         mid_x - new_w // 2 + new_w, box.top))
            else:
                result.append(box)
        return result


_BLOCK_JITTER_GEOMETRY_FIELDS = [
    "cell_pitch_nm", "cell_pitch_bl_nm", "wl_width_nm", "bl_width_nm",
    "contact_size_nm", "capacitor_size_nm",
    "overlay_sigma_nm", "linewidth_sigma_nm", "ler_amplitude_nm", "ler_period_nm",
    "contact_sigma_frac", "jitter_sigma_nm",
]
_BLOCK_JITTER_PROB_FIELDS = [
    "p_missing_contact", "p_missing_capacitor", "p_broken_wl", "p_broken_bl",
    "p_cmp_dishing",
]
_BLOCK_JITTER_COUNT_FIELDS = ["n_particles", "n_scratches"]


def _randomize_block_params(base: DRAMParams, rng: np.random.Generator,
                             frac: float) -> DRAMParams:
    """
    Return a new DRAMParams with every per-block geometry/defect-rate field
    of `base` independently jittered by a uniform factor in [1-frac, 1+frac].
    frac=0.0 returns fields unchanged. Geometry fields are floored at 1e-3nm,
    probability fields clamped to [0, 1], count fields clamped to >= 0.
    rows/cols/dummy_*/output/rasterization/seed/path fields are untouched --
    those are die-level-controlled or global settings, not per-block.
    """
    overrides = {}
    for field in _BLOCK_JITTER_GEOMETRY_FIELDS:
        factor = float(rng.uniform(1.0 - frac, 1.0 + frac))
        overrides[field] = max(1e-3, getattr(base, field) * factor)
    for field in _BLOCK_JITTER_PROB_FIELDS:
        factor = float(rng.uniform(1.0 - frac, 1.0 + frac))
        overrides[field] = min(1.0, max(0.0, getattr(base, field) * factor))
    for field in _BLOCK_JITTER_COUNT_FIELDS:
        factor = float(rng.uniform(1.0 - frac, 1.0 + frac))
        overrides[field] = max(0, int(round(getattr(base, field) * factor)))
    return dataclasses.replace(base, **overrides)


def _place_training_defect(cell: db.Cell, layer: int, pixels_per_nm: float,
                            rng: np.random.Generator, bbox: db.Box,
                            size_px: int) -> dict:
    """
    Insert one square 'training_scratch' defect box at a random location
    inside bbox, sized so its rendered footprint is size_px pixels at
    pixels_per_nm. bbox should cover the full rendered search-image canvas
    (see _canvas_bbox_nm) so every rendered pixel is reachable. Margined so
    the defect's own footprint never clips the rendered edge.
    """
    side_nm = size_px / pixels_per_nm
    half = side_nm / 2.0

    margin = half
    left_lo, left_hi = bbox.left + margin, max(bbox.left + margin + 1.0, bbox.right - margin)
    bottom_lo, bottom_hi = bbox.bottom + margin, max(bbox.bottom + margin + 1.0, bbox.top - margin)
    cx = float(rng.uniform(left_lo, left_hi))
    cy = float(rng.uniform(bottom_lo, bottom_hi))
    box = db.Box(int(round(cx - half)), int(round(cy - half)),
                 int(round(cx + half)), int(round(cy + half)))
    cell.shapes(layer).insert(box)
    return {"type": "training_scratch",
            "bbox_nm": [box.left, box.bottom, box.right, box.top]}


def _canvas_bbox_nm(pixels_per_nm: float, width_px: int, height_px: int,
                     content_top_nm: float) -> db.Box:
    """Full rendered-canvas extent in nm -- covers every pixel the final
    search image can have, including any periodic-lattice margin
    _tile_margin fills when the array/die isn't square.

    The renderer's nm->pixel mapping (_nm_bbox_to_ss_px, used by
    _paint_defects) is unflipped in x -- nm-x 0 is always canvas column 0,
    so x always spans [0, width_px/pixels_per_nm] regardless of content
    size. But it's y-flipped and anchored at the array/die's own top edge
    (content_top_nm, i.e. the "array_h_nm"/"total_h_nm" value that ends up
    in meta["search_bbox_nm"][3] -- NOT the canvas's own top), so the
    reachable y-range is [content_top_nm - height_px/pixels_per_nm,
    content_top_nm], which sits below y=0 whenever the array is wider than
    it is tall (margin gets appended below the content, not above it).
    Passing content_top_nm=0 would silently make the bottom ~content_h_nm
    of tile rows unreachable, which is exactly the class of bug this
    helper exists to avoid -- callers must pass the same top-edge value
    that ends up in meta["search_bbox_nm"][3] for their generator.

    Used to bound single_training_defect placement so every tile of the
    search image's grid is reachable."""
    canvas_w_nm = width_px / pixels_per_nm
    canvas_h_nm = height_px / pixels_per_nm
    return db.Box(0, int(round(content_top_nm - canvas_h_nm)),
                  int(round(canvas_w_nm)), int(round(content_top_nm)))


class DRAMGenerator:
    """
    Orchestrates KLayout geometry construction for a synthetic DRAM array.
    Calls ProcessVariation to perturb shapes, then inserts into the layout.
    """

    def __init__(self, params: DRAMParams) -> None:
        self.p = params

        # Validate rasterization params before any layout work
        if params.output_width_px != 1000:
            raise ValueError("output_width_px must be 1000 for benchmark compatibility")
        if params.output_height_px != 1000:
            raise ValueError("output_height_px must be 1000 for benchmark compatibility")
        if params.reference_width_px != 100:
            raise ValueError("reference_width_px must be 100 for benchmark compatibility")
        if params.reference_height_px != 100:
            raise ValueError("reference_height_px must be 100 for benchmark compatibility")
        if params.zoom_ratio != 10:
            raise ValueError("zoom_ratio must be 10 for benchmark compatibility")

        # Compute rasterization scale
        array_w_nm = params.cols * params.cell_pitch_bl_nm
        array_h_nm = params.rows * params.cell_pitch_nm
        if params.pixels_per_nm is None:
            self._pixels_per_nm: float = min(
                params.output_width_px  / array_w_nm,
                params.output_height_px / array_h_nm,
            )
        else:
            self._pixels_per_nm = params.pixels_per_nm

        self.rng = np.random.default_rng(params.seed)
        self.pv = ProcessVariation(params, self.rng)

        self.layout = db.Layout()
        self.layout.dbu = params.dbu_nm * 1e-3   # 1 DBU = 1 nm (dbu in um)
        self.cell = self.layout.create_cell("DRAM_ARRAY")

        self.L_CAP = self.layout.layer(*LAYER_CAPACITOR)
        self.L_WL  = self.layout.layer(*LAYER_WORDLINE)
        self.L_BL  = self.layout.layer(*LAYER_BITLINE)
        self.L_CON = self.layout.layer(*LAYER_CONTACT)
        self.L_DEF = self.layout.layer(*LAYER_DEFECT)
        self.L_DUM = self.layout.layer(*LAYER_DUMMY)

        # Metadata accumulators (populated by _build_* methods)
        self._cell_centers: List[List[float]] = []
        self._wl_coords: List[dict] = []
        self._bl_coords: List[dict] = []
        self._defect_locs: List[dict] = []

        # Box accumulators for CMP dishing (populated by _build_wordlines/_build_bitlines)
        self._wl_boxes: List[db.Box] = []
        self._bl_boxes: List[db.Box] = []

        # Populated by generate(); used by get_rasterization_config()
        self._last_meta: dict = {}

    # ── Coordinate helpers ──────────────────────────────────────────────────

    def _nm_to_px(self, x_nm: float) -> float:
        return nm_to_pixel(x_nm, self._pixels_per_nm)

    def _px_to_nm(self, x_px: float) -> float:
        return pixel_to_nm(x_px, self._pixels_per_nm)

    def _cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """Nominal (un-jittered) cell centre in nm."""
        cx = col * self.p.cell_pitch_bl_nm + self.p.cell_pitch_bl_nm / 2
        cy = row * self.p.cell_pitch_nm    + self.p.cell_pitch_nm    / 2
        return cx, cy

    def _array_bbox(self) -> db.Box:
        """Bounding box of the active array (no dummy border)."""
        return db.Box(0, 0,
                      int(self.p.cols * self.p.cell_pitch_bl_nm),
                      int(self.p.rows * self.p.cell_pitch_nm))

    def _full_bbox(self) -> db.Box:
        """Bounding box including dummy cells."""
        x_min = int(-self.p.dummy_cols * self.p.cell_pitch_bl_nm)
        y_min = int(-self.p.dummy_rows * self.p.cell_pitch_nm)
        x_max = int((self.p.cols + self.p.dummy_cols) * self.p.cell_pitch_bl_nm)
        y_max = int((self.p.rows + self.p.dummy_rows) * self.p.cell_pitch_nm)
        return db.Box(x_min, y_min, x_max, y_max)

    def _insert(self, layer: int, boxes: Union[db.Box, List[db.Box]]) -> None:
        """Insert one or more db.Box objects into the cell on the given layer."""
        shapes = self.cell.shapes(layer)
        if isinstance(boxes, db.Box):
            shapes.insert(boxes)
        else:
            for b in boxes:
                if b.width() > 0 and b.height() > 0:
                    shapes.insert(b)

    # ── Builder: dummy cells ────────────────────────────────────────────────

    def _build_dummy_cells(self) -> None:
        """
        Place boundary dummy fills on Layer 6 around the active array.
        Dummy cells maintain periodic boundary conditions so edge cells
        see the same electromagnetic environment as interior cells.
        """
        p = self.p
        pitch_x = int(p.cell_pitch_bl_nm)
        pitch_y = int(p.cell_pitch_nm)
        cap_half = int(p.capacitor_size_nm / 2)

        for row in range(-p.dummy_rows, p.rows + p.dummy_rows):
            for col in range(-p.dummy_cols, p.cols + p.dummy_cols):
                # Skip active array interior
                if 0 <= row < p.rows and 0 <= col < p.cols:
                    continue
                cx = int(col * pitch_x + pitch_x // 2)
                cy = int(row * pitch_y + pitch_y // 2)
                box = db.Box(cx - cap_half, cy - cap_half,
                             cx + cap_half, cy + cap_half)
                self._insert(self.L_DUM, box)

    # ── Builder: capacitor nodes ────────────────────────────────────────────

    def _build_capacitors(self) -> List[List[float]]:
        """
        Place storage-node capacitors (Layer 1) for each active cell.
        Applies jitter, overlay shift, and missing-capacitor defects.
        Returns list of [x_nm, y_nm] actual centres (after jitter).
        """
        p = self.p
        cap_half = int(p.capacitor_size_nm / 2)
        centers = []

        for row in range(p.rows):
            for col in range(p.cols):
                nom_cx, nom_cy = self._cell_center(row, col)
                cx, cy = self.pv.jitter_cell(nom_cx, nom_cy)
                centers.append([cx, cy])

                if not p.single_training_defect and self.pv.is_missing(p.p_missing_capacitor):
                    self._defect_locs.append({
                        "type": "missing_capacitor",
                        "bbox_nm": [int(cx) - cap_half, int(cy) - cap_half,
                                    int(cx) + cap_half, int(cy) + cap_half],
                    })
                    continue

                box = db.Box(int(cx) - cap_half, int(cy) - cap_half,
                             int(cx) + cap_half, int(cy) + cap_half)
                box = self.pv.overlay_shift(box)
                self._insert(self.L_CAP, box)

        self._cell_centers = centers
        return centers

    def _build_wordlines(self) -> List[dict]:
        """
        Place horizontal word-line stripes (Layer 2) for each active row.
        WLs extend into the dummy region to maintain array periodicity.
        Applies linewidth variation, LER, overlay shift, and broken-WL defects.
        Returns list of nominal WL coordinate dicts for metadata.
        """
        p = self.p
        half_wl = int(p.wl_width_nm / 2)
        x_start = int(-p.dummy_cols * p.cell_pitch_bl_nm)
        x_end   = int((p.cols + p.dummy_cols) * p.cell_pitch_bl_nm)
        coords  = []

        for row in range(p.rows):
            y_nom = int(row * p.cell_pitch_nm + p.cell_pitch_nm / 2)
            coords.append({
                "row": row,
                "y_nm": y_nom,
                "x_start_nm": x_start,
                "x_end_nm": x_end,
            })

            box = db.Box(x_start, y_nom - half_wl, x_end, y_nom + half_wl)
            box = self.pv.vary_linewidth(box, 'x')
            box = self.pv.overlay_shift(box)

            if not p.single_training_defect and self.pv.is_missing(p.p_broken_wl):
                pieces = self.pv.break_line(box, 'x')
                self._defect_locs.append({
                    "type": "broken_wl",
                    "bbox_nm": [box.left, box.bottom, box.right, box.top],
                })
                valid_pieces = [b for b in pieces if b.width() > 0 and b.height() > 0]
                self._wl_boxes.extend(valid_pieces)
                self._insert(self.L_WL, pieces)
            else:
                # Apply LER to the WL edges
                ler_boxes = self.pv.add_ler(box, 'x')
                valid_ler = [b for b in ler_boxes if b.width() > 0 and b.height() > 0]
                self._wl_boxes.extend(valid_ler)
                self._insert(self.L_WL, ler_boxes)

        self._wl_coords = coords
        return coords

    def _build_bitlines(self) -> List[dict]:
        """
        Place vertical bit-line stripes (Layer 3) for each active column.
        BLs extend into the dummy region. Applies the same variation suite as WLs.
        Returns list of nominal BL coordinate dicts for metadata.
        """
        p = self.p
        half_bl = int(p.bl_width_nm / 2)
        y_start = int(-p.dummy_rows * p.cell_pitch_nm)
        y_end   = int((p.rows + p.dummy_rows) * p.cell_pitch_nm)
        coords  = []

        for col in range(p.cols):
            x_nom = int(col * p.cell_pitch_bl_nm + p.cell_pitch_bl_nm / 2)
            coords.append({
                "col": col,
                "x_nm": x_nom,
                "y_start_nm": y_start,
                "y_end_nm": y_end,
            })

            box = db.Box(x_nom - half_bl, y_start, x_nom + half_bl, y_end)
            box = self.pv.vary_linewidth(box, 'y')
            box = self.pv.overlay_shift(box)

            if not p.single_training_defect and self.pv.is_missing(p.p_broken_bl):
                pieces = self.pv.break_line(box, 'y')
                self._defect_locs.append({
                    "type": "broken_bl",
                    "bbox_nm": [box.left, box.bottom, box.right, box.top],
                })
                valid_pieces = [b for b in pieces if b.width() > 0 and b.height() > 0]
                self._bl_boxes.extend(valid_pieces)
                self._insert(self.L_BL, pieces)
            else:
                ler_boxes = self.pv.add_ler(box, 'y')
                valid_ler = [b for b in ler_boxes if b.width() > 0 and b.height() > 0]
                self._bl_boxes.extend(valid_ler)
                self._insert(self.L_BL, ler_boxes)

        self._bl_coords = coords
        return coords

    def _build_contacts(self) -> List[List[float]]:
        """
        Place transistor-drain contacts (Layer 4) for each active cell.
        The contact is offset from the capacitor node toward the bit line
        (drain side of the access transistor). This offset represents the
        physical gap between the storage node and the bit-line contact.
        Applies contact CD variation, overlay shift, and missing-contact defects.
        Returns list of [x_nm, y_nm] contact centre coordinates.
        """
        p = self.p
        con_half  = int(p.contact_size_nm / 2)
        x_offset  = int(p.cell_pitch_bl_nm / 4)   # drain-side offset in col direction
        contact_centers: List[List[float]] = []

        for row in range(p.rows):
            for col in range(p.cols):
                nom_cx, nom_cy = self._cell_center(row, col)
                cx = nom_cx + x_offset
                cy = nom_cy
                contact_centers.append([cx, cy])

                box = db.Box(int(cx) - con_half, int(cy) - con_half,
                             int(cx) + con_half, int(cy) + con_half)
                box = self.pv.vary_contact(box)
                box = self.pv.overlay_shift(box)

                if not p.single_training_defect and self.pv.is_missing(p.p_missing_contact):
                    self._defect_locs.append({
                        "type": "missing_contact",
                        "bbox_nm": [box.left, box.bottom, box.right, box.top],
                    })
                    continue

                self._insert(self.L_CON, box)

        return contact_centers

    def _add_defects(self) -> List[dict]:
        """
        Place global defects on Layer 5:
          - Particle contamination: random rectangles across the array.
          - Scratches: thin elongated rectangles.
          - CMP dishing: shrunken line widths in a circular region.
            (Dishing affects WL/BL shape records for metadata only --
             geometry already placed on Layers 2/3. We record the region.)
          - Training defect: when single_training_defect is set, particles
            and scratches above are skipped and exactly one small diagonal
            training_scratch defect is placed instead.
        Returns list of new defect dicts.
        """
        p = self.p
        full = self._full_bbox()
        new_defects: List[dict] = []

        if not p.single_training_defect:
            # Particles
            for box in self.pv.add_particles(full, p.n_particles):
                self._insert(self.L_DEF, box)
                new_defects.append({"type": "particle",
                                     "bbox_nm": [box.left, box.bottom,
                                                 box.right, box.top]})

            # Scratches
            for box in self.pv.add_scratches(full, p.n_scratches):
                self._insert(self.L_DEF, box)
                new_defects.append({"type": "scratch",
                                     "bbox_nm": [box.left, box.bottom,
                                                 box.right, box.top]})

        # CMP dishing -- shrink WL/BL geometry in a circular region, then mark it
        if p.p_cmp_dishing > 0 and not p.single_training_defect:
            arr = self._array_bbox()
            dish_cx = float((arr.left + arr.right) / 2
                            + self.rng.uniform(-arr.width() / 4, arr.width() / 4))
            dish_cy = float((arr.bottom + arr.top) / 2
                            + self.rng.uniform(-arr.height() / 4, arr.height() / 4))
            dish_r = float(min(arr.width(), arr.height()) * p.p_cmp_dishing)
            dr = int(dish_r)

            # Apply dishing: clear existing WL/BL shapes and re-insert thinned versions
            dished_wl = self.pv.cmp_dishing(self._wl_boxes, dish_cx, dish_cy, dish_r)
            self.cell.shapes(self.L_WL).clear()
            self._insert(self.L_WL, dished_wl)

            dished_bl = self.pv.cmp_dishing(self._bl_boxes, dish_cx, dish_cy, dish_r)
            self.cell.shapes(self.L_BL).clear()
            self._insert(self.L_BL, dished_bl)

            # Place a circular-region marker on the defect layer
            marker = db.Box(int(dish_cx) - dr, int(dish_cy) - dr,
                            int(dish_cx) + dr, int(dish_cy) + dr)
            self._insert(self.L_DEF, marker)
            new_defects.append({"type": "cmp_dishing",
                                 "bbox_nm": [marker.left, marker.bottom,
                                             marker.right, marker.top]})

        if p.single_training_defect:
            arr = self._array_bbox()
            canvas = _canvas_bbox_nm(self._pixels_per_nm, p.output_width_px, p.output_height_px,
                                      float(arr.top - arr.bottom))
            new_defects.append(_place_training_defect(
                self.cell, self.L_DEF, self._pixels_per_nm, self.rng,
                canvas, p.training_defect_size_px))

        self._defect_locs.extend(new_defects)
        return new_defects

    def generate(self) -> dict:
        """
        Build the full DRAM array layout, write GDS + JSON, return metadata dict.
        Build order: dummy cells -> capacitors -> word lines -> bit lines ->
                     contacts -> defects -> write files -> assemble metadata.
        """
        self._build_dummy_cells()
        self._build_capacitors()
        self._build_wordlines()
        self._build_bitlines()
        self._build_contacts()
        self._add_defects()

        self.layout.write(self.p.output_gds)

        p = self.p
        arr = self._array_bbox()
        array_w_nm = float(arr.right - arr.left)
        array_h_nm = float(arr.top   - arr.bottom)

        # Reference patch selection -- random window fully inside the active array
        ref_w_nm = p.reference_width_px  / self._pixels_per_nm
        ref_h_nm = p.reference_height_px / self._pixels_per_nm
        rx0 = float(self.rng.uniform(0, array_w_nm - ref_w_nm))
        ry0 = float(self.rng.uniform(0, array_h_nm - ref_h_nm))
        rx1 = rx0 + ref_w_nm
        ry1 = ry0 + ref_h_nm
        ref_bbox_nm   = [rx0, ry0, rx1, ry1]
        ref_center_nm = [rx0 + ref_w_nm / 2, ry0 + ref_h_nm / 2]
        ref_bbox_px   = bbox_nm_to_pixel(ref_bbox_nm, self._pixels_per_nm)
        ref_center_px = [ref_center_nm[0] * self._pixels_per_nm,
                         ref_center_nm[1] * self._pixels_per_nm]

        meta = {
            "cell_centers":    self._cell_centers,
            "wordline_coords": self._wl_coords,
            "bitline_coords":  self._bl_coords,
            "bounding_box": {
                "x_min_nm": arr.left,
                "y_min_nm": arr.bottom,
                "x_max_nm": arr.right,
                "y_max_nm": arr.top,
            },
            "defect_locations": self._defect_locs,
            "params":    dataclasses.asdict(p),
            "gds_file":  p.output_gds,
            "metadata_file": p.output_json,
            # ── Rasterization / localization ─────────────────────────────────
            "search_image_size_px":    [p.output_width_px, p.output_height_px],
            "reference_image_size_px": [p.reference_width_px, p.reference_height_px],
            "zoom_ratio":              p.zoom_ratio,
            "pixels_per_nm":           self._pixels_per_nm,
            "search_bbox_nm":          [0.0, 0.0, array_w_nm, array_h_nm],
            "search_bbox_px":          [0.0, 0.0,
                                        float(p.output_width_px),
                                        float(p.output_height_px)],
            "reference_bbox_nm":       ref_bbox_nm,
            "reference_center_nm":     ref_center_nm,
            "reference_bbox_px":       ref_bbox_px,
            "reference_center_px":     ref_center_px,
        }

        Path(p.output_json).write_text(json.dumps(meta, indent=2))
        self._last_meta = meta
        return meta

    def get_rasterization_config(self) -> RasterizationConfig:
        """
        Return a RasterizationConfig from the last generate() call.
        Must be called after generate().
        """
        meta = self._last_meta
        layers = [LAYER_CAPACITOR, LAYER_WORDLINE, LAYER_BITLINE,
                  LAYER_CONTACT, LAYER_DEFECT, LAYER_DUMMY]
        ref_bbox_px = [int(round(v)) for v in meta["reference_bbox_px"]]
        return RasterizationConfig(
            gds_file=self.p.output_gds,
            pixels_per_nm=self._pixels_per_nm,
            origin_nm=(0.0, 0.0),
            width_px=self.p.output_width_px,
            height_px=self.p.output_height_px,
            layers=layers,
            reference_bbox_px=ref_bbox_px,
            reference_center_px=meta["reference_center_px"],
        )


def generate_dram_layout(params: DRAMParams = DRAMParams()) -> dict:
    """Generate synthetic DRAM GDS layout. Returns metadata dict."""
    return DRAMGenerator(params).generate()


class DieGenerator:
    """
    Composites a die_block_rows x die_block_cols grid of independently
    randomized DRAM array blocks into one shared GDS layout, with a street
    layer filling the gaps between blocks. die_block_rows=die_block_cols=1
    (the DRAMParams default) is an exact passthrough to DRAMGenerator.
    """

    def __init__(self, params: DRAMParams) -> None:
        if params.die_block_rows < 1 or params.die_block_cols < 1:
            raise ValueError("die_block_rows and die_block_cols must be >= 1")
        self.p = params
        self.rng = np.random.default_rng(params.seed)

        self.layout = db.Layout()
        self.layout.dbu = params.dbu_nm * 1e-3
        self.cell = self.layout.create_cell("DRAM_DIE")
        self.L_CAP = self.layout.layer(*LAYER_CAPACITOR)
        self.L_WL = self.layout.layer(*LAYER_WORDLINE)
        self.L_BL = self.layout.layer(*LAYER_BITLINE)
        self.L_CON = self.layout.layer(*LAYER_CONTACT)
        self.L_DEF = self.layout.layer(*LAYER_DEFECT)
        self.L_DUM = self.layout.layer(*LAYER_DUMMY)
        self.L_STREET = self.layout.layer(*LAYER_STREET)

        array_w_nm = params.die_block_cols * params.die_block_width_nm
        array_h_nm = params.die_block_rows * params.die_block_height_nm
        if params.pixels_per_nm is None:
            self._pixels_per_nm: float = min(
                params.output_width_px / array_w_nm,
                params.output_height_px / array_h_nm,
            )
        else:
            self._pixels_per_nm = params.pixels_per_nm

        self._blocks_meta: List[dict] = []
        self._cell_centers: List[List[float]] = []
        self._wl_coords: List[dict] = []
        self._bl_coords: List[dict] = []
        self._defect_locs: List[dict] = []
        self._last_meta: dict = {}

    def _total_die_size_nm(self) -> Tuple[float, float]:
        p = self.p
        w = (p.die_block_cols * p.die_block_width_nm
             + (p.die_block_cols - 1) * p.die_street_width_nm)
        h = (p.die_block_rows * p.die_block_height_nm
             + (p.die_block_rows - 1) * p.die_street_width_nm)
        return w, h

    def _block_origin_nm(self, br: int, bc: int) -> Tuple[float, float]:
        p = self.p
        x0 = bc * (p.die_block_width_nm + p.die_street_width_nm)
        y0 = br * (p.die_block_height_nm + p.die_street_width_nm)
        return x0, y0

    def _build_block(self, br: int, bc: int, tmp_dir: str) -> None:
        p = self.p
        block_seed = (p.seed * 100_003 + br * p.die_block_cols + bc) % (2 ** 32 - 1)
        block_rng = np.random.default_rng(block_seed)
        block_params = _randomize_block_params(p, block_rng, p.die_block_variation_frac)

        rows = max(1, int(p.die_block_height_nm / block_params.cell_pitch_nm))
        cols = max(1, int(p.die_block_width_nm / block_params.cell_pitch_bl_nm))

        stem = f"block_{br:02d}_{bc:02d}"
        # When the die places one guaranteed training defect at the die
        # level (in generate(), after all blocks + streets are assembled),
        # blocks must not also place their own random defects of any kind
        # (particles/scratches, CMP dishing, missing capacitors/contacts,
        # broken word/bit lines) or their own training defect -- mirrors
        # Task 1's DRAMGenerator._add_defects gating, which suppresses this
        # same set of probabilities when single_training_defect is True.
        no_defects_kw = ({"n_particles": 0, "n_scratches": 0,
                           "p_cmp_dishing": 0.0, "p_missing_capacitor": 0.0,
                           "p_broken_wl": 0.0, "p_broken_bl": 0.0,
                           "p_missing_contact": 0.0,
                           "single_training_defect": False}
                          if p.single_training_defect else {})
        block_params = dataclasses.replace(
            block_params,
            rows=rows, cols=cols,
            seed=block_seed,
            pixels_per_nm=self._pixels_per_nm,
            output_gds=str(Path(tmp_dir) / f"{stem}.gds"),
            output_json=str(Path(tmp_dir) / f"{stem}.json"),
            **no_defects_kw,
        )

        block_meta = DRAMGenerator(block_params).generate()

        x0_nm, y0_nm = self._block_origin_nm(br, bc)
        dx, dy = int(round(x0_nm)), int(round(y0_nm))

        block_layout = db.Layout()
        block_layout.read(block_params.output_gds)
        block_top = block_layout.top_cell()
        layer_map = [
            (LAYER_CAPACITOR, self.L_CAP), (LAYER_WORDLINE, self.L_WL),
            (LAYER_BITLINE, self.L_BL), (LAYER_CONTACT, self.L_CON),
            (LAYER_DEFECT, self.L_DEF), (LAYER_DUMMY, self.L_DUM),
        ]
        for (ln, dt), dest_layer in layer_map:
            src_li = block_layout.find_layer(ln, dt)
            if src_li is None:
                continue
            for shape in block_top.shapes(src_li).each():
                self.cell.shapes(dest_layer).insert(shape.bbox().moved(dx, dy))

        for f in (block_params.output_gds, block_params.output_json):
            Path(f).unlink(missing_ok=True)

        self._cell_centers.extend(
            [c[0] + x0_nm, c[1] + y0_nm] for c in block_meta["cell_centers"])
        for wl in block_meta["wordline_coords"]:
            self._wl_coords.append({
                **wl,
                "y_nm": wl["y_nm"] + y0_nm,
                "x_start_nm": wl["x_start_nm"] + x0_nm,
                "x_end_nm": wl["x_end_nm"] + x0_nm,
            })
        for bl in block_meta["bitline_coords"]:
            self._bl_coords.append({
                **bl,
                "x_nm": bl["x_nm"] + x0_nm,
                "y_start_nm": bl["y_start_nm"] + y0_nm,
                "y_end_nm": bl["y_end_nm"] + y0_nm,
            })
        for d in block_meta["defect_locations"]:
            bx0, by0, bx1, by1 = d["bbox_nm"]
            self._defect_locs.append({
                **d,
                "bbox_nm": [bx0 + x0_nm, by0 + y0_nm, bx1 + x0_nm, by1 + y0_nm],
            })

        block_bbox_nm = [x0_nm, y0_nm,
                          x0_nm + p.die_block_width_nm, y0_nm + p.die_block_height_nm]
        self._blocks_meta.append({
            "block_row": br, "block_col": bc,
            "bbox_nm": block_bbox_nm,
            "params": dataclasses.asdict(block_params),
        })

    def _build_streets(self) -> None:
        p = self.p
        if p.die_street_width_nm <= 0:
            return
        total_w_nm, total_h_nm = self._total_die_size_nm()
        sw = p.die_street_width_nm
        for bc in range(1, p.die_block_cols):
            x0 = bc * (p.die_block_width_nm + sw) - sw
            box = db.Box(int(round(x0)), 0, int(round(x0 + sw)), int(round(total_h_nm)))
            self.cell.shapes(self.L_STREET).insert(box)
        for br in range(1, p.die_block_rows):
            y0 = br * (p.die_block_height_nm + sw) - sw
            box = db.Box(0, int(round(y0)), int(round(total_w_nm)), int(round(y0 + sw)))
            self.cell.shapes(self.L_STREET).insert(box)

    def generate(self) -> dict:
        """
        Build the full die (all blocks + streets), write GDS + JSON, return
        metadata. die_block_rows=die_block_cols=1 is an exact passthrough to
        DRAMGenerator(self.p).generate() (same GDS bytes, same metadata plus
        a single-entry "blocks" list) -- no block randomization or footprint
        rescaling is applied in that case.
        """
        p = self.p
        if p.die_block_rows == 1 and p.die_block_cols == 1:
            meta = DRAMGenerator(p).generate()
            bb = meta["bounding_box"]
            meta["blocks"] = [{
                "block_row": 0, "block_col": 0,
                "bbox_nm": [bb["x_min_nm"], bb["y_min_nm"], bb["x_max_nm"], bb["y_max_nm"]],
                "params": dataclasses.asdict(p),
            }]
            Path(p.output_json).write_text(json.dumps(meta, indent=2))
            self._last_meta = meta
            return meta

        with tempfile.TemporaryDirectory() as tmp_dir:
            for br in range(p.die_block_rows):
                for bc in range(p.die_block_cols):
                    self._build_block(br, bc, tmp_dir)
            self._build_streets()

        total_w_nm, total_h_nm = self._total_die_size_nm()

        if p.single_training_defect:
            canvas = _canvas_bbox_nm(self._pixels_per_nm, p.output_width_px, p.output_height_px,
                                      total_h_nm)
            self._defect_locs.append(_place_training_defect(
                self.cell, self.L_DEF, self._pixels_per_nm, self.rng,
                canvas, p.training_defect_size_px))

        self.layout.write(p.output_gds)

        ref_br = int(self.rng.integers(0, p.die_block_rows))
        ref_bc = int(self.rng.integers(0, p.die_block_cols))
        block_x0, block_y0 = self._block_origin_nm(ref_br, ref_bc)

        ref_w_nm = p.reference_width_px / self._pixels_per_nm
        ref_h_nm = p.reference_height_px / self._pixels_per_nm
        rx0 = block_x0 + float(self.rng.uniform(0, max(0.0, p.die_block_width_nm - ref_w_nm)))
        ry0 = block_y0 + float(self.rng.uniform(0, max(0.0, p.die_block_height_nm - ref_h_nm)))
        rx1, ry1 = rx0 + ref_w_nm, ry0 + ref_h_nm
        ref_bbox_nm = [rx0, ry0, rx1, ry1]
        ref_center_nm = [rx0 + ref_w_nm / 2, ry0 + ref_h_nm / 2]
        ref_bbox_px = bbox_nm_to_pixel(ref_bbox_nm, self._pixels_per_nm)
        ref_center_px = [ref_center_nm[0] * self._pixels_per_nm,
                          ref_center_nm[1] * self._pixels_per_nm]

        meta = {
            "cell_centers": self._cell_centers,
            "wordline_coords": self._wl_coords,
            "bitline_coords": self._bl_coords,
            "bounding_box": {
                "x_min_nm": 0, "y_min_nm": 0,
                "x_max_nm": total_w_nm, "y_max_nm": total_h_nm,
            },
            "defect_locations": self._defect_locs,
            "blocks": self._blocks_meta,
            "params": dataclasses.asdict(p),
            "gds_file": p.output_gds,
            "metadata_file": p.output_json,
            "search_image_size_px": [p.output_width_px, p.output_height_px],
            "reference_image_size_px": [p.reference_width_px, p.reference_height_px],
            "zoom_ratio": p.zoom_ratio,
            "pixels_per_nm": self._pixels_per_nm,
            "search_bbox_nm": [0.0, 0.0, total_w_nm, total_h_nm],
            "search_bbox_px": [0.0, 0.0, float(p.output_width_px), float(p.output_height_px)],
            "reference_bbox_nm": ref_bbox_nm,
            "reference_center_nm": ref_center_nm,
            "reference_bbox_px": ref_bbox_px,
            "reference_center_px": ref_center_px,
        }
        Path(p.output_json).write_text(json.dumps(meta, indent=2))
        self._last_meta = meta
        return meta


# ═════════════════════════════════════════════════════════════════════════
# 2. SEM-style rasterization (sem_render.py)
# ═════════════════════════════════════════════════════════════════════════

IMG_SIZE  = 1000
SS_FACTOR = 4
SS_SIZE   = IMG_SIZE * SS_FACTOR

SEM_LAYER_ORDER = [(6, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (7, 0)]
SEM_INTENSITY = {
    (6, 0):  40,   # Dummy      - background fill, dim
    (1, 0): 150,   # Capacitor  - dielectric/poly, mid
    (2, 0): 170,   # WordLine   - poly gate, mid-bright
    (3, 0): 190,   # BitLine    - metal-ish, bright
    (4, 0): 255,   # Contact    - metal plug, brightest (highest SE yield)
    (5, 0): 100,   # Defect     - charging artifact, darker
    (7, 0):  70,   # Street     - inter-block scribe line, visible but non-dominant
}
SEM_BG = 25        # bare substrate

DEBUG_LAYER_STYLE = [
    ((6, 0), ( 45,  50,  60)),
    ((1, 0), (220,  80,  80)),
    ((2, 0), ( 60, 150, 255)),
    ((3, 0), ( 60, 220, 130)),
    ((4, 0), (255, 200,  80)),
    ((5, 0), (190, 100, 255)),
    ((7, 0), (140, 140, 150)),
]
DEBUG_BG  = (10, 12, 18)
DEBUG_REF = (255, 255, 0)

CIRCLE_LAYERS = {(4, 0)}  # contacts render as inscribed circles, not rectangles

DEFECT_HALO_MARGIN_NM = 15.0
DEFECT_HALO_INTENSITY = 200
DEFECT_MARK_INTENSITY = 30
DISHING_INTENSITY     = 140

POINT_DEFECT_TYPES = {"particle", "scratch", "training_scratch"}


def _supersampled_masks_for_bbox(gds_path: str, bbox_nm: list, ppn_ss: float,
                                  layer_list: list, ss_w: int, ss_h: int) -> dict:
    """
    Paint each layer as an opaque binary mask for an arbitrary nm bounding
    box, at ss_w x ss_h supersampled resolution. Coordinates are kept in
    float space and rounded once at fill time so downsampling later
    produces true edge antialiasing.
    """
    x0_nm, y0_nm, x1_nm, y1_nm = bbox_nm

    layout = db.Layout()
    layout.read(gds_path)
    top = layout.top_cell()

    masks = {}
    for ln, dt in layer_list:
        li = layout.find_layer(ln, dt)
        mask = np.zeros((ss_h, ss_w), dtype=np.uint8)
        if li is not None:
            for shape in top.shapes(li).each():
                b = shape.bbox()
                px0 = (b.left  - x0_nm) * ppn_ss
                px1 = (b.right - x0_nm) * ppn_ss
                py0 = (y1_nm - b.top)    * ppn_ss
                py1 = (y1_nm - b.bottom) * ppn_ss
                ix0, iy0 = max(0, int(round(px0))), max(0, int(round(py0)))
                ix1, iy1 = min(ss_w, int(round(px1))), min(ss_h, int(round(py1)))
                if ix1 > ix0 and iy1 > iy0:
                    if (ln, dt) in CIRCLE_LAYERS:
                        cx = (px0 + px1) / 2.0
                        cy = (py0 + py1) / 2.0
                        radius = min(px1 - px0, py1 - py0) / 2.0
                        cv2.circle(mask, (int(round(cx)), int(round(cy))),
                                   max(1, int(round(radius))), 255, -1)
                    else:
                        mask[iy0:iy1, ix0:ix1] = 255
        masks[(ln, dt)] = mask.astype(np.float32) / 255.0
    return masks


def _supersampled_layer_masks(gds_path: str, meta: dict, layer_list: list) -> dict:
    """Whole-array masks at the layout's native pixels_per_nm, SS_SIZE x SS_SIZE."""
    ppn_ss = meta["pixels_per_nm"] * SS_FACTOR
    array_w_nm = meta["search_bbox_nm"][2]
    array_h_nm = meta["search_bbox_nm"][3]
    bbox_nm = [0.0, 0.0, array_w_nm, array_h_nm]
    return _supersampled_masks_for_bbox(gds_path, bbox_nm, ppn_ss, layer_list, SS_SIZE, SS_SIZE)


def _nm_bbox_to_ss_px(bbox_nm: list, origin_bbox_nm: list, ppn_ss: float) -> tuple:
    """Convert an nm bbox into supersampled pixel coords relative to the
    current render's own origin bbox -- the same transform as
    _supersampled_masks_for_bbox, so this is correct for both the
    whole-array search render and an arbitrary reference-patch crop."""
    x0_nm, y0_nm, x1_nm, y1_nm = origin_bbox_nm
    bx0, by0, bx1, by1 = bbox_nm
    px0 = (bx0 - x0_nm) * ppn_ss
    px1 = (bx1 - x0_nm) * ppn_ss
    py0 = (y1_nm - by1) * ppn_ss
    py1 = (y1_nm - by0) * ppn_ss
    return px0, py0, px1, py1


def _paint_defects(intensity: np.ndarray, defect_locations: list,
                    origin_bbox_nm: list, ppn_ss: float) -> None:
    """
    Defect rendering, in place on `intensity` (same 0-255 scale as
    SEM_INTENSITY). Call after the generic per-layer fill so this overrides
    the flat defect-layer intensity within each defect's footprint:
      - particle: a lighter "disturbed-zone" halo (inflated bbox) with the
        actual defect mark (original bbox) drawn on top, darker.
      - scratch: a single thin anti-aliased dark line along the defect's
        long axis (no halo) -- a small, subtle mark rather than a blocky
        two-tone region.
      - training_scratch: a single thin anti-aliased diagonal line from the
        defect bbox's top-left to bottom-right corner (no halo) -- visually
        distinct from the axis-aligned scratch above.
      - cmp_dishing: a softer translucent tint over the (already large,
        diffuse) affected region -- not the halo+mark treatment.
    A defect only partially inside origin_bbox_nm is naturally clipped by
    the same bounds-checking used elsewhere in this module (cv2.line clips
    out-of-bounds coordinates on its own).
    """
    ss_h, ss_w = intensity.shape[:2]
    for defect in defect_locations:
        if defect["type"] not in POINT_DEFECT_TYPES and defect["type"] != "cmp_dishing":
            continue

        bbox_nm = defect["bbox_nm"]

        if defect["type"] == "cmp_dishing":
            px0, py0, px1, py1 = _nm_bbox_to_ss_px(bbox_nm, origin_bbox_nm, ppn_ss)
            ix0, iy0 = max(0, int(round(px0))), max(0, int(round(py0)))
            ix1, iy1 = min(ss_w, int(round(px1))), min(ss_h, int(round(py1)))
            if ix1 > ix0 and iy1 > iy0:
                region = intensity[iy0:iy1, ix0:ix1]
                intensity[iy0:iy1, ix0:ix1] = 0.5 * region + 0.5 * DISHING_INTENSITY
            continue

        if defect["type"] == "scratch":
            px0, py0, px1, py1 = _nm_bbox_to_ss_px(bbox_nm, origin_bbox_nm, ppn_ss)
            w, h = px1 - px0, py1 - py0
            thickness = max(1, int(round(min(w, h))))
            if w >= h:
                pt1 = (px0, (py0 + py1) / 2.0)
                pt2 = (px1, (py0 + py1) / 2.0)
            else:
                pt1 = ((px0 + px1) / 2.0, py0)
                pt2 = ((px0 + px1) / 2.0, py1)
            cv2.line(intensity,
                      (int(round(pt1[0])), int(round(pt1[1]))),
                      (int(round(pt2[0])), int(round(pt2[1]))),
                      DEFECT_MARK_INTENSITY, thickness, cv2.LINE_AA)
            continue

        if defect["type"] == "training_scratch":
            px0, py0, px1, py1 = _nm_bbox_to_ss_px(bbox_nm, origin_bbox_nm, ppn_ss)
            thickness = max(1, int(round(min(px1 - px0, py1 - py0) * 0.15)))
            cv2.line(intensity,
                      (int(round(px0)), int(round(py0))),
                      (int(round(px1)), int(round(py1))),
                      DEFECT_MARK_INTENSITY, thickness, cv2.LINE_AA)
            continue

        bx0, by0, bx1, by1 = bbox_nm
        halo_nm = [bx0 - DEFECT_HALO_MARGIN_NM, by0 - DEFECT_HALO_MARGIN_NM,
                   bx1 + DEFECT_HALO_MARGIN_NM, by1 + DEFECT_HALO_MARGIN_NM]

        hpx0, hpy0, hpx1, hpy1 = _nm_bbox_to_ss_px(halo_nm, origin_bbox_nm, ppn_ss)
        ix0, iy0 = max(0, int(round(hpx0))), max(0, int(round(hpy0)))
        ix1, iy1 = min(ss_w, int(round(hpx1))), min(ss_h, int(round(hpy1)))
        if ix1 > ix0 and iy1 > iy0:
            intensity[iy0:iy1, ix0:ix1] = DEFECT_HALO_INTENSITY

        mpx0, mpy0, mpx1, mpy1 = _nm_bbox_to_ss_px(bbox_nm, origin_bbox_nm, ppn_ss)
        ix0, iy0 = max(0, int(round(mpx0))), max(0, int(round(mpy0)))
        ix1, iy1 = min(ss_w, int(round(mpx1))), min(ss_h, int(round(mpy1)))
        if ix1 > ix0 and iy1 > iy0:
            intensity[iy0:iy1, ix0:ix1] = DEFECT_MARK_INTENSITY


def _downsample(img: np.ndarray) -> np.ndarray:
    """Area-average downsample SS_SIZE -> IMG_SIZE."""
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def _content_and_period_px(meta: dict) -> tuple:
    """
    Content extent and cell periodicity in supersampled pixels.

    pixels_per_nm is a single uniform scale (chosen as the tighter of the
    two axis fits, so pixels stay square and features aren't distorted) --
    when the array isn't square, one axis fills the canvas exactly and the
    other leaves a margin. Returns (content_w_px, content_h_px, period_x_px,
    period_y_px) so that margin can be filled by continuing the periodic
    lattice rather than left blank.
    """
    ppn = meta["pixels_per_nm"] * SS_FACTOR
    x0, y0, x1, y1 = meta["search_bbox_nm"]
    content_w = int(round((x1 - x0) * ppn))
    content_h = int(round((y1 - y0) * ppn))
    period_x  = int(round(meta["params"]["cell_pitch_bl_nm"] * ppn))
    period_y  = int(round(meta["params"]["cell_pitch_nm"] * ppn))
    return content_w, content_h, period_x, period_y


def _tile_margin(img: np.ndarray, content_w: int, content_h: int,
                  period_x: int, period_y: int) -> np.ndarray:
    """
    Extend rendered content into unused canvas margin by repeating the
    lattice at its actual pitch, instead of leaving flat dead space. The
    array is genuinely periodic, so this continues real structure rather
    than fabricating new geometry.
    """
    H, W = img.shape[:2]
    if 0 < content_w < W and period_x > 0:
        n_cols = W - content_w
        src = content_w - period_x + (np.arange(n_cols) % period_x)
        src = np.clip(src, 0, content_w - 1)
        img[:, content_w:] = img[:, src]
    if 0 < content_h < H and period_y > 0:
        n_rows = H - content_h
        src = content_h - period_y + (np.arange(n_rows) % period_y)
        src = np.clip(src, 0, content_h - 1)
        img[content_h:, :] = img[src, :]
    return img


def render_sem_image(
    gds_path: str,
    meta: dict,
    rng: np.random.Generator,
    beam_sigma_px: float = 0.7,
    photon_scale: float = 4000.0,
    read_noise_sigma: float = 0.012,
    edge_gain: float = 0.35,
) -> np.ndarray:
    """GDS -> grayscale synthetic SEM image of the whole array. Model-facing output."""
    masks = _supersampled_layer_masks(gds_path, meta, SEM_LAYER_ORDER)
    content_w, content_h, period_x, period_y = _content_and_period_px(meta)

    intensity = np.full((SS_SIZE, SS_SIZE), SEM_BG, dtype=np.float32)
    for layer in SEM_LAYER_ORDER:
        m = masks[layer] > 0.5
        intensity[m] = SEM_INTENSITY[layer]
    intensity = _tile_margin(intensity, content_w, content_h, period_x, period_y)

    array_w_nm = meta["search_bbox_nm"][2]
    array_h_nm = meta["search_bbox_nm"][3]
    ppn_ss = meta["pixels_per_nm"] * SS_FACTOR
    _paint_defects(intensity, meta.get("defect_locations", []),
                   [0.0, 0.0, array_w_nm, array_h_nm], ppn_ss)

    intensity /= 255.0

    gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-6)
    intensity = np.clip(intensity + edge_gain * edge_mag, 0, 1)

    img = _downsample(intensity)
    img = cv2.GaussianBlur(img, (0, 0), beam_sigma_px)
    img = rng.poisson(img * photon_scale).astype(np.float32) / photon_scale
    img += rng.normal(0, read_noise_sigma, img.shape).astype(np.float32)

    gain   = float(rng.uniform(0.85, 1.15))
    offset = float(rng.uniform(-0.05, 0.05))
    img = img * gain + offset

    return (img.clip(0, 1) * 255).astype(np.uint8)


def render_sem_patch(
    gds_path: str,
    bbox_nm: list,
    ppn: float,
    width_px: int,
    height_px: int,
    rng: np.random.Generator,
    defect_locations: list = None,
    beam_sigma_px: float = 0.7,
    photon_scale: float = 4000.0,
    read_noise_sigma: float = 0.012,
    edge_gain: float = 0.35,
) -> np.ndarray:
    """
    GDS -> grayscale synthetic SEM image of an arbitrary nm bbox at an
    arbitrary pixel scale. Same noise pipeline as render_sem_image, but for
    a caller-specified crop (used for the high-resolution reference patch)
    instead of the whole array. No margin tiling: the caller picks
    width_px/height_px to exactly match bbox_nm at ppn, so content always
    fills the canvas exactly.
    """
    ss_w = width_px * SS_FACTOR
    ss_h = height_px * SS_FACTOR
    ppn_ss = ppn * SS_FACTOR

    masks = _supersampled_masks_for_bbox(gds_path, bbox_nm, ppn_ss, SEM_LAYER_ORDER, ss_w, ss_h)

    intensity = np.full((ss_h, ss_w), SEM_BG, dtype=np.float32)
    for layer in SEM_LAYER_ORDER:
        m = masks[layer] > 0.5
        intensity[m] = SEM_INTENSITY[layer]

    _paint_defects(intensity, defect_locations or [], bbox_nm, ppn_ss)

    intensity /= 255.0

    gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-6)
    intensity = np.clip(intensity + edge_gain * edge_mag, 0, 1)

    img = cv2.resize(intensity, (width_px, height_px), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), beam_sigma_px)
    img = rng.poisson(img * photon_scale).astype(np.float32) / photon_scale
    img += rng.normal(0, read_noise_sigma, img.shape).astype(np.float32)

    gain   = float(rng.uniform(0.85, 1.15))
    offset = float(rng.uniform(-0.05, 0.05))
    img = img * gain + offset

    return (img.clip(0, 1) * 255).astype(np.uint8)


def render_debug_image(gds_path: str, meta: dict) -> np.ndarray:
    """
    Layer-colored RGB image with the ground-truth reference box/center burned
    in. FOR HUMAN QA ONLY -- never feed this file to a model.
    """
    layer_list = [ln for ln, _ in DEBUG_LAYER_STYLE]
    masks = _supersampled_layer_masks(gds_path, meta, layer_list)
    content_w, content_h, period_x, period_y = _content_and_period_px(meta)

    canvas_ss = np.empty((SS_SIZE, SS_SIZE, 3), dtype=np.float32)
    canvas_ss[:] = DEBUG_BG
    for layer, color in DEBUG_LAYER_STYLE:
        m = masks[layer] > 0.5
        canvas_ss[m] = color
    canvas_ss = _tile_margin(canvas_ss, content_w, content_h, period_x, period_y)

    canvas = _downsample(canvas_ss).astype(np.uint8)

    # meta["reference_bbox_px"]/["reference_center_px"] are a plain nm->px
    # scale (no y-flip), but this canvas -- like every rasterized image in
    # this module -- has row 0 at *maximum* nm-y. Recompute pixel coords
    # from the nm values with the same flip used everywhere else, so this
    # QA overlay actually shows where the reference patch was rendered.
    x0_nm, y0_nm, x1_nm, y1_nm = meta["reference_bbox_nm"]
    ppn = meta["pixels_per_nm"]
    content_h_px = meta["search_bbox_nm"][3] * ppn

    rx0 = int(round(x0_nm * ppn))
    rx1 = int(round(x1_nm * ppn))
    ry0 = int(round(content_h_px - y1_nm * ppn))
    ry1 = int(round(content_h_px - y0_nm * ppn))
    cx = int(round(meta["reference_center_nm"][0] * ppn))
    cy = int(round(content_h_px - meta["reference_center_nm"][1] * ppn))
    cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), DEBUG_REF, 2)
    cv2.circle(canvas, (cx, cy), 4, DEBUG_REF, -1)
    return canvas


# ═════════════════════════════════════════════════════════════════════════
# 3. Reference/search sampling (reference_search_pairs.py)
# ═════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS_KW = dict(
    rows=64, cols=64,
    cell_pitch_nm=80.0, cell_pitch_bl_nm=60.0,
    wl_width_nm=24.0, bl_width_nm=18.0,
    contact_size_nm=40.0, capacitor_size_nm=30.0,
    n_particles=3, n_scratches=2,
    p_broken_wl=0.02, p_broken_bl=0.02, p_cmp_dishing=0.0,
    overlay_sigma_nm=6.0, linewidth_sigma_nm=4.0, ler_amplitude_nm=3.0,
    contact_sigma_frac=0.15, jitter_sigma_nm=2.5,
)

@dataclass
class ReferenceSearchSample:
    reference_img: np.ndarray    # uint8, grayscale
    search_img: np.ndarray       # uint8, grayscale, 1000x1000
    true_center_px: Tuple[float, float]
    zoom_ratio: float
    seed: int
    # Bounding box (x0, y0, x1, y1), in search-image pixel coords, of a
    # particle/scratch defect that overlaps the reference patch -- for
    # visualization only (e.g. the benchmark's example PNGs). None if the
    # reference patch (chosen uniform-randomly, never biased toward
    # defects) doesn't happen to contain one. Not used for scoring.
    defect_bbox_px: Optional[Tuple[float, float, float, float]] = None
    # Ground truth for the single_training_defect mode (dram_localization_pipeline.py
    # DRAMParams.single_training_defect). None unless that flag was set.
    training_defect_center_px: Optional[Tuple[float, float]] = None
    training_defect_bbox_px: Optional[Tuple[float, float, float, float]] = None
    training_defect_tile: Optional[Tuple[int, int]] = None   # (row, col), each 0-9


def _find_reference_defect(meta: dict) -> Optional[dict]:
    """First particle/scratch defect whose bbox overlaps the reference
    patch, or None. cmp_dishing and the missing-feature/broken-line types
    aren't point-like and aren't rendered with a distinct visual marker, so
    they're not useful to circle in a QA image. training_scratch (from
    single_training_defect) is intentionally excluded too -- its overlap
    info is carried by the dedicated training_defect_* fields instead, so
    don't "fix" this to include it."""
    rx0, ry0, rx1, ry1 = meta["reference_bbox_nm"]
    for d in meta["defect_locations"]:
        if d["type"] not in ("particle", "scratch"):
            continue
        bx0, by0, bx1, by1 = d["bbox_nm"]
        if bx1 > rx0 and bx0 < rx1 and by1 > ry0 and by0 < ry1:
            return d
    return None


def _defect_bbox_to_search_px(bbox_nm: list, pixels_per_nm: float,
                               content_h_px: float) -> tuple:
    """Convert a defect's nm bbox into search-image pixel coords, applying
    the same y-flip used for true_center_px."""
    x0, y0, x1, y1 = bbox_nm
    xs = [x0 * pixels_per_nm, x1 * pixels_per_nm]
    ys = [content_h_px - y0 * pixels_per_nm, content_h_px - y1 * pixels_per_nm]
    return (min(xs), min(ys), max(xs), max(ys))


def generate_sample(seed: int, tmp_dir: str, **params_kw) -> ReferenceSearchSample:
    """Generate one independent reference/search pair for the given seed."""
    kw = {**DEFAULT_PARAMS_KW, **params_kw}
    stem = f"pair_{seed:06d}"
    gds_path  = str(Path(tmp_dir) / f"{stem}.gds")
    json_path = str(Path(tmp_dir) / f"{stem}_meta.json")

    p = DRAMParams(**kw, seed=seed, output_gds=gds_path, output_json=json_path)
    gen = DieGenerator(p)
    meta = gen.generate()

    seed_seq = np.random.SeedSequence(seed)
    search_seed, ref_seed = seed_seq.spawn(2)
    rng_search = np.random.default_rng(search_seed)
    rng_ref = np.random.default_rng(ref_seed)

    # -- Search image: whole array at native resolution, with noise --
    search_img = render_sem_image(gds_path, meta, rng_search)

    # meta["reference_center_px"] is a plain nm->px scale (y increases downward
    # in nm == increases in pixel row). The rasterizer above flips y (image
    # row 0 is *maximum* nm-y), so the label must be flipped into the same
    # frame as the rendered pixels, or it names the wrong row entirely.
    raw_cx, raw_cy = meta["reference_center_px"]
    content_h_px = meta["search_bbox_nm"][3] * meta["pixels_per_nm"]
    true_center_px = (raw_cx, content_h_px - raw_cy)

    # -- Reference image: same physical patch at zoom_ratio x finer pixel pitch --
    zoom_ratio = float(meta["zoom_ratio"])
    ref_w_px = int(round(meta["reference_image_size_px"][0] * zoom_ratio))
    ref_h_px = int(round(meta["reference_image_size_px"][1] * zoom_ratio))
    ppn_hi = meta["pixels_per_nm"] * zoom_ratio

    reference_img = render_sem_patch(gds_path, meta["reference_bbox_nm"], ppn_hi,
                                      ref_w_px, ref_h_px, rng_ref,
                                      defect_locations=meta["defect_locations"])

    defect = _find_reference_defect(meta)
    defect_bbox_px = None
    if defect is not None:
        defect_bbox_px = _defect_bbox_to_search_px(
            defect["bbox_nm"], meta["pixels_per_nm"], content_h_px)

    training_defect_center_px = None
    training_defect_bbox_px = None
    training_defect_tile = None
    training_defect = next(
        (d for d in meta["defect_locations"] if d["type"] == "training_scratch"), None)
    if training_defect is not None:
        tx0, ty0, tx1, ty1 = _defect_bbox_to_search_px(
            training_defect["bbox_nm"], meta["pixels_per_nm"], content_h_px)
        training_defect_bbox_px = (tx0, ty0, tx1, ty1)
        tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
        training_defect_center_px = (tcx, tcy)
        training_defect_tile = (min(9, max(0, int(tcy // 100))),
                                 min(9, max(0, int(tcx // 100))))

    for f in (gds_path, json_path):
        Path(f).unlink(missing_ok=True)

    return ReferenceSearchSample(
        reference_img=reference_img,
        search_img=search_img,
        true_center_px=true_center_px,
        zoom_ratio=zoom_ratio,
        seed=seed,
        defect_bbox_px=defect_bbox_px,
        training_defect_center_px=training_defect_center_px,
        training_defect_bbox_px=training_defect_bbox_px,
        training_defect_tile=training_defect_tile,
    )


# ═════════════════════════════════════════════════════════════════════════
# 4. SuperPoint + LightGlue matching (matcher.py)
# ═════════════════════════════════════════════════════════════════════════
# Install: pip install "git+https://github.com/cvg/LightGlue.git"
# Weights download automatically (via torch.hub) on first use and are cached
# in ~/.cache/torch/hub/checkpoints/ afterward -- first construction of
# SuperPointLightGlueMatcher requires network access; subsequent runs do not.

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


# ═════════════════════════════════════════════════════════════════════════
# 5. Sequential-RANSAC similarity-transform localization (localizer.py)
# ═════════════════════════════════════════════════════════════════════════
# The dataset only ever applies translation and a known uniform zoom -- no
# rotation, no perspective distortion -- so a similarity transform (uniform
# scale + translation, cv2.estimateAffinePartial2D) is the correct geometric
# model. Full homography is out of scope unless perspective distortion is
# introduced to the dataset later.

@dataclass
class LocalizationResult:
    predicted_center_px: Optional[Tuple[float, float]]
    confidence: float
    elapsed_ms: float
    num_matches: int
    num_inliers: int
    success: bool
    failure_reason: Optional[str]


class DeepMatchLocalizer:
    def __init__(self, matcher, min_matches: int = 8, min_inliers: int = 4,
                 max_hypotheses: int = 3, ransac_reproj_threshold: float = 3.0):
        self.matcher = matcher
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.max_hypotheses = max_hypotheses
        self.ransac_reproj_threshold = ransac_reproj_threshold

    def localize(self, reference_img: np.ndarray, search_img: np.ndarray,
                 zoom_ratio: float) -> LocalizationResult:
        t0 = time.perf_counter()

        ref_h, ref_w = reference_img.shape[:2]
        low_w = max(1, round(ref_w / zoom_ratio))
        low_h = max(1, round(ref_h / zoom_ratio))
        ref_lowres = cv2.resize(reference_img, (low_w, low_h), interpolation=cv2.INTER_AREA)

        feats_ref = self.matcher.extract(ref_lowres)
        feats_search = self.matcher.extract(search_img)
        matches = self.matcher.match(feats_ref, feats_search)
        num_matches = len(matches.scores)

        if num_matches < self.min_matches:
            return LocalizationResult(None, 0.0, _elapsed_ms(t0), num_matches, 0,
                                       False, "insufficient_matches")

        search_h, search_w = search_img.shape[:2]
        search_center = (search_w / 2.0, search_h / 2.0)
        ref_center = np.array([low_w / 2.0, low_h / 2.0], dtype=np.float64)

        hypotheses = self._find_hypotheses(matches, ref_center)

        if not hypotheses:
            return LocalizationResult(None, 0.0, _elapsed_ms(t0), num_matches, 0,
                                       False, "ransac_failed")

        best = min(hypotheses, key=lambda h: _dist(h["predicted_center"], search_center))

        inlier_ratio = best["num_inliers"] / num_matches
        confidence = inlier_ratio * best["mean_score"]
        if not (0.5 <= best["scale"] <= 2.0):
            confidence *= 0.5

        return LocalizationResult(
            predicted_center_px=best["predicted_center"],
            confidence=float(confidence),
            elapsed_ms=_elapsed_ms(t0),
            num_matches=num_matches,
            num_inliers=best["num_inliers"],
            success=True,
            failure_reason=None,
        )

    def _find_hypotheses(self, matches, ref_center: np.ndarray) -> list:
        pts_a, pts_b, scores = matches.kpts_a, matches.kpts_b, matches.scores
        remaining = np.arange(len(scores))
        hypotheses = []

        for _ in range(self.max_hypotheses):
            if len(remaining) < self.min_inliers:
                break
            M, inlier_mask = cv2.estimateAffinePartial2D(
                pts_a[remaining], pts_b[remaining],
                method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_threshold)
            if M is None:
                break
            inlier_mask = inlier_mask.ravel().astype(bool)
            n_inliers = int(inlier_mask.sum())
            if n_inliers < self.min_inliers:
                break

            inlier_idx = remaining[inlier_mask]
            predicted = M @ np.array([ref_center[0], ref_center[1], 1.0])
            hypotheses.append({
                "predicted_center": (float(predicted[0]), float(predicted[1])),
                "num_inliers": n_inliers,
                "mean_score": float(scores[inlier_idx].mean()),
                "scale": float(np.hypot(M[0, 0], M[1, 0])),
            })
            remaining = remaining[~inlier_mask]

        return hypotheses


def _dist(a: tuple, b: tuple) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


# ═════════════════════════════════════════════════════════════════════════
# 6. Benchmark CLI (benchmark.py)
# ═════════════════════════════════════════════════════════════════════════

EXAMPLE_SEARCH_SEED_OFFSET = 10_000_000  # disjoint from the statistical seed range
EXAMPLE_SEARCH_MAX_ATTEMPTS = 500


def run_benchmark(n: int, tolerance_px: float, out_dir: Path, n_examples: int = 5,
                   seed_start: int = 0) -> dict:
    matcher = SuperPointLightGlueMatcher()
    localizer = DeepMatchLocalizer(matcher)

    records = []
    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(n):
            seed = seed_start + i
            sample = generate_sample(seed, tmp_dir)
            result = localizer.localize(sample.reference_img, sample.search_img, sample.zoom_ratio)

            record = {
                "seed": seed,
                "success": result.success,
                "failure_reason": result.failure_reason,
                "predicted_center_px": result.predicted_center_px,
                "true_center_px": sample.true_center_px,
                "confidence": result.confidence,
                "num_matches": result.num_matches,
                "num_inliers": result.num_inliers,
                "elapsed_ms": result.elapsed_ms,
            }
            if result.success:
                err = float(np.hypot(result.predicted_center_px[0] - sample.true_center_px[0],
                                      result.predicted_center_px[1] - sample.true_center_px[1]))
                record["pixel_error"] = err
                record["within_tolerance"] = err <= tolerance_px
            else:
                record["pixel_error"] = None
                record["within_tolerance"] = False
            records.append(record)

        # Example PNGs are saved separately from the statistical n samples
        # above (which stay uniform-random and unbiased, per design) --
        # searching a disjoint seed range specifically for defect-containing
        # reference patches makes the illustrative examples actually show
        # the defect rendering, without touching what's measured.
        if n_examples > 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for k in range(EXAMPLE_SEARCH_MAX_ATTEMPTS):
                if saved >= n_examples:
                    break
                example_seed = EXAMPLE_SEARCH_SEED_OFFSET + k
                sample = generate_sample(example_seed, tmp_dir)
                if sample.defect_bbox_px is None:
                    continue
                result = localizer.localize(sample.reference_img, sample.search_img, sample.zoom_ratio)
                _save_example(sample, result, out_dir / f"example_{example_seed:04d}.png")
                saved += 1

    n_success = sum(r["success"] for r in records)
    n_within = sum(r["within_tolerance"] for r in records)
    times = [r["elapsed_ms"] for r in records]
    errors = [r["pixel_error"] for r in records if r["pixel_error"] is not None]

    summary = {
        "n_samples": n,
        "tolerance_px": tolerance_px,
        "n_success": n_success,
        "n_within_tolerance": n_within,
        "success_rate_pct": 100.0 * n_within / n,
        "compute_time_ms": {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p95": _percentile(times, 95),
            "max": max(times),
        },
        "pixel_error": {
            "mean": statistics.mean(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "max": max(errors) if errors else None,
        },
        "records": records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def _percentile(values: list, pct: float):
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1))))
    return s[idx]


EXAMPLE_DEFECT_BOX_MARGIN_PX = 4  # visual padding so the square clears the defect's edges


def _save_example(sample, result, path: Path) -> None:
    canvas = cv2.cvtColor(sample.search_img, cv2.COLOR_GRAY2BGR)

    if sample.defect_bbox_px is not None:
        bx0, by0, bx1, by1 = sample.defect_bbox_px
        gx = (bx0 + bx1) / 2.0
        gy = (by0 + by1) / 2.0
        m = EXAMPLE_DEFECT_BOX_MARGIN_PX
        cv2.rectangle(canvas,
                      (int(round(bx0)) - m, int(round(by0)) - m),
                      (int(round(bx1)) + m, int(round(by1)) + m),
                      (0, 255, 0), 2)
    else:
        gx, gy = sample.true_center_px
    cv2.drawMarker(canvas, (int(round(gx)), int(round(gy))), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    if result.success:
        px, py = [int(round(v)) for v in result.predicted_center_px]
        cv2.drawMarker(canvas, (px, py), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
    cv2.imwrite(str(path), canvas)


def main():
    ap = argparse.ArgumentParser(description="SuperPoint+LightGlue localization benchmark")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tolerance-px", type=float, default=5.0)
    ap.add_argument("--out-dir", type=str, default="localization_results")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=5)
    args = ap.parse_args()

    summary = run_benchmark(args.n, args.tolerance_px, Path(args.out_dir),
                             n_examples=args.n_examples, seed_start=args.seed_start)

    print(f"\n{args.n} samples, tolerance={args.tolerance_px}px")
    print(f"  Success rate      : {summary['success_rate_pct']:.1f}%  "
          f"({summary['n_within_tolerance']}/{args.n} within tolerance)")
    print(f"  Localizer success : {summary['n_success']}/{args.n} (matches+RANSAC found a hypothesis)")
    print(f"  Compute time (ms) : mean={summary['compute_time_ms']['mean']:.1f}  "
          f"median={summary['compute_time_ms']['median']:.1f}  "
          f"p95={summary['compute_time_ms']['p95']:.1f}")
    if summary["pixel_error"]["mean"] is not None:
        print(f"  Pixel error       : mean={summary['pixel_error']['mean']:.2f}  "
              f"median={summary['pixel_error']['median']:.2f}  "
              f"max={summary['pixel_error']['max']:.2f}")
    print(f"  Results written to: {args.out_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()
