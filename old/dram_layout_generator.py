"""
DRAM Layout Generator
=====================
Generates a fully synthetic DRAM memory-cell-array GDSII layout using
the KLayout Python API (klayout.db). No proprietary process data.

Why each structure is not proprietary:
  Capacitor nodes  — 1T-1C concept, Dennard (1968), expired patent
  Word lines       — Sze & Ng (2006) Physics of Semiconductor Devices §6
  Bit lines        — Razavi (2002) Design of Analog CMOS ICs §12
  Contacts         — Jaeger (2002) Intro to Microelectronic Fabrication Ch.6
  Defects          — Postek & Vladár (2011) SEM metrology literature
  Dummy cells      — JEDEC JESD79 array margin rules

Cell geometry is fully abstracted: cell_pitch_nm is a free parameter
with no F² constraint. No real process node is implied.
"""
from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import klayout.db as db


# ── GDS layer constants ────────────────────────────────────────────────────
LAYER_CAPACITOR = (1, 0)   # storage node rectangles
LAYER_WORDLINE = (2, 0)    # horizontal WL stripes
LAYER_BITLINE = (3, 0)     # vertical BL stripes
LAYER_CONTACT = (4, 0)     # small square contacts
LAYER_DEFECT = (5, 0)      # particles, scratches, dishing
LAYER_DUMMY = (6, 0)       # boundary dummy fills


@dataclass(frozen=True)
class DRAMParams:
    """All generator parameters. Frozen — never mutated after construction."""

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
    overlay_sigma_nm: float = 3.0     # layer-to-layer misalignment σ
    linewidth_sigma_nm: float = 2.0   # CD variation σ (half applied per edge)
    ler_amplitude_nm: float = 1.5     # line-edge roughness amplitude
    ler_period_nm: float = 20.0       # LER spatial correlation length
    contact_sigma_frac: float = 0.08  # fractional σ of contact CD
    jitter_sigma_nm: float = 1.0      # cell-placement jitter σ

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
    dbu_nm: float = 1.0               # 1 DBU = 1 nm (layout.dbu = dbu_nm*1e-3 µm)

    # ── Rasterization / localization benchmark ──────────────────────────────
    output_width_px: int = 1000       # search image width in pixels
    output_height_px: int = 1000      # search image height in pixels
    reference_width_px: int = 100     # reference crop width in pixels
    reference_height_px: int = 100    # reference crop height in pixels
    zoom_ratio: int = 10              # output_width_px / reference_width_px
    pixels_per_nm: Optional[float] = None
    # If None, auto-computed: min(output_width_px/array_w_nm,
    #                             output_height_px/array_h_nm)


@dataclass(frozen=True)
class RasterizationConfig:
    """
    Everything a rasterizer needs to convert the GDS to a 1000×1000 image.
    Returned by DRAMGenerator.get_rasterization_config(). No rasterization
    is performed here — this is a pure parameter bundle.
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
    Pure geometric perturbation methods — no KLayout cell/layout calls.
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
        axis='x': line is horizontal — vary height (y dimension).
        axis='y': line is vertical   — vary width  (x dimension).
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
        axis='x': horizontal line — perturb top/bottom edges per segment.
        axis='y': vertical line   — perturb left/right edges per segment.
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
        axis='x': horizontal — remove a x-range sub-segment.
        axis='y': vertical   — remove a y-range sub-segment.
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
        dish centre (cx, cy). Models CMP dishing — a polishing artefact
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
        self.layout.dbu = params.dbu_nm * 1e-3   # 1 DBU = 1 nm (dbu in µm)
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

                if self.pv.is_missing(p.p_missing_capacitor):
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

            if self.pv.is_missing(p.p_broken_wl):
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

            if self.pv.is_missing(p.p_broken_bl):
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

                if self.pv.is_missing(p.p_missing_contact):
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
            (Dishing affects WL/BL shape records for metadata only —
             geometry already placed on Layers 2/3. We record the region.)
        Returns list of new defect dicts (particles + scratches + dishing).
        """
        p = self.p
        full = self._full_bbox()
        new_defects: List[dict] = []

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

        # CMP dishing — shrink WL/BL geometry in a circular region, then mark it
        if p.p_cmp_dishing > 0:
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

        # Reference patch selection — random window fully inside the active array
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


# ── Module-level entry point ───────────────────────────────────────────────

def generate_dram_layout(params: DRAMParams = DRAMParams()) -> dict:
    """Generate synthetic DRAM GDS layout. Returns metadata dict."""
    return DRAMGenerator(params).generate()


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate a synthetic DRAM GDSII layout.")
    ap.add_argument("--rows",               type=int,   default=64)
    ap.add_argument("--cols",               type=int,   default=64)
    ap.add_argument("--cell-pitch-nm",      type=float, default=80.0)
    ap.add_argument("--cell-pitch-bl-nm",   type=float, default=60.0)
    ap.add_argument("--wl-width-nm",        type=float, default=24.0)
    ap.add_argument("--bl-width-nm",        type=float, default=18.0)
    ap.add_argument("--contact-size-nm",    type=float, default=14.0)
    ap.add_argument("--capacitor-size-nm",  type=float, default=30.0)
    ap.add_argument("--dummy-rows",         type=int,   default=2)
    ap.add_argument("--dummy-cols",         type=int,   default=2)
    ap.add_argument("--overlay-sigma-nm",   type=float, default=3.0)
    ap.add_argument("--linewidth-sigma-nm", type=float, default=2.0)
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--output-gds",         type=str,   default="dram.gds")
    ap.add_argument("--output-json",        type=str,   default="dram_metadata.json")
    args = ap.parse_args()

    params = DRAMParams(
        rows=args.rows,
        cols=args.cols,
        cell_pitch_nm=args.cell_pitch_nm,
        cell_pitch_bl_nm=args.cell_pitch_bl_nm,
        wl_width_nm=args.wl_width_nm,
        bl_width_nm=args.bl_width_nm,
        contact_size_nm=args.contact_size_nm,
        capacitor_size_nm=args.capacitor_size_nm,
        dummy_rows=args.dummy_rows,
        dummy_cols=args.dummy_cols,
        overlay_sigma_nm=args.overlay_sigma_nm,
        linewidth_sigma_nm=args.linewidth_sigma_nm,
        seed=args.seed,
        output_gds=args.output_gds,
        output_json=args.output_json,
    )

    meta = generate_dram_layout(params)
    print(f"Generated: {args.output_gds}")
    print(f"  Cells:    {len(meta['cell_centers'])}")
    print(f"  WLs:      {len(meta['wordline_coords'])}")
    print(f"  BLs:      {len(meta['bitline_coords'])}")
    print(f"  Defects:  {len(meta['defect_locations'])}")
    print(f"  Metadata: {args.output_json}")
