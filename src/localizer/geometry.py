"""Coordinate geometry for the localizer.

Single source of truth for the mapping between correlation-map cells and
Search-image pixels. Pure math -- no torch, no cv2 -- so it is cheap to
import and to test exhaustively.

The mapping is x = STRIDE*c + HALF + align_offset. `align_offset` is an
empirically calibrated constant (see encoder.calibrate_alignment): the
derivation assumes feature cell i corresponds to input pixel STRIDE*i, but
the true alignment depends on the ResNet stem's padding and must be
measured rather than assumed.
"""

SCALE = 10                          # reference nm/px : search nm/px
STRIDE = 4                          # encoder output stride
REF_PX = 1000                       # reference native size
SEARCH_PX = 1000                    # search native size
REF_DS_PX = REF_PX // SCALE         # 100 -- reference at search scale
HALF = REF_DS_PX // 2               # 50 -- template half-size
REF_FEAT = REF_DS_PX // STRIDE      # 25
SEARCH_FEAT = SEARCH_PX // STRIDE   # 250
CORR = SEARCH_FEAT - REF_FEAT + 1   # 226 -- valid-mode correlation size


def cell_to_pixel(c, align_offset: float = 0.0) -> float:
    """Correlation cell index -> Search-image pixel coordinate of the
    template centre."""
    return STRIDE * float(c) + HALF + align_offset


def pixel_to_cell(x: float, align_offset: float = 0.0) -> int:
    """Search-image pixel -> nearest correlation cell, clamped to the grid."""
    c = int(round((float(x) - HALF - align_offset) / STRIDE))
    return max(0, min(c, CORR - 1))


def pixel_to_cell_offset(x: float, align_offset: float = 0.0):
    """Split a pixel coordinate into (cell, residual). The residual is what
    the offset head must predict, and lies in [-STRIDE/2, +STRIDE/2].
    """
    c = pixel_to_cell(x, align_offset)
    return c, float(x) - cell_to_pixel(c, align_offset)
