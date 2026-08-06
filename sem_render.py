"""Shared SEM-style rasterization: opaque supersampled painting, grayscale
SEM noise pipeline, periodic-margin tiling. Used by generate_dram_images.py
(whole-array render) and reference_search_pairs.py (high-res patch render).
"""
import cv2
import numpy as np
import klayout.db as db

IMG_SIZE  = 1000
SS_FACTOR = 4
SS_SIZE   = IMG_SIZE * SS_FACTOR

SEM_LAYER_ORDER = [(6, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
SEM_INTENSITY = {
    (6, 0):  40,   # Dummy      - background fill, dim
    (1, 0): 150,   # Capacitor  - dielectric/poly, mid
    (2, 0): 170,   # WordLine   - poly gate, mid-bright
    (3, 0): 190,   # BitLine    - metal-ish, bright
    (4, 0): 255,   # Contact    - metal plug, brightest (highest SE yield)
    (5, 0): 100,   # Defect     - charging artifact, darker
}
SEM_BG = 25        # bare substrate

DEBUG_LAYER_STYLE = [
    ((6, 0), ( 45,  50,  60)),
    ((1, 0), (220,  80,  80)),
    ((2, 0), ( 60, 150, 255)),
    ((3, 0), ( 60, 220, 130)),
    ((4, 0), (255, 200,  80)),
    ((5, 0), (190, 100, 255)),
]
DEBUG_BG  = (10, 12, 18)
DEBUG_REF = (255, 255, 0)

CIRCLE_LAYERS = {(4, 0)}  # contacts render as inscribed circles, not rectangles

DEFECT_HALO_MARGIN_NM = 15.0
DEFECT_HALO_INTENSITY = 200
DEFECT_MARK_INTENSITY = 30
DISHING_INTENSITY     = 140

POINT_DEFECT_TYPES = {"particle", "scratch"}


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
    Two-tone defect rendering, in place on `intensity` (same 0-255 scale
    as SEM_INTENSITY). Call after the generic per-layer fill so this
    overrides the flat defect-layer intensity within each defect's
    footprint:
      - particle/scratch: a lighter "disturbed-zone" halo (inflated bbox)
        with the actual defect mark (original bbox) drawn on top, darker.
      - cmp_dishing: a softer translucent tint over the (already large,
        diffuse) affected region -- not the halo+mark treatment.
    A defect only partially inside origin_bbox_nm is naturally clipped by
    the same bounds-checking used elsewhere in this module.
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

    rx0, ry0, rx1, ry1 = [int(round(v)) for v in meta["reference_bbox_px"]]
    cx = int(round(meta["reference_center_px"][0]))
    cy = int(round(meta["reference_center_px"][1]))
    cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), DEBUG_REF, 2)
    cv2.circle(canvas, (cx, cy), 4, DEBUG_REF, -1)
    return canvas
