import gdstk
import numpy as np
from PIL import Image, ImageDraw

def gds_to_mask(
    gds_path: str,
    layer: tuple = (1, 0),
    resolution: int = 512,
    margin_nm: float = 20.0,
) -> np.ndarray:
    """
    Convert a GDS layer to a binary float32 mask.
    GDS units assumed: 1 DBU = 1 nm (dbu=0.001 µm).

    Returns:
        np.ndarray, shape (resolution, resolution), dtype float32, values in {0,1}
    """
    lib = gdstk.read_gds(gds_path)
    cell = lib.top_level()[0]

    # Collect polygons from target layer
    polys = [
        p.points * 1000  # µm → nm  (gdstk stores in library units)
        for p in cell.get_polygons(layer=layer[0], datatype=layer[1])
    ]
    if not polys:
        raise ValueError(f"No polygons on layer {layer} in {gds_path}")

    all_pts = np.vstack(polys)
    xmin, ymin = all_pts.min(axis=0) - margin_nm
    xmax, ymax = all_pts.max(axis=0) + margin_nm
    w_nm = xmax - xmin
    h_nm = ymax - ymin
    scale = resolution / max(w_nm, h_nm)   # px / nm

    img = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(img)

    for pts in polys:
        px = [((x - xmin) * scale, (ymax - y) * scale) for x, y in pts]
        draw.polygon(px, fill=255)

    mask = np.array(img, dtype=np.float32) / 255.0
    return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mask = gds_to_mask("finfet.gds", layer=(1, 0), resolution=512)
    plt.imshow(mask, cmap="gray")
    plt.title("Binary Mask (FinFET fins, layer 1/0)")
    plt.colorbar()
    plt.savefig("mask_preview.png", dpi=150)
