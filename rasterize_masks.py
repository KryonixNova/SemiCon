import sys
import os
# Remove Desktop from sys.path so the local gdstk.py doesn't shadow the package
_desktop = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _desktop]

import gdstk
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DESKTOP = "/home/nihal/Desktop"
OUT     = os.path.join(DESKTOP, "masks")
os.makedirs(OUT, exist_ok=True)


def gds_to_mask(gds_path, layer=(1, 0), resolution=512, margin_nm=20.0):
    """
    Read one layer from a GDSII file and rasterize it to a binary float32 mask.
    Returns: (mask np.ndarray [0,1], nm_per_pixel float)
    """
    lib  = gdstk.read_gds(gds_path)
    cell = lib.top_level()[0]

    # gdstk stores coordinates in µm; multiply by 1000 → nm
    polys = [
        p.points * 1000
        for p in cell.get_polygons(layer=layer[0], datatype=layer[1])
    ]
    if not polys:
        raise ValueError(f"No polygons on layer {layer} in {gds_path}")

    all_pts = np.vstack(polys)
    xmin, ymin = all_pts.min(axis=0) - margin_nm
    xmax, ymax = all_pts.max(axis=0) + margin_nm
    extent_nm  = max(xmax - xmin, ymax - ymin)
    scale      = resolution / extent_nm          # px / nm
    nm_per_px  = 1.0 / scale

    img  = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(img)
    for pts in polys:
        px = [((x - xmin) * scale, (ymax - y) * scale) for x, y in pts]
        draw.polygon(px, fill=255)

    mask = np.array(img, dtype=np.float32) / 255.0
    return mask, nm_per_px


# ── Layout definitions ────────────────────────────────────────────────────
LAYOUTS = {
    "finfet": {
        "gds": "finfet.gds",
        "layers": [
            ((1, 0), "Fins / Active",        "#4FC3F7"),
            ((2, 0), "Gate / Poly",           "#EF9A9A"),
        ],
    },
    "dram": {
        "gds": "dram.gds",
        "layers": [
            ((1, 0), "Capacitor Nodes",       "#4FC3F7"),
            ((2, 0), "Word Lines",            "#EF9A9A"),
            ((3, 0), "Bit Lines",             "#A5D6A7"),
        ],
    },
    "sram": {
        "gds": "sram.gds",
        "layers": [
            ((1, 0), "Active Regions",        "#4FC3F7"),
            ((2, 0), "Poly / Gate",           "#EF9A9A"),
            ((3, 0), "Metal Interconnect",    "#FFF176"),
        ],
    },
    "nand": {
        "gds": "nand.gds",
        "layers": [
            ((1, 0), "Channel / Tun.Ox.",     "#4FC3F7"),
            ((2, 0), "Floating Gates",        "#EF9A9A"),
            ((3, 0), "Word Lines (CG)",       "#A5D6A7"),
        ],
    },
}

# ── Rasterize all layers ──────────────────────────────────────────────────
results = {}
print(f"\n{'Layout':<8}  {'Layer':<26}  {'File':<30}  {'nm/px':>7}  {'Fill%':>7}")
print("─" * 84)

for name, spec in LAYOUTS.items():
    results[name] = []
    for layer, label, color in spec["layers"]:
        gds_path = os.path.join(DESKTOP, spec["gds"])
        try:
            mask, nm_per_px = gds_to_mask(gds_path, layer=layer)
            cov   = mask.mean() * 100
            fname = f"{name}_L{layer[0]}DT{layer[1]}.png"
            fpath = os.path.join(OUT, fname)
            Image.fromarray((mask * 255).astype(np.uint8)).save(fpath)
            results[name].append((layer, label, color, mask, nm_per_px, cov))
            print(f"{name:<8}  {label:<26}  {fname:<30}  {nm_per_px:>7.3f}  {cov:>6.1f}%")
        except ValueError as e:
            print(f"{name:<8}  {label:<26}  ERROR: {e}")

print(f"\nIndividual masks saved → {OUT}/")

# ── Composite preview figure ──────────────────────────────────────────────
print("Rendering composite figure ...")

max_layers = max(len(v) for v in results.values())
n_rows     = len(results)

fig = plt.figure(figsize=(4 * (max_layers + 1), 4 * n_rows), facecolor="#0C1018")
fig.suptitle(
    "GDSII  →  Binary Mask Rasterization   |   All Layouts & Layers",
    color="white", fontsize=13, fontweight="bold",
    fontfamily="monospace", y=0.995
)
gs = gridspec.GridSpec(
    n_rows, max_layers + 1,
    figure=fig, hspace=0.40, wspace=0.08,
    left=0.02, right=0.98, top=0.96, bottom=0.02
)

LAYER_TINTS = ["#4FC3F7", "#EF9A9A", "#A5D6A7", "#FFF176"]

for row, (name, layer_data) in enumerate(results.items()):
    # Composite RGB image (each layer gets a distinct tint)
    composite = np.zeros((512, 512, 3), dtype=np.float32)
    for li, (layer, label, color, mask, nm_per_px, cov) in enumerate(layer_data):
        hx = LAYER_TINTS[li].lstrip("#")
        r, g, b = [int(hx[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        composite[..., 0] += mask * r
        composite[..., 1] += mask * g
        composite[..., 2] += mask * b
    composite = np.clip(composite, 0, 1)

    ax_c = fig.add_subplot(gs[row, 0])
    ax_c.imshow(composite)
    ax_c.set_title(
        f"{name.upper()}\nComposite ({len(layer_data)} layers)",
        color="#7B9CFF", fontsize=9, fontfamily="monospace",
        fontweight="bold", pad=5
    )
    ax_c.axis("off")

    # Per-layer grayscale panels
    for col, (layer, label, color, mask, nm_per_px, cov) in enumerate(layer_data):
        ax = fig.add_subplot(gs[row, col + 1])
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"Layer {layer[0]} / DT{layer[1]}\n{label}\n{nm_per_px:.2f} nm/px",
            color="#C8D3E8", fontsize=8, fontfamily="monospace", pad=4
        )
        ax.text(
            0.97, 0.03, f"{cov:.1f}% fill",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#A5D6A7", fontsize=7, fontfamily="monospace"
        )
        ax.axis("off")

    # Fill empty columns in this row
    for col in range(len(layer_data), max_layers):
        ax = fig.add_subplot(gs[row, col + 1])
        ax.set_facecolor("#0C1018")
        ax.axis("off")

fig_path = os.path.join(OUT, "all_masks_preview.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0C1018")
plt.close(fig)
print(f"Preview figure saved → {fig_path}")
print("\nStep 2 complete.")
