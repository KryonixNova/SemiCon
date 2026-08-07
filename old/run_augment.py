"""
Step 4 — OpenCV Post-Processing & Augmentation
Reads resist images from Desktop/litho_output/,
applies SEM-realistic augmentations + synthetic defects,
saves results to Desktop/augmented/.
"""
import sys, os
_desktop = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _desktop)   # need augment.py from Desktop

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from augment import augment, inject_defect

DESKTOP  = "/home/nihal/Desktop"
LITHO_DIR = os.path.join(DESKTOP, "litho_output")
OUT_DIR   = os.path.join(DESKTOP, "augmented")
os.makedirs(OUT_DIR, exist_ok=True)

DEFECT_KINDS = ["bridge", "open", "particle", "scratch"]

# All resist images produced in step 3
RESIST_FILES = sorted([
    f for f in os.listdir(LITHO_DIR)
    if f.endswith("_resist.png")
])

rng = np.random.default_rng(42)

# ── Per-image labels ─────────────────────────────────────────────────────
LABELS = {
    "finfet_L1DT0": "FinFET · Fins",
    "finfet_L2DT0": "FinFET · Gate",
    "dram_L1DT0":   "DRAM · Cap. Nodes",
    "dram_L2DT0":   "DRAM · Word Lines",
    "dram_L3DT0":   "DRAM · Bit Lines",
    "sram_L1DT0":   "SRAM · Active",
    "sram_L2DT0":   "SRAM · Poly/Gate",
    "sram_L3DT0":   "SRAM · Metal",
    "nand_L1DT0":   "NAND · Channel",
    "nand_L2DT0":   "NAND · Float. Gates",
    "nand_L3DT0":   "NAND · Word Lines",
}

# ── Process each resist image ────────────────────────────────────────────
records = []

print(f"\n{'Resist image':<30}  {'Aug. μ':>7}  {'Aug. σ':>7}  {'Defect':<12}  Saved")
print("─" * 82)

for fname in RESIST_FILES:
    stem  = fname.replace("_resist.png", "")
    label = LABELS.get(stem, stem)

    resist_np = np.array(
        Image.open(os.path.join(LITHO_DIR, fname)).convert("L"),
        dtype=np.float32
    ) / 255.0

    # ── Augment ──────────────────────────────────────────────────────────
    aug = augment(
        resist_np,
        gaussian_sigma    = rng.uniform(0.02, 0.05),
        poisson_scale     = int(rng.integers(3000, 6000)),
        sem_texture_sigma = rng.uniform(0.01, 0.025),
        blur_sigma        = rng.uniform(0.5, 1.5),
        max_grad          = rng.uniform(0.05, 0.15),
        max_rotate_deg    = rng.uniform(0.5, 2.0),
        scale_range       = (rng.uniform(0.82, 0.90), 1.0),
        rng               = rng,
    )

    # ── Inject one defect (50 % chance) ──────────────────────────────────
    defect_kind = None
    if rng.random() < 0.5:
        defect_kind = rng.choice(DEFECT_KINDS)
        aug_defect  = inject_defect(aug, kind=defect_kind, rng=rng)
    else:
        aug_defect = aug

    # ── Save ─────────────────────────────────────────────────────────────
    clean_path  = os.path.join(OUT_DIR, f"{stem}_aug_clean.png")
    defect_path = os.path.join(OUT_DIR, f"{stem}_aug_defect.png")
    Image.fromarray((aug        * 255).astype(np.uint8)).save(clean_path)
    Image.fromarray((aug_defect * 255).astype(np.uint8)).save(defect_path)

    records.append({
        "stem":        stem,
        "label":       label,
        "resist":      resist_np,
        "aug":         aug,
        "aug_defect":  aug_defect,
        "defect_kind": defect_kind,
    })
    dk = defect_kind or "none"
    print(f"{fname:<30}  {aug.mean():>7.4f}  {aug.std():>7.4f}  {dk:<12}  ✓")

print(f"\nSaved to → {OUT_DIR}/")


# ── Preview figure: resist | clean aug | defect aug ──────────────────────
print("Rendering preview figure ...")

n   = len(records)
fig, axes = plt.subplots(n, 3, figsize=(13, 3.0 * n), facecolor="#0C1018")
fig.suptitle(
    "Step 4 — OpenCV Augmentation   |   Resist → Clean Aug → Defect Aug",
    color="white", fontsize=12, fontweight="bold",
    fontfamily="monospace", y=1.002
)

COL_TITLES = [
    "Resist Image  (litho output)",
    "Augmented — clean\n(noise · blur · gradient · crop)",
    "Augmented + Defect\n(bridge / open / particle / scratch)",
]

for col, title in enumerate(COL_TITLES):
    axes[0][col].set_title(title, color="#7B9CFF", fontsize=8.5,
                           fontfamily="monospace", fontweight="bold", pad=6)

for row, r in enumerate(records):
    imgs = [r["resist"], r["aug"], r["aug_defect"]]
    for col, img in enumerate(imgs):
        ax = axes[row][col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#243047")

    # Defect badge on right panel
    dk = r["defect_kind"]
    if dk:
        badge_colors = {
            "bridge":   "#EF9A9A",
            "open":     "#FFF176",
            "particle": "#A5D6A7",
            "scratch":  "#CE93D8",
        }
        axes[row][2].text(
            0.97, 0.03, f"⚠ {dk}", transform=axes[row][2].transAxes,
            ha="right", va="bottom", fontsize=7.5, fontfamily="monospace",
            color=badge_colors.get(dk, "white"),
            bbox=dict(facecolor="#0C1018", edgecolor=badge_colors.get(dk,"white"),
                      boxstyle="round,pad=0.3", linewidth=0.8)
        )
    else:
        axes[row][2].text(
            0.97, 0.03, "no defect", transform=axes[row][2].transAxes,
            ha="right", va="bottom", fontsize=7, fontfamily="monospace",
            color="#5E7099"
        )

    # Row label
    axes[row][0].set_ylabel(r["label"], color="#C8D3E8",
                            fontsize=8, fontfamily="monospace", labelpad=6)

fig.tight_layout(rect=[0, 0, 1, 1])
preview_path = os.path.join(OUT_DIR, "augment_preview.png")
fig.savefig(preview_path, dpi=130, bbox_inches="tight", facecolor="#0C1018")
plt.close(fig)
print(f"Preview → {preview_path}")
print("\nStep 4 complete.")
