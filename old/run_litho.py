"""
Step 3 — Optical Lithography Simulation
Reads binary masks from Desktop/masks/, runs Hopkins litho model,
saves aerial + resist images to Desktop/litho_output/.
"""
import sys, os
_desktop = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _desktop]

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.transforms.functional import gaussian_blur

DESKTOP = "/home/nihal/Desktop"
MASK_DIR = os.path.join(DESKTOP, "masks")
OUT_DIR  = os.path.join(DESKTOP, "litho_output")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ── Lithography simulator ────────────────────────────────────────────────

class LithoSim:
    """
    Hopkins partially coherent optical lithography simulator.
    wavelength_nm : exposure wavelength  (193 nm = ArF immersion)
    na            : numerical aperture   (1.35 = ArF immersion max)
    sigma         : partial coherence factor  (0 = coherent, 1 = incoherent)
    pixel_size_nm : physical size of one mask pixel in nm
    defocus_nm    : defocus (z-shift) in nm; 0 = best focus
    resist_threshold : aerial intensity that defines the resist edge (~0.225)
    resist_blur_nm   : resist diffusion length in nm
    """
    def __init__(self, wavelength_nm=193.0, na=1.35, sigma=0.30,
                 pixel_size_nm=1.0, defocus_nm=0.0,
                 resist_threshold=0.225, resist_blur_nm=5.0, device="cpu"):
        self.wl      = wavelength_nm
        self.na      = na
        self.sigma   = sigma
        self.px      = pixel_size_nm
        self.defocus = defocus_nm
        self.thresh  = resist_threshold
        self.rbsig   = resist_blur_nm / pixel_size_nm
        self.dev     = torch.device(device)

    def _freq_grid(self, N):
        f = torch.fft.fftfreq(N, d=self.px * self.na / self.wl)
        FX, FY = torch.meshgrid(f, f, indexing="ij")
        return FX.to(self.dev), FY.to(self.dev)

    def _pupil(self, N):
        FX, FY = self._freq_grid(N)
        r = torch.sqrt(FX**2 + FY**2)
        P = (r <= 1.0).to(torch.complex64)
        if self.defocus != 0:
            phase = (torch.pi * self.defocus * self.na**2 / self.wl) * r**2
            P = P * torch.exp(1j * phase)
        return P

    def _source(self, N):
        FX, FY = self._freq_grid(N)
        r  = torch.sqrt(FX**2 + FY**2)
        S  = (r <= self.sigma).float()
        return S / (S.sum() + 1e-12)

    def aerial(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute aerial image intensity [0,1] at wafer plane."""
        N    = mask.shape[-1]
        mask = mask.to(self.dev).to(torch.complex64)
        M    = torch.fft.fft2(mask)
        P    = self._pupil(N)
        S    = self._source(N)
        # Coherent amplitude
        amp  = torch.fft.ifft2(P * M)
        I    = amp.abs()**2
        # Partial coherence: convolve intensity with source
        I    = I.real.float()
        I    = torch.fft.ifft2(torch.fft.fft2(I) * torch.fft.fft2(S)).real.float()
        I    = torch.clamp(I, 0)
        return I / (I.max() + 1e-8)

    def resist(self, aerial: torch.Tensor) -> torch.Tensor:
        """Gaussian blur + sigmoid threshold resist model."""
        k       = max(int(self.rbsig * 4) | 1, 3)
        blurred = gaussian_blur(aerial.unsqueeze(0), [k, k], self.rbsig).squeeze(0)
        return torch.sigmoid(20.0 * (blurred - self.thresh))

    def simulate(self, mask: torch.Tensor) -> dict:
        ai = self.aerial(mask)
        ri = self.resist(ai)
        return {"aerial": ai.cpu(), "resist": ri.cpu()}


# ── Mask catalogue (file → physical pixel size in nm) ───────────────────
# pixel_size_nm tells the simulator the physical scale of each pixel.
# Derived from the rasterizer output: nm/px column from step 2.
MASK_CATALOGUE = {
    "finfet_L1DT0.png": dict(label="FinFET · Fins",          px_nm=0.475, layout="finfet"),
    "finfet_L2DT0.png": dict(label="FinFET · Gate",          px_nm=0.555, layout="finfet"),
    "dram_L1DT0.png":   dict(label="DRAM · Cap. Nodes",      px_nm=1.230, layout="dram"),
    "dram_L2DT0.png":   dict(label="DRAM · Word Lines",      px_nm=1.266, layout="dram"),
    "dram_L3DT0.png":   dict(label="DRAM · Bit Lines",       px_nm=1.266, layout="dram"),
    "sram_L1DT0.png":   dict(label="SRAM · Active",          px_nm=3.164, layout="sram"),
    "sram_L2DT0.png":   dict(label="SRAM · Poly/Gate",       px_nm=3.008, layout="sram"),
    "sram_L3DT0.png":   dict(label="SRAM · Metal",           px_nm=2.461, layout="sram"),
    "nand_L1DT0.png":   dict(label="NAND · Channel",         px_nm=1.191, layout="nand"),
    "nand_L2DT0.png":   dict(label="NAND · Float. Gates",    px_nm=1.191, layout="nand"),
    "nand_L3DT0.png":   dict(label="NAND · Word Lines",      px_nm=1.250, layout="nand"),
}

# Shared litho parameters (ArF immersion, best focus)
LITHO_PARAMS = dict(
    wavelength_nm    = 193.0,
    na               = 1.35,
    sigma            = 0.30,
    defocus_nm       = 0.0,
    resist_threshold = 0.225,
    resist_blur_nm   = 5.0,
    device           = DEVICE,
)


# ── Run simulation ───────────────────────────────────────────────────────
results = []

print(f"\n{'Mask file':<26}  {'px_nm':>6}  {'Aerial μ':>9}  {'Resist μ':>9}  {'Saved'}")
print("─" * 80)

for fname, meta in MASK_CATALOGUE.items():
    mask_path = os.path.join(MASK_DIR, fname)
    if not os.path.exists(mask_path):
        print(f"{fname:<26}  MISSING – run rasterize_masks.py first")
        continue

    mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask_t  = torch.from_numpy(mask_np)

    sim     = LithoSim(pixel_size_nm=meta["px_nm"], **LITHO_PARAMS)
    out     = sim.simulate(mask_t)

    aerial_np = out["aerial"].numpy()
    resist_np = out["resist"].numpy()

    stem          = fname.replace(".png", "")
    aerial_fname  = f"{stem}_aerial.png"
    resist_fname  = f"{stem}_resist.png"

    Image.fromarray((aerial_np * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, aerial_fname))
    Image.fromarray((resist_np * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, resist_fname))

    results.append({
        "fname":  fname,
        "label":  meta["label"],
        "layout": meta["layout"],
        "px_nm":  meta["px_nm"],
        "mask":   mask_np,
        "aerial": aerial_np,
        "resist": resist_np,
    })
    print(f"{fname:<26}  {meta['px_nm']:>6.3f}  "
          f"{aerial_np.mean():>9.4f}  {resist_np.mean():>9.4f}  "
          f"{aerial_fname}, {resist_fname}")

print(f"\nAll outputs saved → {OUT_DIR}/")


# ── Composite preview figure ─────────────────────────────────────────────
print("Rendering preview figure ...")

n = len(results)
fig, axes = plt.subplots(n, 3, figsize=(13, 3.2 * n), facecolor="#0C1018")
fig.suptitle(
    "Step 3 — Optical Lithography Simulation  (ArF 193nm · NA 1.35 · σ 0.30 · best focus)",
    color="white", fontsize=12, fontweight="bold", fontfamily="monospace", y=1.002
)

COL_TITLES = ["Mask  (input)", "Aerial Image  (wafer plane)", "Resist Image  (post develop)"]
COL_CMAPS  = ["gray", "inferno", "gray"]

for col, (title, cmap) in enumerate(zip(COL_TITLES, COL_CMAPS)):
    axes[0][col].set_title(title, color="#7B9CFF", fontsize=9,
                           fontfamily="monospace", fontweight="bold", pad=6)

for row, r in enumerate(results):
    imgs  = [r["mask"], r["aerial"], r["resist"]]
    stats = [
        f"fill {r['mask'].mean()*100:.1f}%",
        f"μ={r['aerial'].mean():.3f}  σ={r['aerial'].std():.3f}",
        f"μ={r['resist'].mean():.3f}  σ={r['resist'].std():.3f}",
    ]
    for col, (img, cmap, stat) in enumerate(zip(imgs, COL_CMAPS, stats)):
        ax = axes[row][col]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#243047")
        ax.text(0.02, 0.97, stat, transform=ax.transAxes,
                va="top", ha="left", color="#A5D6A7",
                fontsize=6.5, fontfamily="monospace")
    # Row label
    axes[row][0].set_ylabel(r["label"], color="#C8D3E8",
                            fontsize=8, fontfamily="monospace", labelpad=6)

fig.tight_layout(rect=[0, 0, 1, 1])
preview_path = os.path.join(OUT_DIR, "litho_preview.png")
fig.savefig(preview_path, dpi=130, bbox_inches="tight", facecolor="#0C1018")
plt.close(fig)
print(f"Preview figure saved → {preview_path}")
print("\nStep 3 complete.")
