"""
Step 5 — Batch Dataset Generation
Loops the full pipeline (layout → mask → litho → augment) to produce
a large dataset split into train / val / test with a metadata manifest.
"""
import sys, os
_desktop = os.path.dirname(os.path.abspath(__file__))
# Keep Desktop on path for layout_generators and augment, but strip it
# before any import that would collide with a stdlib/package name.
_clean_path = [p for p in sys.path if os.path.abspath(p) != _desktop]

# ── Imports that need the clean path ─────────────────────────────────────
sys.path = _clean_path
import gdstk
sys.path.insert(0, _desktop)   # restore Desktop for our own modules

import json, shutil, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from layout_generators import finfet_layout, dram_layout, sram_cell_layout, nand_flash_layout
from augment import augment, inject_defect

# ── Config ────────────────────────────────────────────────────────────────
DESKTOP   = "/home/nihal/Desktop"
OUT_ROOT  = os.path.join(DESKTOP, "dataset")
TMP_DIR   = "/tmp/wafer_gds_tmp"
N_TOTAL   = 500          # total images to generate
RESOLUTION = 512         # pixels per side
DEFECT_RATE = 0.35       # fraction of images with a defect
SEED      = 42
SPLITS    = {"train": 0.80, "val": 0.10, "test": 0.10}
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

LAYOUT_GEN = {
    "finfet": finfet_layout,
    "dram":   dram_layout,
    "sram":   sram_cell_layout,
    "nand":   nand_flash_layout,
}
# Physical pixel size per layout type (nm/px at 512 resolution) — from step 2
LAYOUT_PX_NM = {
    "finfet": 0.50,
    "dram":   1.25,
    "sram":   3.00,
    "nand":   1.20,
}
DEFECT_KINDS = ["bridge", "open", "particle", "scratch"]


# ── Inline rasterizer (avoids gdstk.py name collision) ───────────────────
def gds_to_mask(gds_path, layer=(1, 0), resolution=512, margin_nm=20.0):
    lib  = gdstk.read_gds(gds_path)
    cell = lib.top_level()[0]
    polys = [p.points * 1000
             for p in cell.get_polygons(layer=layer[0], datatype=layer[1])]
    if not polys:
        raise ValueError(f"No polygons on layer {layer}")
    all_pts = np.vstack(polys)
    xmin, ymin = all_pts.min(axis=0) - margin_nm
    xmax, ymax = all_pts.max(axis=0) + margin_nm
    scale = resolution / max(xmax - xmin, ymax - ymin)
    img   = Image.new("L", (resolution, resolution), 0)
    draw  = ImageDraw.Draw(img)
    for pts in polys:
        px = [((x - xmin) * scale, (ymax - y) * scale) for x, y in pts]
        draw.polygon(px, fill=255)
    return np.array(img, dtype=np.float32) / 255.0


# ── Inline litho simulator ────────────────────────────────────────────────
class LithoSim:
    def __init__(self, wavelength_nm=193.0, na=1.35, sigma=0.30,
                 pixel_size_nm=1.0, defocus_nm=0.0,
                 resist_threshold=0.225, resist_blur_nm=5.0, device="cpu"):
        self.wl, self.na, self.sigma = wavelength_nm, na, sigma
        self.px, self.defocus        = pixel_size_nm, defocus_nm
        self.thresh                  = resist_threshold
        self.rbsig                   = resist_blur_nm / pixel_size_nm
        self.dev                     = torch.device(device)

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
        r = torch.sqrt(FX**2 + FY**2)
        S = (r <= self.sigma).float()
        return S / (S.sum() + 1e-12)

    def simulate(self, mask_np):
        mask = torch.from_numpy(mask_np).to(self.dev).to(torch.complex64)
        N    = mask.shape[-1]
        M    = torch.fft.fft2(mask)
        P    = self._pupil(N)
        S    = self._source(N)
        I    = torch.fft.ifft2(P * M).abs()**2
        I    = I.real.float()
        I    = torch.fft.ifft2(torch.fft.fft2(I) * torch.fft.fft2(S)).real.float()
        I    = torch.clamp(I, 0)
        I    = I / (I.max() + 1e-8)
        k    = max(int(self.rbsig * 4) | 1, 3)
        R    = gaussian_blur(I.unsqueeze(0), [k, k], self.rbsig).squeeze(0)
        R    = torch.sigmoid(20.0 * (R - self.thresh))
        return R.cpu().numpy()


# ── Dataset generator ─────────────────────────────────────────────────────
def generate_dataset():
    rng  = np.random.default_rng(SEED)
    root = Path(OUT_ROOT)
    tmp  = Path(TMP_DIR)
    tmp.mkdir(parents=True, exist_ok=True)

    # Build split id ranges
    split_ranges = {}
    cur = 0
    for sp, frac in SPLITS.items():
        n = int(N_TOTAL * frac)
        split_ranges[sp] = range(cur, cur + n)
        for sub in ("images", "masks"):
            (root / sp / sub).mkdir(parents=True, exist_ok=True)
        cur += n

    layout_names = list(LAYOUT_GEN.keys())
    records      = []
    errors       = 0
    t0           = time.time()

    print(f"\nGenerating {N_TOTAL} images  |  device={DEVICE}  |  output={OUT_ROOT}\n")

    for idx in tqdm(range(N_TOTAL), desc="Samples", unit="img"):
        sp = next(s for s, r in split_ranges.items() if idx in r)

        # ── Randomise parameters ──────────────────────────────────────────
        ltype      = rng.choice(layout_names)
        na         = float(rng.uniform(1.20, 1.35))
        sigma      = float(rng.uniform(0.15, 0.50))
        defocus    = float(rng.uniform(-40, 40))
        px_nm      = LAYOUT_PX_NM[ltype] * float(rng.uniform(0.9, 1.1))
        has_defect = rng.random() < DEFECT_RATE
        defect_k   = str(rng.choice(DEFECT_KINDS)) if has_defect else None

        # ── Layout → GDS ──────────────────────────────────────────────────
        gds_path = str(tmp / f"layout_{idx}.gds")
        try:
            LAYOUT_GEN[ltype](filename=gds_path)
            mask_np = gds_to_mask(gds_path, resolution=RESOLUTION)
        except Exception as e:
            tqdm.write(f"  [SKIP {idx}] layout/mask error: {e}")
            errors += 1
            continue
        finally:
            if os.path.exists(gds_path):
                os.remove(gds_path)

        # ── Lithography simulation ─────────────────────────────────────────
        try:
            sim    = LithoSim(na=na, sigma=sigma, defocus_nm=defocus,
                               pixel_size_nm=px_nm, device=DEVICE)
            resist = sim.simulate(mask_np)
        except Exception as e:
            tqdm.write(f"  [SKIP {idx}] litho error: {e}")
            errors += 1
            continue

        # ── Augmentation ──────────────────────────────────────────────────
        aug = augment(
            resist,
            gaussian_sigma    = float(rng.uniform(0.02, 0.05)),
            poisson_scale     = int(rng.integers(3000, 6000)),
            sem_texture_sigma = float(rng.uniform(0.01, 0.025)),
            blur_sigma        = float(rng.uniform(0.5, 1.5)),
            max_grad          = float(rng.uniform(0.05, 0.15)),
            max_rotate_deg    = float(rng.uniform(0.5, 2.0)),
            scale_range       = (float(rng.uniform(0.82, 0.92)), 1.0),
            rng               = rng,
        )
        if has_defect:
            aug = inject_defect(aug, kind=defect_k, rng=rng)

        # ── Save ──────────────────────────────────────────────────────────
        fname = f"{idx:06d}"
        cv2.imwrite(str(root / sp / "images" / f"{fname}.png"),
                    (aug     * 255).astype(np.uint8))
        cv2.imwrite(str(root / sp / "masks"  / f"{fname}.png"),
                    (mask_np * 255).astype(np.uint8))

        records.append({
            "id":         fname,
            "split":      sp,
            "layout":     ltype,
            "na":         round(na, 3),
            "sigma":      round(sigma, 3),
            "defocus_nm": round(defocus, 1),
            "px_nm":      round(px_nm, 3),
            "has_defect": bool(has_defect),
            "defect":     defect_k,
        })

    # ── Manifest ──────────────────────────────────────────────────────────
    manifest = root / "metadata.json"
    manifest.write_text(json.dumps(records, indent=2))

    elapsed = time.time() - t0
    ok      = len(records)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  Total generated : {ok} / {N_TOTAL}  ({errors} errors skipped)")
    print(f"  Time elapsed    : {elapsed:.1f}s  ({elapsed/max(ok,1):.2f}s/img)")
    print(f"  Output root     : {OUT_ROOT}")
    print(f"{'─'*52}")
    for sp, rng_ in split_ranges.items():
        n = sum(1 for r in records if r["split"] == sp)
        print(f"    {sp:<6}: {n} images")
    nd = sum(1 for r in records if r["has_defect"])
    print(f"\n  Defect images   : {nd} ({nd/max(ok,1)*100:.1f}%)")
    print(f"  Clean images    : {ok-nd} ({(ok-nd)/max(ok,1)*100:.1f}%)")

    # Layout distribution
    from collections import Counter
    lc = Counter(r["layout"] for r in records)
    print(f"\n  Layout mix:")
    for lt, cnt in sorted(lc.items()):
        print(f"    {lt:<8}: {cnt}")

    print(f"\n  Manifest → {manifest}")
    print(f"\nStep 5 complete.\n")


if __name__ == "__main__":
    generate_dataset()
