import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import gaussian_blur


class LithoSim:
    """
    Partially coherent optical lithography simulator (Hopkins model).
    All spatial quantities in nm.
    """
    def __init__(
        self,
        wavelength_nm: float = 193.0,
        na: float = 1.35,
        sigma: float = 0.30,     # partial coherence (0=coherent, 1=full)
        pixel_size_nm: float = 1.0,
        defocus_nm: float = 0.0,
        resist_threshold: float = 0.225,
        resist_blur_nm: float = 5.0,
        device: str = "cpu",
    ):
        self.wl     = wavelength_nm
        self.na     = na
        self.sigma  = sigma
        self.px     = pixel_size_nm
        self.defocus = defocus_nm
        self.thresh = resist_threshold
        self.rbsig  = resist_blur_nm / pixel_size_nm
        self.device = torch.device(device)

    def _freq_grid(self, N: int) -> torch.Tensor:
        # Normalised spatial frequencies in pupil units (NA/λ scale)
        f = torch.fft.fftfreq(N, d=self.px * self.na / self.wl)
        FX, FY = torch.meshgrid(f, f, indexing="ij")
        return FX.to(self.device), FY.to(self.device)

    def _pupil(self, N: int) -> torch.Tensor:
        FX, FY = self._freq_grid(N)
        r = torch.sqrt(FX**2 + FY**2)
        P = (r <= 1.0).to(torch.complex64)
        if self.defocus != 0:
            # Defocus phase: W(r) = π·z·NA²·r²/λ
            phase = (torch.pi * self.defocus * self.na**2 / self.wl) * r**2
            P = P * torch.exp(1j * phase)
        return P

    def _source(self, N: int) -> torch.Tensor:
        # Circular illumination source (disk of radius sigma)
        FX, FY = self._freq_grid(N)
        r = torch.sqrt(FX**2 + FY**2)
        S = (r <= self.sigma).float()
        S = S / (S.sum() + 1e-12)
        return S

    def aerial(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute aerial image (intensity in wafer plane).
        mask: (H, W) float tensor, values in [0,1]
        """
        N = mask.shape[-1]
        mask = mask.to(self.device).to(torch.complex64)

        M = torch.fft.fft2(mask)
        P = self._pupil(N)
        S = self._source(N)

        # Coherent aerial: amplitude = IFFT(P · M)
        amplitude = torch.fft.ifft2(P * M)
        I_coh = amplitude.abs() ** 2

        # Partial coherence: convolve coherent intensity with source
        I_coh = I_coh.real.float()
        A  = torch.fft.fft2(I_coh)
        Sf = torch.fft.fft2(S)
        I  = torch.fft.ifft2(A * Sf).real.float()
        I  = torch.clamp(I, 0)
        I  = I / (I.max() + 1e-8)
        return I

    def resist(self, aerial: torch.Tensor) -> torch.Tensor:
        """Gaussian blur + sigmoid threshold resist model."""
        k = max(int(self.rbsig * 4) | 1, 3)
        blurred = gaussian_blur(
            aerial.unsqueeze(0), kernel_size=[k, k], sigma=self.rbsig
        ).squeeze(0)
        steepness = 20.0   # higher = sharper resist edge
        return torch.sigmoid(steepness * (blurred - self.thresh))

    def simulate(self, mask: torch.Tensor) -> dict:
        ai = self.aerial(mask)
        ri = self.resist(ai)
        return {"mask": mask, "aerial": ai, "resist": ri}


# ── Quick usage ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from gds_to_mask import gds_to_mask
    mask_np = gds_to_mask("finfet.gds")
    mask_t  = torch.from_numpy(mask_np)

    sim     = LithoSim(na=1.35, sigma=0.3, defocus_nm=0)
    results = sim.simulate(mask_t)

    print("Aerial range:", results["aerial"].min().item(), results["aerial"].max().item())
    print("Resist range:", results["resist"].min().item(), results["resist"].max().item())
