import cv2
import numpy as np


def augment(
    img: np.ndarray,
    gaussian_sigma: float = 0.03,
    poisson_scale: int = 4000,
    sem_texture_sigma: float = 0.015,
    blur_sigma: float = 1.0,
    max_grad: float = 0.12,
    max_rotate_deg: float = 1.5,
    scale_range: tuple = (0.85, 1.0),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Apply SEM-realistic augmentations to a float32 grayscale image [0,1].
    All parameters can be randomised by the caller for dataset diversity.
    """
    rng = rng or np.random.default_rng()
    H, W = img.shape
    out = img.copy()

    # 1. Brightness / contrast variation
    alpha = rng.uniform(0.85, 1.15)   # contrast
    beta  = rng.uniform(-0.08, 0.08)  # brightness offset
    out   = np.clip(alpha * out + beta, 0, 1)

    # 2. Gaussian noise (electronic noise floor)
    if gaussian_sigma > 0:
        noise = rng.normal(0, gaussian_sigma, (H, W)).astype(np.float32)
        out   = np.clip(out + noise, 0, 1)

    # 3. Poisson (shot) noise — dominant in real SEM images
    if poisson_scale > 0:
        out = np.random.poisson(out * poisson_scale).astype(np.float32) / poisson_scale
        out = np.clip(out, 0, 1)

    # 4. SEM grain texture (electron charging / secondary emission noise)
    if sem_texture_sigma > 0:
        grain = rng.normal(0, sem_texture_sigma, (H, W)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (3, 3), 0.8)
        out   = np.clip(out + grain, 0, 1)

    # 5. Optical / scan blur
    if blur_sigma > 0:
        s = rng.uniform(0, blur_sigma)
        if s > 0.1:
            out = cv2.GaussianBlur(out, (0, 0), s)

    # 6. Illumination gradient (e-beam dose non-uniformity)
    if max_grad > 0:
        gx = rng.uniform(-max_grad, max_grad)
        gy = rng.uniform(-max_grad, max_grad)
        xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        out = np.clip(out * (1 + gx*xx + gy*yy), 0, 1)

    # 7. Small rotation (sample tilt / stage vibration)
    angle = rng.uniform(-max_rotate_deg, max_rotate_deg)
    M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
    out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

    # 8. Random scale crop (zoom variation between acquisitions)
    scale  = rng.uniform(*scale_range)
    nh, nw = int(H*scale), int(W*scale)
    y0 = rng.integers(0, H-nh+1)
    x0 = rng.integers(0, W-nw+1)
    out = cv2.resize(out[y0:y0+nh, x0:x0+nw], (W, H), interpolation=cv2.INTER_AREA)

    return out.astype(np.float32)


def inject_defect(
    img: np.ndarray,
    kind: str = "bridge",
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Insert one synthetic defect into the image.
    kind: 'bridge' | 'open' | 'particle' | 'scratch'
    """
    rng = rng or np.random.default_rng()
    H, W = img.shape
    out = img.copy()

    if kind == "bridge":
        # Extra material connecting two features (bright blob)
        x, y = rng.integers(W//4, 3*W//4), rng.integers(H//4, 3*H//4)
        r    = int(rng.integers(3, 9))
        cv2.circle(out, (x, y), r, 1.0, -1)
        out = cv2.GaussianBlur(out, (5, 5), 1.5)

    elif kind == "open":
        # Missing material — dark gap through a bright line
        x, y = rng.integers(W//4, 3*W//4), rng.integers(H//4, 3*H//4)
        ln   = int(rng.integers(10, 30))
        th   = int(rng.integers(2, 5))
        ang  = rng.uniform(0, np.pi)
        dx, dy = int(ln/2*np.cos(ang)), int(ln/2*np.sin(ang))
        cv2.line(out, (x-dx, y-dy), (x+dx, y+dy), 0.0, th)

    elif kind == "particle":
        # Surface particle — bright ellipse with soft edge
        x, y = rng.integers(0, W), rng.integers(0, H)
        ra   = int(rng.integers(2, 8))
        rb   = int(rng.integers(2, 8))
        val  = float(rng.uniform(0.7, 1.0))
        cv2.ellipse(out, (x,y), (ra,rb), rng.uniform(0,180), 0, 360, val, -1)

    elif kind == "scratch":
        # Mechanical scratch — thin dark diagonal line
        x1, y1 = rng.integers(0, W), rng.integers(0, H)
        ln  = int(rng.integers(40, 120))
        ang = rng.uniform(0, np.pi)
        x2  = int(np.clip(x1 + ln*np.cos(ang), 0, W-1))
        y2  = int(np.clip(y1 + ln*np.sin(ang), 0, H-1))
        val = float(rng.uniform(0.0, 0.25))
        cv2.line(out, (x1,y1), (x2,y2), val, 1)

    return np.clip(out, 0, 1).astype(np.float32)
