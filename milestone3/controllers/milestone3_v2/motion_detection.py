import numpy as np

RESIZE_H = 150
RESIZE_W = 150
N_COLS = 32          # column blocks — finer blocks (8 cam-px each) = less wall bleed
FLOW_THRESHOLD = 1.5 # per-pixel flow magnitude threshold (px/frame)
_LK_WIN = 5          # LK window size for per-pixel structure tensor
_DET_MIN = 50.0      # minimum det — lower catches weaker ball-edge pixels

_GAUSSIAN = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
_SOBEL_V  = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_H  = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]], dtype=np.float64)


def get_blocked_columns(frame1_bgra, frame2_bgra, cam_w):
    """
    Returns a set of camera pixel columns (0..cam_w-1) occupied by moving objects.
    Inputs: numpy arrays (H, W, 4) uint8, BGRA channel order (Webots format).
    Returns empty set on first call (frame1_bgra is None).
    """
    if frame1_bgra is None or frame2_bgra is None:
        return set()

    f1 = _resize(frame1_bgra).astype(np.float64)
    f2 = _resize(frame2_bgra).astype(np.float64)

    # BGRA: channel 2 = R, 1 = G, 0 = B
    f1_gray = 0.299 * f1[:, :, 2] + 0.587 * f1[:, :, 1] + 0.114 * f1[:, :, 0]
    f2_gray = 0.299 * f2[:, :, 2] + 0.587 * f2[:, :, 1] + 0.114 * f2[:, :, 0]

    f1_blur = _convolve(_GAUSSIAN, f1_gray)
    f2_blur = _convolve(_GAUSSIAN, f2_gray)

    # _lk_column_flows returns flows with interior-fill already applied
    flows = _lk_column_flows(f1_blur, f2_blur, N_COLS)

    blocked = set()
    col_w = RESIZE_W // N_COLS
    for k, mag in enumerate(flows):
        if mag > FLOW_THRESHOLD:
            for res_col in range(k * col_w, (k + 1) * col_w):
                blocked.add(int(res_col * cam_w / RESIZE_W))

    return blocked


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resize(img):
    """Nearest-neighbour downsample to RESIZE_H × RESIZE_W."""
    h, w = img.shape[:2]
    rows = (np.arange(RESIZE_H) * h // RESIZE_H)
    cols = (np.arange(RESIZE_W) * w // RESIZE_W)
    return img[rows[:, None], cols[None, :], :]


def _convolve(kernel, array):
    """Vectorised 2-D convolution."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(array, ((ph, ph), (pw, pw)), mode='constant')
    out = np.zeros_like(array, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += padded[i:i+array.shape[0], j:j+array.shape[1]] * kernel[i, j]
    return out


def _lk_column_flows(gray1, gray2, n_cols):
    """
    Per-pixel Lucas-Kanade optical flow using a _LK_WIN x _LK_WIN structure tensor.
    Solves per pixel: [Sxx Sxy][u] = [-Sxt]
                      [Sxy Syy][v]   [-Syt]

    After computing per-block max flow, fills the contiguous span between the
    outermost hot blocks (±1 block margin) so the ball interior — which has no
    spatial gradient and therefore no valid LK estimate — is covered without
    expanding into unrelated wall regions.
    """
    Ix = _convolve(_SOBEL_V, gray2)   # dI/dx
    Iy = _convolve(_SOBEL_H, gray2)   # dI/dy
    It = gray2 - gray1                 # dI/dt

    # Accumulate structure tensor over _LK_WIN x _LK_WIN window per pixel
    ones = np.ones((_LK_WIN, _LK_WIN), dtype=np.float64)
    Sxx = _convolve(ones, Ix * Ix)
    Sxy = _convolve(ones, Ix * Iy)
    Syy = _convolve(ones, Iy * Iy)
    Sxt = _convolve(ones, Ix * It)
    Syt = _convolve(ones, Iy * It)

    det = Sxx * Syy - Sxy ** 2
    valid = det > _DET_MIN   # reject featureless / ill-conditioned pixels

    safe_det = np.where(valid, det, 1.0)
    u = np.where(valid, (-Syy * Sxt + Sxy * Syt) / safe_det, 0.0)
    v = np.where(valid, ( Sxy * Sxt - Sxx * Syt) / safe_det, 0.0)
    mag = np.sqrt(u ** 2 + v ** 2)

    col_w = RESIZE_W // n_cols
    flows = np.zeros(n_cols, dtype=np.float64)
    for k in range(n_cols):
        c0, c1 = k * col_w, (k + 1) * col_w
        flows[k] = float(np.max(mag[:, c0:c1]))

    # Interior fill: find the outermost hot blocks and fill everything between
    # them (plus ±1 block margin).  LK only fires at the ball's corner pixels;
    # the uniform interior has det≈0 and contributes nothing.  Filling in block
    # space (each block ≈ CAM_W/N_COLS camera pixels wide) keeps the expansion
    # tightly scoped to the ball, unlike a raw pixel-level dilation.
    hot = [k for k in range(n_cols) if flows[k] > FLOW_THRESHOLD]
    if hot:
        lo = max(0, hot[0] - 1)
        hi = min(n_cols - 1, hot[-1] + 1)
        fill_val = FLOW_THRESHOLD + 0.1
        for k in range(lo, hi + 1):
            if flows[k] <= FLOW_THRESHOLD:
                flows[k] = fill_val

    return flows
