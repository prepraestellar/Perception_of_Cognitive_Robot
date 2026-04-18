import numpy as np

RESIZE_H = 150
RESIZE_W = 150
DIFF_THRESHOLD = 120
EDGE_THRESHOLD = 75

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
    f1_gray   = 0.299 * f1[:,:,2] + 0.587 * f1[:,:,1] + 0.114 * f1[:,:,0]
    f2_gray   = 0.299 * f2[:,:,2] + 0.587 * f2[:,:,1] + 0.114 * f2[:,:,0]

    f1_blur   = _convolve(_GAUSSIAN, f1_gray)
    f2_blur   = _convolve(_GAUSSIAN, f2_gray)

    # Frame-difference mask: pixels that changed significantly
    diff_mask = (f2_blur - f1_blur) > DIFF_THRESHOLD

    # Edge detection on frame2 across all channels
    edge_sum = np.zeros((RESIZE_H, RESIZE_W), dtype=np.uint8)
    for ch in (0, 1, 2):  # B, G, R
        ch_blur = _convolve(_GAUSSIAN, f2[:,:,ch])
        edge_sum += (_edges(ch_blur) > EDGE_THRESHOLD).astype(np.uint8)
    edge_sum += (_edges(f2_blur) > EDGE_THRESHOLD).astype(np.uint8)

    # Group interior regions (zero-edge pixels) into blobs
    blobs = _blobize(edge_sum)

    # A blob is "moving" if any of its pixels overlap the diff mask
    diff_set = set(zip(*np.where(diff_mask))) if diff_mask.any() else set()
    blocked  = set()
    for blob in blobs:
        if blob & diff_set:
            for (_, c) in blob:
                blocked.add(int(c * cam_w / RESIZE_W))

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
    """Vectorised 2-D convolution (much faster than pixel-by-pixel loops)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(array, ((ph, ph), (pw, pw)), mode='constant')
    out = np.zeros_like(array, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += padded[i:i+array.shape[0], j:j+array.shape[1]] * kernel[i, j]
    return out


def _edges(blurred):
    """Sobel edge magnitude."""
    return np.sqrt(_convolve(_SOBEL_V, blurred)**2 + _convolve(_SOBEL_H, blurred)**2)


def _blobize(edge_map):
    """
    Flood-fill connected regions of NON-edge (zero) pixels.
    Each blob is a set of (row, col) representing one interior region.
    """
    rows, cols = edge_map.shape
    visited = np.zeros((rows, cols), dtype=bool)
    blobs   = []

    for i in range(rows):
        for j in range(cols):
            if edge_map[i, j] != 0 or visited[i, j]:
                continue
            blob  = set()
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                if x < 0 or x >= rows or y < 0 or y >= cols:
                    continue
                if visited[x, y] or edge_map[x, y] != 0:
                    continue
                visited[x, y] = True
                blob.add((x, y))
                stack.extend([(x+1,y),(x-1,y),(x,y+1),(x,y-1)])
            if blob:
                blobs.append(blob)

    return blobs
