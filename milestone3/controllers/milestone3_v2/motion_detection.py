"""
motion_detection.py
-------------------
Detects moving objects in consecutive camera frames and returns the set of
camera pixel-columns that contain motion.

Pipeline (called once per time step):
  1. Resize & grayscale both frames
  2. Lucas-Kanade optical flow  →  per-pixel (u, v) + validity mask
  3. Ego-motion compensation    →  subtract expected flow from robot movement
     - For valid (textured) pixels: residual = LK flow − ego flow
     - For invalid (smooth) pixels: fallback to frame-difference signal
  4. Threshold residual magnitude → hot pixels
  5. Count hot pixels per column → blocked columns

Public API
----------
  get_blocked_columns(frame1_bgra, frame2_bgra, cam_w,
                      pose1, pose2, cam_hfov,
                      lidar_ranges, lidar_fov)  →  set of int
"""

import numpy as np

# ── Tunable parameters ────────────────────────────────────────────────────────
RESIZE_H       = 120    # height to downscale frames before processing
RESIZE_W       = 160    # width  to downscale frames before processing

MIN_HOT_PIXELS = 8      # columns with fewer hot pixels are ignored
K_SIGMA        = 1.0    # threshold = mean + K_SIGMA * std of residual magnitudes
MIN_ABS_FLOW   = 1.5    # hard minimum threshold (pixels/frame)

_LK_WIN        = 7      # Lucas-Kanade summation window (pixels, must be odd)
_DET_MIN       = 15.0   # minimum structure-tensor determinant to trust LK result
_DIFF_SCALE    = 35.0   # frame-diff fallback: ~35 intensity units ≈ 1 px motion
                        # (shadows ≈ 20-30 units → ignored; balls ≈ 80+ → detected)
# ─────────────────────────────────────────────────────────────────────────────

# Pre-built convolution kernels
_GAUSSIAN = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float64) / 16.0
_SOBEL_X  = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
_SOBEL_Y  = np.array([[ 1,2,1],[ 0,0,0],[-1,-2,-1]], dtype=np.float64)


# ── Public API ────────────────────────────────────────────────────────────────

def get_blocked_columns(frame1_bgra, frame2_bgra, cam_w,
                        pose1=None, pose2=None,
                        cam_hfov=1.085,
                        lidar_ranges=None,
                        lidar_fov=None):
    """
    Return the set of camera pixel-columns (in original resolution) that
    contain moving objects between frame1 and frame2.

    Parameters
    ----------
    frame1_bgra, frame2_bgra : np.ndarray  (H, W, 4) uint8  BGRA
    cam_w                    : int   original camera width in pixels
    pose1, pose2             : (x, y, yaw)  robot pose at each frame
    cam_hfov                 : float  camera horizontal FOV in radians
    lidar_ranges             : list[float]  LiDAR range image
    lidar_fov                : float  LiDAR FOV in radians
    """
    if frame1_bgra is None or frame2_bgra is None:
        return set()

    # --- Step 1: resize and convert to grayscale ---
    f1 = _resize(frame1_bgra).astype(np.float64)
    f2 = _resize(frame2_bgra).astype(np.float64)

    f1g = 0.299*f1[:,:,2] + 0.587*f1[:,:,1] + 0.114*f1[:,:,0]
    f2g = 0.299*f2[:,:,2] + 0.587*f2[:,:,1] + 0.114*f2[:,:,0]

    f1g = _convolve(_GAUSSIAN, f1g)
    f2g = _convolve(_GAUSSIAN, f2g)

    # --- Step 2: Lucas-Kanade optical flow ---
    u, v, valid = _lk_flow(f1g, f2g)

    # --- Step 3: ego-motion compensation ---
    if pose1 is not None and pose2 is not None:
        depth_map = _build_depth_map(lidar_ranges, lidar_fov, cam_hfov)
        u, v = _compensate_ego(u, v, valid, pose1, pose2, cam_hfov, depth_map)
    else:
        u = np.where(valid, u, 0.0)
        v = np.where(valid, v, 0.0)

    # --- Step 4: residual magnitude ---
    mag = np.sqrt(u**2 + v**2)

    # Fallback for low-texture pixels (invalid LK): use frame-difference signal.
    # Shadows change luminance but not structure, so large diffs = real object motion.
    frame_diff = np.abs(f2g - f1g)
    mag = np.where(valid, mag, frame_diff / _DIFF_SCALE)

    # --- Step 5: threshold and count hot pixels per column ---
    threshold  = max(float(np.mean(mag)) + K_SIGMA * float(np.std(mag)), MIN_ABS_FLOW)
    hot_counts = np.sum(mag > threshold, axis=0)

    blocked = set()
    for res_col in range(RESIZE_W):
        if hot_counts[res_col] >= MIN_HOT_PIXELS:
            blocked.add(int(res_col * cam_w / RESIZE_W))

    return blocked


# ── Private helpers (called by get_blocked_columns) ───────────────────────────

def _lk_flow(gray1, gray2):
    """
    Lucas-Kanade optical flow on two grayscale frames.
    Returns (u, v, valid) where valid[r,c] is True when the local structure
    tensor is well-conditioned (determinant > _DET_MIN).
    Invalid pixels have u=v=0.
    """
    Ix = _convolve(_SOBEL_X, gray2)
    Iy = _convolve(_SOBEL_Y, gray2)
    It = gray2 - gray1

    ones = np.ones((_LK_WIN, _LK_WIN), dtype=np.float64)
    Sxx  = _convolve(ones, Ix * Ix)
    Sxy  = _convolve(ones, Ix * Iy)
    Syy  = _convolve(ones, Iy * Iy)
    Sxt  = _convolve(ones, Ix * It)
    Syt  = _convolve(ones, Iy * It)

    det      = Sxx * Syy - Sxy**2
    valid    = det > _DET_MIN
    safe_det = np.where(valid, det, 1.0)

    u = np.where(valid, (-Syy * Sxt + Sxy * Syt) / safe_det, 0.0)
    v = np.where(valid, ( Sxy * Sxt - Sxx * Syt) / safe_det, 0.0)
    return u, v, valid


def _compensate_ego(u, v, valid, pose1, pose2, cam_hfov, depth_map):
    """
    Subtract the optical flow caused by the robot's own movement so that
    only genuinely moving objects remain in the residual.

    Sources of ego-motion modelled:
      - Yaw rotation  → uniform horizontal shift
      - Forward translation → depth-scaled radial flow
      - Lateral translation → depth-scaled horizontal shift

    Invalid LK pixels are zeroed out (no reliable flow estimate to compare).
    """
    x1, y1, yaw1 = pose1
    x2, y2, yaw2 = pose2

    d_yaw = yaw2 - yaw1
    if d_yaw >  np.pi: d_yaw -= 2*np.pi
    if d_yaw < -np.pi: d_yaw += 2*np.pi

    cos_y     = np.cos(yaw1)
    sin_y     = np.sin(yaw1)
    dx_w      = x2 - x1
    dy_w      = y2 - y1
    d_forward = dx_w * cos_y  + dy_w * sin_y
    d_lateral = -dx_w * sin_y + dy_w * cos_y

    fx = (RESIZE_W / 2.0) / np.tan(cam_hfov / 2.0)   # focal length in resized pixels

    col_centres = np.arange(RESIZE_W, dtype=np.float64) - RESIZE_W / 2.0
    row_centres = np.arange(RESIZE_H, dtype=np.float64) - RESIZE_H / 2.0
    X = np.tile(col_centres,               (RESIZE_H, 1))
    Y = np.tile(row_centres.reshape(-1,1), (1, RESIZE_W))
    Z = np.maximum(np.tile(depth_map, (RESIZE_H, 1)), 0.05)   # clamp to avoid /0

    ego_u  = d_yaw * fx * np.ones((RESIZE_H, RESIZE_W))   # rotation
    ego_v  = np.zeros((RESIZE_H, RESIZE_W))
    ego_u += d_forward * X / Z                             # forward translation
    ego_v += d_forward * Y / Z
    ego_u += fx * d_lateral / Z                            # lateral translation

    res_u = np.where(valid, u - ego_u, 0.0)
    res_v = np.where(valid, v - ego_v, 0.0)
    return res_u, res_v


def _build_depth_map(lidar_ranges, lidar_fov, cam_hfov):
    """
    Map LiDAR ranges onto RESIZE_W columns that correspond to the camera FOV.
    Returns a 1-D array of depth values (metres) per resized image column.
    Falls back to 1.0 m for columns with no valid LiDAR ray.
    """
    DEFAULT_DEPTH = 1.0
    depth = np.full(RESIZE_W, DEFAULT_DEPTH, dtype=np.float64)

    if lidar_ranges is None or lidar_fov is None:
        return depth

    ranges     = np.array(lidar_ranges, dtype=np.float64)
    n          = len(ranges)
    ray_angles = -lidar_fov/2 + np.arange(n) / max(n-1, 1) * lidar_fov

    in_fov = np.abs(ray_angles) <= cam_hfov / 2
    valid  = in_fov & (ranges > 0) & np.isfinite(ranges)
    if not np.any(valid):
        return depth

    valid_angles = ray_angles[valid]
    valid_ranges = ranges[valid]

    for col in range(RESIZE_W):
        col_angle  = (0.5 - col / RESIZE_W) * cam_hfov
        nearest    = int(np.argmin(np.abs(valid_angles - col_angle)))
        depth[col] = float(valid_ranges[nearest])

    return depth


def _resize(img):
    """Nearest-neighbour downsample to (RESIZE_H, RESIZE_W)."""
    h, w = img.shape[:2]
    rows = np.arange(RESIZE_H) * h // RESIZE_H
    cols = np.arange(RESIZE_W) * w // RESIZE_W
    return img[rows[:, None], cols[None, :], :]


def _convolve(kernel, array):
    """2-D convolution (no scipy dependency). Pads with zeros at borders."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(array, ((ph, ph), (pw, pw)), mode='constant')
    out    = np.zeros_like(array, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += padded[i:i+array.shape[0], j:j+array.shape[1]] * kernel[i, j]
    return out
