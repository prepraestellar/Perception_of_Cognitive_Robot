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

# import numpy as np

# # ── Tunable parameters ────────────────────────────────────────────────────────
# RESIZE_H       = 120    # height to downscale frames before processing
# RESIZE_W       = 160    # width  to downscale frames before processing

# MIN_HOT_PIXELS = 8      # columns with fewer hot pixels are ignored
# K_SIGMA        = 1.0    # threshold = mean + K_SIGMA * std of residual magnitudes
# MIN_ABS_FLOW   = 1.5    # hard minimum threshold (pixels/frame)

# _LK_WIN        = 7      # Lucas-Kanade summation window (pixels, must be odd)
# _DET_MIN       = 15.0   # minimum structure-tensor determinant to trust LK result
# _DIFF_SCALE    = 35.0   # frame-diff fallback: ~35 intensity units ≈ 1 px motion
#                         # (shadows ≈ 20-30 units → ignored; balls ≈ 80+ → detected)

# _MASK_RADIUS          = 1    # morphology radius to connect fragmented masks
# _MIN_COMPONENT_PIXELS = 28   # suppress tiny motion blobs (noise, wall shimmer)
# _MIN_COMPONENT_WIDTH  = 3
# _MIN_COMPONENT_HEIGHT = 3
# _MAX_COLUMN_GAP       = 2    # bridge short breaks between adjacent object columns
# # ─────────────────────────────────────────────────────────────────────────────

# # Pre-built convolution kernels
# _GAUSSIAN = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float64) / 16.0
# _SOBEL_X  = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
# _SOBEL_Y  = np.array([[ 1,2,1],[ 0,0,0],[-1,-2,-1]], dtype=np.float64)

# def _lk_flow(gray1, gray2):
#     """
#     Lucas-Kanade optical flow on two grayscale frames.
#     Returns (u, v, valid) where valid[r,c] is True when the local structure
#     tensor is well-conditioned (determinant > _DET_MIN).
#     Invalid pixels have u=v=0.
#     """
#     Ix = _convolve(_SOBEL_X, gray2)
#     Iy = _convolve(_SOBEL_Y, gray2)
#     It = gray2 - gray1

#     ones = np.ones((_LK_WIN, _LK_WIN), dtype=np.float64)
#     Sxx  = _convolve(ones, Ix * Ix)
#     Sxy  = _convolve(ones, Ix * Iy)
#     Syy  = _convolve(ones, Iy * Iy)
#     Sxt  = _convolve(ones, Ix * It)
#     Syt  = _convolve(ones, Iy * It)

#     det      = Sxx * Syy - Sxy**2
#     valid    = det > _DET_MIN
#     safe_det = np.where(valid, det, 1.0)

#     u = np.where(valid, (-Syy * Sxt + Sxy * Syt) / safe_det, 0.0)
#     v = np.where(valid, ( Sxy * Sxt - Sxx * Syt) / safe_det, 0.0)
#     return u, v, valid


# def _compensate_ego(u, v, valid, pose1, pose2, cam_hfov, depth_map):
#     """
#     Subtract the optical flow caused by the robot's own movement so that
#     only genuinely moving objects remain in the residual.

#     Sources of ego-motion modelled:
#       - Yaw rotation  → uniform horizontal shift
#       - Forward translation → depth-scaled radial flow
#       - Lateral translation → depth-scaled horizontal shift

#     Invalid LK pixels are zeroed out (no reliable flow estimate to compare).
#     """
#     x1, y1, yaw1 = pose1
#     x2, y2, yaw2 = pose2

#     d_yaw = yaw2 - yaw1
#     if d_yaw >  np.pi: d_yaw -= 2*np.pi
#     if d_yaw < -np.pi: d_yaw += 2*np.pi

#     cos_y     = np.cos(yaw1)
#     sin_y     = np.sin(yaw1)
#     dx_w      = x2 - x1
#     dy_w      = y2 - y1
#     d_forward = dx_w * cos_y  + dy_w * sin_y
#     d_lateral = -dx_w * sin_y + dy_w * cos_y

#     fx = (RESIZE_W / 2.0) / np.tan(cam_hfov / 2.0)   # focal length in resized pixels

#     col_centres = np.arange(RESIZE_W, dtype=np.float64) - RESIZE_W / 2.0
#     row_centres = np.arange(RESIZE_H, dtype=np.float64) - RESIZE_H / 2.0
#     X = np.tile(col_centres,               (RESIZE_H, 1))
#     Y = np.tile(row_centres.reshape(-1,1), (1, RESIZE_W))
#     Z = np.maximum(np.tile(depth_map, (RESIZE_H, 1)), 0.05)   # clamp to avoid /0

#     ego_u  = d_yaw * fx * np.ones((RESIZE_H, RESIZE_W))   # rotation
#     ego_v  = np.zeros((RESIZE_H, RESIZE_W))
#     ego_u += d_forward * X / Z                             # forward translation
#     ego_v += d_forward * Y / Z
#     ego_u += fx * d_lateral / Z                            # lateral translation

#     res_u = np.where(valid, u - ego_u, 0.0)
#     res_v = np.where(valid, v - ego_v, 0.0)

#     # Remove global residual drift due to imperfect calibration/model mismatch.
#     if np.any(valid):
#         bias_u = float(np.median(res_u[valid]))
#         bias_v = float(np.median(res_v[valid]))
#         res_u = np.where(valid, res_u - bias_u, 0.0)
#         res_v = np.where(valid, res_v - bias_v, 0.0)

#     return res_u, res_v


# def _build_depth_map(lidar_ranges, lidar_fov, cam_hfov):
#     """
#     Map LiDAR ranges onto RESIZE_W columns that correspond to the camera FOV.
#     Returns a 1-D array of depth values (metres) per resized image column.
#     Falls back to 1.0 m for columns with no valid LiDAR ray.
#     """
#     DEFAULT_DEPTH = 1.0
#     depth = np.full(RESIZE_W, DEFAULT_DEPTH, dtype=np.float64)

#     if lidar_ranges is None or lidar_fov is None:
#         return depth

#     ranges     = np.array(lidar_ranges, dtype=np.float64)
#     n          = len(ranges)
#     ray_angles = -lidar_fov/2 + np.arange(n) / max(n-1, 1) * lidar_fov

#     in_fov = np.abs(ray_angles) <= cam_hfov / 2
#     valid  = in_fov & (ranges > 0) & np.isfinite(ranges)
#     if not np.any(valid):
#         return depth

#     valid_angles = ray_angles[valid]
#     valid_ranges = ranges[valid]

#     for col in range(RESIZE_W):
#         col_angle  = (0.5 - col / RESIZE_W) * cam_hfov
#         nearest    = int(np.argmin(np.abs(valid_angles - col_angle)))
#         depth[col] = float(valid_ranges[nearest])

#     return depth


# def _resize(img):
#     """Nearest-neighbour downsample to (RESIZE_H, RESIZE_W)."""
#     h, w = img.shape[:2]
#     rows = np.arange(RESIZE_H) * h // RESIZE_H
#     cols = np.arange(RESIZE_W) * w // RESIZE_W
#     return img[rows[:, None], cols[None, :], :]


# def get_blocked_columns(frame1_bgra, frame2_bgra, cam_w,
#                         pose1=None, pose2=None,
#                         cam_hfov=1.085,
#                         lidar_ranges=None,
#                         lidar_fov=None):
#     """
#     Return the set of camera pixel-columns (in original resolution) that
#     contain moving objects between frame1 and frame2.

#     Parameters
#     ----------
#     frame1_bgra, frame2_bgra : np.ndarray  (H, W, 4) uint8  BGRA
#     cam_w                    : int   original camera width in pixels
#     pose1, pose2             : (x, y, yaw)  robot pose at each frame
#     cam_hfov                 : float  camera horizontal FOV in radians
#     lidar_ranges             : list[float]  LiDAR range image
#     lidar_fov                : float  LiDAR FOV in radians
#     """
#     if frame1_bgra is None or frame2_bgra is None:
#         return set()

#     # --- Step 1: resize and convert to grayscale ---
#     f1 = _resize(frame1_bgra).astype(np.float64)
#     f2 = _resize(frame2_bgra).astype(np.float64)

#     f1g = 0.299*f1[:,:,2] + 0.587*f1[:,:,1] + 0.114*f1[:,:,0]
#     f2g = 0.299*f2[:,:,2] + 0.587*f2[:,:,1] + 0.114*f2[:,:,0]

#     f1g = _convolve(_GAUSSIAN, f1g)
#     f2g = _convolve(_GAUSSIAN, f2g)

#     # --- Step 2: Lucas-Kanade optical flow ---
#     u, v, valid = _lk_flow(f1g, f2g)

#     # --- Step 3: ego-motion compensation ---
#     if pose1 is not None and pose2 is not None:
#         depth_map = _build_depth_map(lidar_ranges, lidar_fov, cam_hfov)
#         u, v = _compensate_ego(u, v, valid, pose1, pose2, cam_hfov, depth_map)
#     else:
#         u = np.where(valid, u, 0.0)
#         v = np.where(valid, v, 0.0)

#     # --- Step 4: residual magnitude ---
#     mag = np.sqrt(u**2 + v**2)

#     # Fallback for low-texture pixels (invalid LK): use frame-difference signal.
#     # Shadows change luminance but not structure, so large diffs = real object motion.
#     frame_diff = np.abs(f2g - f1g)
#     mag = np.where(valid, mag, frame_diff / _DIFF_SCALE)

#     # --- Step 5: robust threshold + mask cleanup ---
#     mean_mag = float(np.mean(mag))
#     std_mag  = float(np.std(mag))
#     med_mag  = float(np.median(mag))
#     mad_mag  = float(np.median(np.abs(mag - med_mag)))
#     robust_std = 1.4826 * mad_mag

#     threshold = max(
#         mean_mag + K_SIGMA * std_mag,
#         med_mag + 2.0 * robust_std,
#         MIN_ABS_FLOW,
#     )

#     hot_mask = mag > threshold
#     hot_mask = _binary_close(hot_mask, _MASK_RADIUS)
#     hot_mask = _binary_open(hot_mask, _MASK_RADIUS)
#     hot_mask = _filter_connected_components(
#         hot_mask,
#         min_pixels=_MIN_COMPONENT_PIXELS,
#         min_width=_MIN_COMPONENT_WIDTH,
#         min_height=_MIN_COMPONENT_HEIGHT,
#     )

#     hot_counts = np.sum(hot_mask, axis=0)
#     blocked_res_cols = _bridge_column_gaps(hot_counts >= MIN_HOT_PIXELS, _MAX_COLUMN_GAP)

#     blocked = set()
#     for res_col in range(RESIZE_W):
#         if blocked_res_cols[res_col]:
#             blocked.add(int(res_col * cam_w / RESIZE_W))

#     return blocked


# def _binary_dilate(mask, radius=1):
#     """Binary dilation using a square structuring element."""
#     if radius <= 0:
#         return mask.astype(bool)
#     kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
#     return _convolve(kernel, mask.astype(np.float64)) > 0.0


# def _binary_erode(mask, radius=1):
#     """Binary erosion using a square structuring element."""
#     if radius <= 0:
#         return mask.astype(bool)
#     kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
#     required = float(kernel.size)
#     return _convolve(kernel, mask.astype(np.float64)) >= (required - 1e-9)


# def _binary_close(mask, radius=1):
#     return _binary_erode(_binary_dilate(mask, radius), radius)


# def _binary_open(mask, radius=1):
#     return _binary_dilate(_binary_erode(mask, radius), radius)


# def _filter_connected_components(mask, min_pixels, min_width=1, min_height=1):
#     """Keep only connected components that are large enough to be real objects."""
#     h, w = mask.shape
#     visited = np.zeros((h, w), dtype=bool)
#     kept = np.zeros((h, w), dtype=bool)

#     for r in range(h):
#         for c in range(w):
#             if not mask[r, c] or visited[r, c]:
#                 continue

#             stack = [(r, c)]
#             visited[r, c] = True
#             pixels = []
#             r_min = r_max = r
#             c_min = c_max = c

#             while stack:
#                 rr, cc = stack.pop()
#                 pixels.append((rr, cc))

#                 if rr < r_min:
#                     r_min = rr
#                 if rr > r_max:
#                     r_max = rr
#                 if cc < c_min:
#                     c_min = cc
#                 if cc > c_max:
#                     c_max = cc

#                 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
#                     nr = rr + dr
#                     nc = cc + dc
#                     if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
#                         visited[nr, nc] = True
#                         stack.append((nr, nc))

#             comp_h = r_max - r_min + 1
#             comp_w = c_max - c_min + 1
#             if len(pixels) >= min_pixels and comp_w >= min_width and comp_h >= min_height:
#                 for rr, cc in pixels:
#                     kept[rr, cc] = True

#     return kept


# def _bridge_column_gaps(col_mask, max_gap):
#     """Fill short column gaps so one object produces one connected blocked region."""
#     out = np.array(col_mask, dtype=bool)
#     if max_gap <= 0:
#         return out

#     idx = np.flatnonzero(out)
#     if idx.size < 2:
#         return out

#     for left, right in zip(idx[:-1], idx[1:]):
#         gap = right - left - 1
#         if 0 < gap <= max_gap:
#             out[left:right + 1] = True

#     return out


# def _convolve(kernel, array):
#     """2-D convolution (no scipy dependency). Pads with zeros at borders."""
#     kh, kw = kernel.shape
#     ph, pw = kh // 2, kw // 2
#     padded = np.pad(array, ((ph, ph), (pw, pw)), mode='constant')
#     out    = np.zeros_like(array, dtype=np.float64)
#     for i in range(kh):
#         for j in range(kw):
#             out += padded[i:i+array.shape[0], j:j+array.shape[1]] * kernel[i, j]
#     return out


"""
motion_detection.py
-------------------
Minimal, mathematically rigorous pipeline for forward-camera motion detection.
"""

import numpy as np

# ── Tunable parameters ────────────────────────────────────────────────────────
RESIZE_H       = 120    
RESIZE_W       = 160    
MIN_HOT_PIXELS = 5      
MIN_ABS_FLOW   = 1.5    
CAM_HEIGHT_M   = 0.2    
MAX_COLUMN_GAP = 2  # Bridge gaps of 1-3 columns 
MIN_CLUSTER_WIDTH = 3  # A moving object must span at least 3 adjacent columns

_LK_WIN        = 7      
_DET_MIN       = 15.0   

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
    
    if frame1_bgra is None or frame2_bgra is None:
        return set()

    # 1. Bilinear resize + grayscale (Fixes Aliasing)
    f1g = _resize_and_gray(frame1_bgra)
    f2g = _resize_and_gray(frame2_bgra)

    f1g = _convolve(_GAUSSIAN, f1g)
    f2g = _convolve(_GAUSSIAN, f2g)

    # 2. Raw Optical Flow
    u, v, valid = _lk_flow(f1g, f2g)
    
    # --- Step 3: Visual Ego-Motion Compensation ---
    u, v = _compensate_ego_visual(u, v, valid)

    # 4. Residual Calculation (Only on valid pixels)
    mag = np.zeros_like(u)
    mag[valid] = np.sqrt(u[valid]**2 + v[valid]**2)

    # 5. Robust Statistics (Your MAD logic - kept because it's good engineering)
    blocked = set()
    if np.any(valid):
        valid_mags = mag[valid]
        med_mag  = float(np.median(valid_mags))
        mad_mag  = float(np.median(np.abs(valid_mags - med_mag)))
        robust_std = 1.4826 * mad_mag

        threshold = max(med_mag + 3.0 * robust_std, MIN_ABS_FLOW)
        
        # 6. Simple Column Binning & Gap Bridging
        hot_mask = mag > threshold
        hot_counts = np.sum(hot_mask, axis=0)
        
        # Create a 1D boolean array of valid columns
        blocked_cols_1d = hot_counts >= MIN_HOT_PIXELS
        
        # Step A: Bridge small gaps caused by textureless object centers
        blocked_cols_1d = _bridge_1d_gaps(blocked_cols_1d, MAX_COLUMN_GAP)
        
        # Step B: Destroy small isolated clusters (false alarms from robot walking)
        blocked_cols_1d = _remove_1d_noise(blocked_cols_1d, MIN_CLUSTER_WIDTH)
        
        for res_col in range(RESIZE_W):
            if blocked_cols_1d[res_col]:
                blocked.add(int(res_col * cam_w / RESIZE_W))

    return blocked


# ── Private Helpers ───────────────────────────────────────────────────────────
def _remove_1d_noise(col_mask, min_width):
    """
    Removes contiguous blocks of True values that are smaller than min_width.
    Uses C-optimized NumPy edge detection for maximum speed.
    """
    out = col_mask.copy()
    if min_width <= 1:
        return out

    # Pad with 0s to easily detect edges at the boundaries of the array
    padded = np.pad(out, (1, 1), mode='constant').astype(np.int8)
    
    # +1 means transition to True (start), -1 means transition to False (end)
    edges = np.diff(padded)
    
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    # Slice out any cluster that is too narrow
    for s, e in zip(starts, ends):
        if (e - s) < min_width:
            out[s:e] = False

    return out

def _bridge_1d_gaps(col_mask, max_gap):
    """
    Fills short gaps in a 1D boolean array.
    Extremely fast because it only iterates over the detected indices, not the whole image.
    """
    out = col_mask.copy()
    idx = np.flatnonzero(out)
    
    # If fewer than 2 columns are blocked, there are no gaps to bridge
    if idx.size < 2 or max_gap <= 0:
        return out
        
    for left, right in zip(idx[:-1], idx[1:]):
        # Calculate distance between two "hot" columns
        gap = right - left - 1
        if 0 < gap <= max_gap:
            # Slice assignment is C-optimized in NumPy
            out[left:right] = True
            
    return out

def _resize_and_gray(img_bgra):
    """Fast bilinear interpolation + grayscale in one pass using NumPy meshgrid."""
    h, w = img_bgra.shape[:2]
    
    # Create coordinate grid
    x = np.linspace(0, w - 1, RESIZE_W)
    y = np.linspace(0, h - 1, RESIZE_H)
    xv, yv = np.meshgrid(x, y)

    # Get integer coords and fractional weights
    x0 = np.floor(xv).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(yv).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)

    wx = xv - x0
    wy = yv - y0

    # Convert to grayscale immediately to save memory bandwidth
    gray = 0.299*img_bgra[:,:,2] + 0.587*img_bgra[:,:,1] + 0.114*img_bgra[:,:,0]

    # Bilinear mix
    top = gray[y0, x0] * (1 - wx) + gray[y0, x1] * wx
    bot = gray[y1, x0] * (1 - wx) + gray[y1, x1] * wx
    out = top * (1 - wy) + bot * wy
    
    return out.astype(np.float64)


def _build_2d_depth_map(lidar_ranges, lidar_fov, cam_hfov):
    """
    Fuses 1D LiDAR with a flat-ground assumption to create a realistic 3D depth map.
    """
    # Focal length in pixels
    fx = (RESIZE_W / 2.0) / np.tan(cam_hfov / 2.0)
    
    # 1. Build the 1D LiDAR map for the horizon
    lidar_depth_1d = np.full(RESIZE_W, 5.0, dtype=np.float64) # default max range
    if lidar_ranges is not None and lidar_fov is not None:
        ranges = np.array(lidar_ranges, dtype=np.float64)
        n = len(ranges)
        ray_angles = -lidar_fov/2 + np.arange(n) / max(n-1, 1) * lidar_fov
        in_fov = np.abs(ray_angles) <= cam_hfov / 2
        valid  = in_fov & (ranges > 0) & np.isfinite(ranges)
        
        if np.any(valid):
            valid_angles = ray_angles[valid]
            valid_ranges = ranges[valid]
            for col in range(RESIZE_W):
                col_angle = np.arctan((col - RESIZE_W/2) / fx)
                nearest = int(np.argmin(np.abs(valid_angles - col_angle)))
                lidar_depth_1d[col] = float(valid_ranges[nearest])

    # 2. Project to 2D using Ground Plane Math
    depth_2d = np.zeros((RESIZE_H, RESIZE_W), dtype=np.float64)
    cy = RESIZE_H / 2.0
    
    for row in range(RESIZE_H):
        # Angle of this row below/above the horizon
        row_angle = np.arctan((row - cy) / fx)
        
        if row_angle > 0.05:  # Looking down at the floor
            # Z = height / tan(angle)
            ground_z = CAM_HEIGHT_M / np.tan(row_angle)
            # Take the closer value: the floor, or the lidar hit
            depth_2d[row, :] = np.minimum(lidar_depth_1d, ground_z)
        else:
            # Looking at horizon or above: trust the LiDAR
            depth_2d[row, :] = lidar_depth_1d
            
    return np.maximum(depth_2d, 0.05) # Clamp to avoid div/0


def _compensate_ego(u, v, valid, pose1, pose2, cam_hfov, depth_2d):
    """Calculates expected optical flow given true 2D depth."""
    x1, y1, yaw1 = pose1
    x2, y2, yaw2 = pose2

    d_yaw = yaw2 - yaw1
    if d_yaw >  np.pi: d_yaw -= 2*np.pi
    if d_yaw < -np.pi: d_yaw += 2*np.pi

    cos_y = np.cos(yaw1)
    sin_y = np.sin(yaw1)
    dx_w  = x2 - x1
    dy_w  = y2 - y1
    d_forward = dx_w * cos_y  + dy_w * sin_y
    d_lateral = -dx_w * sin_y + dy_w * cos_y

    fx = (RESIZE_W / 2.0) / np.tan(cam_hfov / 2.0)

    col_centres = np.arange(RESIZE_W, dtype=np.float64) - RESIZE_W / 2.0
    row_centres = np.arange(RESIZE_H, dtype=np.float64) - RESIZE_H / 2.0
    X = np.tile(col_centres, (RESIZE_H, 1))
    Y = np.tile(row_centres.reshape(-1,1), (1, RESIZE_W))

    # Calculate expected flow
    ego_u  = d_yaw * fx * np.ones((RESIZE_H, RESIZE_W))
    ego_v  = np.zeros((RESIZE_H, RESIZE_W))
    
    ego_u += d_forward * X / depth_2d
    ego_v += d_forward * Y / depth_2d
    ego_u += fx * d_lateral / depth_2d

    res_u = np.where(valid, u - ego_u, 0.0)
    res_v = np.where(valid, v - ego_v, 0.0)

    return res_u, res_v

def _compensate_ego_visual(u, v, valid):
    """
    Calculates background motion directly from the image flow using
    a 2-pass Iteratively Reweighted Least Squares (IRLS) Affine fit.
    Ignores odometry to bypass hardware synchronization errors.
    """
    if not np.any(valid):
        return np.zeros_like(u), np.zeros_like(v)

    h, w = u.shape
    # Create coordinate grid for every pixel
    Y_grid, X_grid = np.mgrid[0:h, 0:w]

    # Extract coordinates and vectors ONLY for valid, textured pixels
    X = X_grid[valid].astype(np.float64)
    Y = Y_grid[valid].astype(np.float64)
    U = u[valid]
    V = v[valid]

    # If the image is completely washed out, abort
    if len(X) < 10: 
        return np.zeros_like(u), np.zeros_like(v)

    # Design matrix for 2D Affine transform (Scale, Shear, Translation)
    # Equation: u = m1*x + m2*y + tx
    A = np.column_stack([X, Y, np.ones_like(X)])

    # --- PASS 1: Coarse Global Fit ---
    # Fit the affine model to all valid pixels. 
    pu = np.linalg.lstsq(A, U, rcond=None)[0]
    pv = np.linalg.lstsq(A, V, rcond=None)[0]

    # Calculate how badly each pixel deviates from the global average
    pred_U = A @ pu
    pred_V = A @ pv
    errors = np.sqrt((U - pred_U)**2 + (V - pred_V)**2)

    # --- PASS 2: Outlier Rejection ---
    # Assuming at least half the camera sees the static background,
    # the median error isolates the background from the moving objects.
    median_err = np.median(errors)
    bg_mask = errors <= median_err

    # Safety fallback if something goes entirely wrong
    if np.sum(bg_mask) < 10:
        bg_mask = np.ones_like(errors, dtype=bool) 

    # --- PASS 3: Refined Background Fit ---
    # Recalculate the exact background motion ignoring the moving objects
    pu_clean = np.linalg.lstsq(A[bg_mask], U[bg_mask], rcond=None)[0]
    pv_clean = np.linalg.lstsq(A[bg_mask], V[bg_mask], rcond=None)[0]

    # --- APPLY COMPENSATION ---
    # Calculate expected background motion for the ENTIRE image
    A_full = np.column_stack([X_grid.ravel(), Y_grid.ravel(), np.ones_like(X_grid.ravel())])
    ego_u_flat = A_full @ pu_clean
    ego_v_flat = A_full @ pv_clean

    ego_u = ego_u_flat.reshape(h, w)
    ego_v = ego_v_flat.reshape(h, w)

    # Subtract the calculated background motion from the raw motion
    res_u = np.where(valid, u - ego_u, 0.0)
    res_v = np.where(valid, v - ego_v, 0.0)

    return res_u, res_v

def _lk_flow(gray1, gray2):
    # [Same as your previous implementation, unmodified]
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

def _convolve(kernel, array):
    # [Same as your previous implementation, unmodified]
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(array, ((ph, ph), (pw, pw)), mode='constant')
    out    = np.zeros_like(array, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += padded[i:i+array.shape[0], j:j+array.shape[1]] * kernel[i, j]
    return out