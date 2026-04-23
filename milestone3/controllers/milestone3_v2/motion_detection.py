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