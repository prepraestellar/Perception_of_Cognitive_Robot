import math
import heapq
import numpy as np
from controller import Supervisor, Display
from motion_detection import get_blocked_columns

# =============================================================
# PART 1: GRAPH-BASED MAP
# =============================================================
class GraphMap:
    """
    Dictionary-based occupancy map.
    Key   = (grid_x, grid_y) integer tuple
    Value = probability of being a wall (0.0 to 1.0)
    Grows in any direction — no fixed size like Milestone 2.
    """

    def __init__(self, resolution=0.05):
        self.nodes = {}
        self.resolution = resolution

    def world_to_grid(self, wx, wy):
        gx = int(math.floor(wx / self.resolution))
        gy = int(math.floor(wy / self.resolution))
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + self.resolution / 2
        wy = gy * self.resolution + self.resolution / 2
        return (wx, wy)

    def add_to_map(self, wx, wy, increment=0.01):
        key = self.world_to_grid(wx, wy)
        self.nodes[key] = min(1.0, self.nodes.get(key, 0.0) + increment)

    def is_occupied(self, gx, gy, threshold=0.5):
        val = self.nodes.get((gx, gy))
        if val is None:
            return None          # unknown cell
        return val >= threshold  # True = wall, False = free

# =============================================================
# PART 3: A* PATH FINDING
# =============================================================
def a_star(start, goal, graph_map, threshold=0.5, max_iter=20000):
    """
    Finds a path from start to goal on the graph map.
    Returns list of (gx, gy) waypoints, or None if unreachable.
    """
    DIRECTIONS = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    iterations = 0

    while open_set:
        iterations += 1
        if iterations > max_iter:
            return None

        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if graph_map.is_occupied(neighbor[0], neighbor[1], threshold) is True:
                continue  # wall — skip
            tentative_g = g_score[current] + math.dist(current, neighbor)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + math.dist(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None  # no path found


# =============================================================
# PART 4: ROBOT CONTROLLER
# =============================================================
class MyRobot:

    MAX_SPEED = 6.28

    def __init__(self, mode="AUTO"):
        self.supervisor = Supervisor()
        self.time_step  = int(self.supervisor.getBasicTimeStep())
        self.mode       = mode  # "TELEOP", "AUTO", or "EXPLORE"

        # --- Devices ---
        self.lidar = self.supervisor.getDevice('LDS-01')
        self.lidar.enable(self.time_step)

        self.camera = self.supervisor.getDevice('camera')
        self.camera.enable(self.time_step)
        self.camera_w = self.camera.getWidth()
        self.camera_h = self.camera.getHeight()

        self.keyboard = self.supervisor.getKeyboard()
        self.keyboard.enable(self.time_step)

        self.display = self.supervisor.getDevice('display')
        self.display_w = self.display.getWidth()
        self.display_h = self.display.getHeight()

        self.global_display = self.supervisor.getDevice('global_display')
        self.gdisplay_w = self.global_display.getWidth()
        self.gdisplay_h = self.global_display.getHeight()

        self.motion_display = self.supervisor.getDevice('motion_display')
        if self.motion_display is not None:
            self.mdisplay_w = self.motion_display.getWidth()
            self.mdisplay_h = self.motion_display.getHeight()
        else:
            self.mdisplay_w = 0
            self.mdisplay_h = 0

        self.left_motor  = self.supervisor.getDevice('left wheel motor')
        self.right_motor = self.supervisor.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # --- Wheel encoders ---
        self.left_encoder  = self.supervisor.getDevice('left wheel sensor')
        self.right_encoder = self.supervisor.getDevice('right wheel sensor')
        self.left_encoder.enable(self.time_step)
        self.right_encoder.enable(self.time_step)

        # --- Compass for heading ---
        self.compass = self.supervisor.getDevice('compass')
        self.compass.enable(self.time_step)

        # --- IMU heading source (optional); fallback stays compass ---
        self.imu = self.supervisor.getDevice('inertial unit')
        if self.imu is not None:
            self.imu.enable(self.time_step)
        self._imu_yaw_offset = 0.0
        self._imu_calibrated = False

        # --- Odometry state ---
        self.wheel_radius   = 0.033
        self.axle_length    = 0.160
        self.x              = 0.0
        self.y              = 0.0
        self.theta          = 0.0
        self.prev_left_enc  = 0.0
        self.prev_right_enc = 0.0

        # EKF state [x, y, theta] and covariance.
        self._ekf_state = np.array([self.x, self.y, self.theta], dtype=np.float64)
        self._ekf_cov   = np.diag([1e-3, 1e-3, 2e-3]).astype(np.float64)

        # --- Systems ---
        self.graph_map  = GraphMap(resolution=0.05)
        self.prev_frame      = None
        self.prev_pose = None
        self._blocked_buffer = []
        self._BLOCK_HISTORY  = 4   # union of last 12 frames (~384 ms)

        # Store starting grid cell for utility scoring
        rx, ry, _ = self.get_pose()
        self.start_grid = self.graph_map.world_to_grid(rx, ry)

        # EXPLORE mode state
        self.current_path = []
        self.target_grid  = None

        # AUTO mode: committed turn to prevent oscillation
        self._avoid_timer = 0
        self._avoid_dir   = 1   # +1 = turn left, -1 = turn right

    # ----------------------------------------------------------
    # ODOMETRY + POSE
    # ----------------------------------------------------------
    # Noise parameters
    _ENC_NOISE_STD     = 0.002   # metres — Gaussian noise on each wheel displacement
    _IMU_NOISE_STD     = 0.02    # radians (~1.1°) — Gaussian noise on heading sensor

    # EKF tuning (variance terms)
    _KF_Q_XY_BASE      = 2e-4
    _KF_Q_XY_SCALE     = 4e-3
    _KF_Q_TH_BASE      = 4e-4
    _KF_Q_TH_SCALE     = 4e-3
    _KF_R_HEADING_MIN  = 1e-4

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _read_heading_measurement(self):
        """
        Heading measurement for EKF correction.
        Prefer inertial unit yaw when available, aligned to compass frame once.
        """
        if self.imu is not None and hasattr(self.imu, 'getRollPitchYaw'):
            rpy = self.imu.getRollPitchYaw()
            imu_yaw = self._wrap_angle(-rpy[2])

            if not self._imu_calibrated:
                cv = self.compass.getValues()
                compass_yaw = self._wrap_angle(math.atan2(cv[0], cv[1]))
                self._imu_yaw_offset = self._wrap_angle(compass_yaw - imu_yaw)
                self._imu_calibrated = True

            return self._wrap_angle(imu_yaw + self._imu_yaw_offset)

        cv = self.compass.getValues()
        return self._wrap_angle(math.atan2(cv[0], cv[1]))

    def update_odometry(self):
        left_enc  = self.left_encoder.getValue()
        right_enc = self.right_encoder.getValue()
        d_left  = (left_enc  - self.prev_left_enc)  * self.wheel_radius
        d_right = (right_enc - self.prev_right_enc) * self.wheel_radius
        self.prev_left_enc  = left_enc
        self.prev_right_enc = right_enc

        # Encoder noise
        d_left  += np.random.normal(0.0, self._ENC_NOISE_STD)
        d_right += np.random.normal(0.0, self._ENC_NOISE_STD)

        # 1) Prediction from wheel encoders (odometry model).
        d_center = (d_right + d_left) / 2.0
        d_theta  = (d_right - d_left) / self.axle_length

        x, y, theta = self._ekf_state
        theta_mid   = theta + 0.5 * d_theta

        x_pred     = x + d_center * math.cos(theta_mid)
        y_pred     = y + d_center * math.sin(theta_mid)
        theta_pred = self._wrap_angle(theta + d_theta)
        state_pred = np.array([x_pred, y_pred, theta_pred], dtype=np.float64)

        F = np.array([
            [1.0, 0.0, -d_center * math.sin(theta_mid)],
            [0.0, 1.0,  d_center * math.cos(theta_mid)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        q_xy = self._KF_Q_XY_BASE + abs(d_center) * self._KF_Q_XY_SCALE
        q_th = self._KF_Q_TH_BASE + abs(d_theta)  * self._KF_Q_TH_SCALE
        Q = np.diag([q_xy, q_xy, q_th])

        cov_pred = F @ self._ekf_cov @ F.T + Q

        # 2) Correction from IMU/compass heading.
        heading_meas = self._read_heading_measurement()
        heading_meas = self._wrap_angle(heading_meas + np.random.normal(0.0, self._IMU_NOISE_STD))

        H = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        R = max(self._KF_R_HEADING_MIN, self._IMU_NOISE_STD ** 2)

        innovation = self._wrap_angle(heading_meas - theta_pred)
        S = float((H @ cov_pred @ H.T)[0, 0] + R)
        K = (cov_pred @ H.T) / S

        self._ekf_state = state_pred + K[:, 0] * innovation
        self._ekf_state[2] = self._wrap_angle(self._ekf_state[2])
        self._ekf_cov = (np.eye(3) - K @ H) @ cov_pred

        self.x, self.y, self.theta = self._ekf_state.tolist()

    def get_pose(self):
        return self.x, self.y, -self.theta

    # ----------------------------------------------------------
    # MAPPING  (camera-gated LiDAR)
    # ----------------------------------------------------------
    def update_map(self, blocked_cols):
        """
        Only map LiDAR rays that the camera can see.
        Within that arc, skip rays whose angle overlaps a detected moving object.
        """
        rx, ry, ryaw = self.get_pose()

        ranges    = self.lidar.getRangeImage()
        n         = len(ranges)
        lidar_fov = self.lidar.getFov()   # ~2π
        cam_hfov  = self.camera.getFov()  # ~1.085 rad (~62°)
        cam_w     = self.camera.getWidth()

        for i, dist in enumerate(ranges):
            # Angle of this ray relative to robot forward
            ray_angle = -lidar_fov / 2 + (i / (n - 1)) * lidar_fov

            # Only process rays the camera can see
            if abs(ray_angle) > cam_hfov / 2:
                continue

            # Skip bad readings
            if dist <= 0 or math.isinf(dist) or dist > 1.5:
                continue

            # Map ray angle to camera pixel column
            px = (0.5 - ray_angle / cam_hfov) * cam_w

            # Skip if column belongs to a detected moving object (±1 px tolerance)
            is_blocked = any(int(px + offset) in blocked_cols for offset in (-1, 0, 1))
            if is_blocked:
                continue

            # Mark free cells along the ray
            res = self.graph_map.resolution
            steps = int(dist / res)
            for s in range(1, steps):
                fx = rx + (s * res) * math.cos(ryaw + ray_angle)
                fy = ry - (s * res) * math.sin(ryaw + ray_angle)
                key = self.graph_map.world_to_grid(fx, fy)
                if key not in self.graph_map.nodes:
                    self.graph_map.nodes[key] = 0.0  # free

            # Mark wall cell at ray endpoint
            wx = rx + dist * math.cos(ryaw + ray_angle)
            wy = ry - dist * math.sin(ryaw + ray_angle)
            self.graph_map.add_to_map(wx, wy)

    # ----------------------------------------------------------
    # DISPLAY
    # ----------------------------------------------------------
    def draw_map(self):
        rx, ry, _ = self.get_pose()
        robot_gx, robot_gy = self.graph_map.world_to_grid(rx, ry)
        cx, cy = self.display_w // 2, self.display_h // 2

        # Light grey background = undiscovered
        self.display.setColor(0xC0C0C0)
        self.display.fillRectangle(0, 0, self.display_w, self.display_h)

        frontier_set = set(self._get_frontiers())

        # Draw each known cell
        for (gx, gy), occ in self.graph_map.nodes.items():
            rel_x = gx - robot_gx
            rel_y = gy - robot_gy
            px = cx - rel_y
            py = cy - rel_x
            if not (0 <= px < self.display_w and 0 <= py < self.display_h):
                continue
            if occ >= 0.5:
                self.display.setColor(0x000000)   # occupied = black
            elif (gx, gy) in frontier_set:
                self.display.setColor(0x00BFFF)   # frontier = light blue
            else:
                self.display.setColor(0xFFFFFF)   # free = white
            self.display.drawPixel(px, py)

        # Robot = red dot at centre
        self.display.setColor(0xFF0000)
        self.display.fillRectangle(cx - 2, cy - 2, 4, 4)

    # ----------------------------------------------------------
    # GLOBAL MAP DISPLAY  (fixed-origin, full explored area)
    # ----------------------------------------------------------
    def draw_global_map(self):
        if not self.graph_map.nodes:
            return

        keys = list(self.graph_map.nodes.keys())
        min_gx = min(k[0] for k in keys)
        max_gx = max(k[0] for k in keys)
        min_gy = min(k[1] for k in keys)
        max_gy = max(k[1] for k in keys)

        span_x = max(max_gx - min_gx + 1, 1)
        span_y = max(max_gy - min_gy + 1, 1)
        scale  = min(self.gdisplay_w / span_y, self.gdisplay_h / span_x)
        scale  = max(scale, 1.0)

        # Light grey background = undiscovered
        self.global_display.setColor(0xC0C0C0)
        self.global_display.fillRectangle(0, 0, self.gdisplay_w, self.gdisplay_h)

        frontier_set = set(self._get_frontiers())

        sz = max(2, math.ceil(scale))
        for (gx, gy), occ in self.graph_map.nodes.items():
            px = int((gy - min_gy) * scale)
            py = int((gx - min_gx) * scale)
            if not (0 <= px < self.gdisplay_w and 0 <= py < self.gdisplay_h):
                continue
            if occ >= 0.5:
                self.global_display.setColor(0x000000)   # occupied = black
            elif (gx, gy) in frontier_set:
                self.global_display.setColor(0x00BFFF)   # frontier = light blue
            else:
                self.global_display.setColor(0xFFFFFF)   # free = white
            self.global_display.fillRectangle(px, py, sz, sz)

        # Robot position in red
        rx, ry, _ = self.get_pose()
        rgx, rgy  = self.graph_map.world_to_grid(rx, ry)
        rpx = int((rgy - min_gy) * scale)
        rpy = int((rgx - min_gx) * scale)
        dot = max(sz * 3, 5)
        self.global_display.setColor(0xFF0000)
        self.global_display.fillRectangle(rpx - dot // 2, rpy - dot // 2, dot, dot)

    def draw_motion_display(self, frame_bgra, blocked_cols, effective_blocked):
        """
        Draw live camera view, then overlay motion columns:
        - red: buffered columns used by mapping
        - green: columns detected in current frame
        """
        if self.motion_display is None:
            return

        cam_h, cam_w = frame_bgra.shape[:2]
        if self.mdisplay_w == cam_w and self.mdisplay_h == cam_h:
            bgra_bytes = frame_bgra.tobytes()
        else:
            # Nearest-neighbor resize so camera content fits the motion display.
            x_idx = (np.arange(self.mdisplay_w) * cam_w / self.mdisplay_w).astype(np.int32)
            y_idx = (np.arange(self.mdisplay_h) * cam_h / self.mdisplay_h).astype(np.int32)
            scaled = frame_bgra[y_idx[:, None], x_idx[None, :], :]
            bgra_bytes = np.ascontiguousarray(scaled).tobytes()

        image_ref = self.motion_display.imageNew(
            bgra_bytes,
            Display.BGRA,
            self.mdisplay_w,
            self.mdisplay_h,
        )
        self.motion_display.imagePaste(image_ref, 0, 0, False)
        self.motion_display.imageDelete(image_ref)

        cam_w = max(cam_w, 1)

        # Buffered detections used for mapping.
        self.motion_display.setColor(0xFF4040)
        for col in effective_blocked:
            x = int(col * self.mdisplay_w / cam_w)
            if 0 <= x < self.mdisplay_w:
                self.motion_display.drawLine(x, 0, x, self.mdisplay_h - 1)

        # Current-frame detections.
        self.motion_display.setColor(0x00FF66)
        band_h = max(2, self.mdisplay_h // 6)
        for col in blocked_cols:
            x = int(col * self.mdisplay_w / cam_w)
            if 0 <= x < self.mdisplay_w:
                self.motion_display.drawLine(x, 0, x, band_h)

        # Camera center column marker.
        cx = self.mdisplay_w // 2
        self.motion_display.setColor(0xFFD700)
        self.motion_display.drawLine(cx, 0, cx, self.mdisplay_h - 1)

        self.motion_display.setColor(0x000000)
        self.motion_display.fillRectangle(0, 0, 130, 18)
        self.motion_display.setColor(0xFFFFFF)
        self.motion_display.drawText(
            f"R buf:{len(effective_blocked)} G now:{len(blocked_cols)}",
            2,
            2,
        )

    # ----------------------------------------------------------
    # NAVIGATION MODE: TELEOP  (W/A/S/D, press M to go AUTO)
    # ----------------------------------------------------------
    def handle_teleop(self):
        key = self.keyboard.getKey()
        left, right = 0.0, 0.0

        if   key == ord('W'): left = right = self.MAX_SPEED
        elif key == ord('S'): left = right = -self.MAX_SPEED
        elif key == ord('A'): left, right = -self.MAX_SPEED * 0.4, self.MAX_SPEED * 0.4
        elif key == ord('D'): left, right = self.MAX_SPEED * 0.4, -self.MAX_SPEED * 0.4
        elif key == ord('M'):
            self.mode = "AUTO"
            print("Switched to AUTO mode")

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    # ----------------------------------------------------------
    # NAVIGATION MODE: AUTO  (random walk, press M to go TELEOP)
    # ----------------------------------------------------------
    def handle_auto(self):
        ranges = self.lidar.getRangeImage()
        n      = len(ranges)

        # Full 360° LiDAR for obstacle avoidance (mapping uses camera FOV only)
        # Positive ray indices = right side (wy = ry - dist*sin)
        front       = min(ranges[n // 2 - 20 : n // 2 + 20])
        front_right = min(ranges[n // 2 + 20 : n // 2 + 80])
        front_left  = min(ranges[n // 2 - 80 : n // 2 - 20])

        if self._avoid_timer > 0:
            # Committed spin for a frontal obstacle — don't re-evaluate until clear
            self._avoid_timer -= 1
            spd = self.MAX_SPEED * 0.4
            self.left_motor.setVelocity(-self._avoid_dir * spd)
            self.right_motor.setVelocity( self._avoid_dir * spd)
        elif front < 0.25:
            # Frontal obstacle — pick open side and commit to spinning
            self._avoid_dir   = 1 if front_left >= front_right else -1
            self._avoid_timer = 25
            spd = self.MAX_SPEED * 0.4
            self.left_motor.setVelocity(-self._avoid_dir * spd)
            self.right_motor.setVelocity( self._avoid_dir * spd)
        elif front_right < 0.20:
            # Wall on the right — curve left, no spin commit
            self.left_motor.setVelocity(self.MAX_SPEED * 0.3)
            self.right_motor.setVelocity(self.MAX_SPEED * 0.8)
        elif front_left < 0.20:
            # Wall on the left — curve right, no spin commit
            self.left_motor.setVelocity(self.MAX_SPEED * 0.8)
            self.right_motor.setVelocity(self.MAX_SPEED * 0.3)
        else:
            self.left_motor.setVelocity(self.MAX_SPEED)
            self.right_motor.setVelocity(self.MAX_SPEED)

        if self.keyboard.getKey() == ord('M'):
            self.mode = "TELEOP"
            print("Switched to TELEOP mode")

    # =============================================================
    # PART 5: EXPLORE MODE  (frontier-based A* navigation)
    # =============================================================
    def handle_explore(self):
        if self.keyboard.getKey() == ord('M'):
            self.mode = "TELEOP"
            print("Switched to TELEOP mode")
            return

        rx, ry, ryaw = self.get_pose()
        robot_gx, robot_gy = self.graph_map.world_to_grid(rx, ry)

        # Follow existing path if we have one
        if self.current_path:
            self._follow_path(rx, ry, ryaw)
            return

        # Find frontier cells (free cells next to unknown cells)
        frontiers = self._get_frontiers()
        if not frontiers:
            # Exploration complete — spin slowly
            self.left_motor.setVelocity(1.0)
            self.right_motor.setVelocity(-1.0)
            return

        # Group frontiers into connected regions
        regions = self._get_frontier_regions(frontiers)

        target = self._choose_target(regions, robot_gx, robot_gy)
        if target is None:
            self.left_motor.setVelocity(self.MAX_SPEED * 0.3)
            self.right_motor.setVelocity(-self.MAX_SPEED * 0.3)
            return

        # Plan path with A*
        path = a_star((robot_gx, robot_gy), target, self.graph_map)
        if not path:  # None (unreachable) or [] (already there)
            self.current_path = [target]
            self.target_grid  = target
        else:
            self.current_path = path
            self.target_grid  = target

    def _get_frontiers(self):
        """Free cells that have at least one unknown 4-connected neighbour."""
        DIRS = [(0,1),(0,-1),(1,0),(-1,0)]
        frontiers = []
        for (gx, gy), occ in self.graph_map.nodes.items():
            if occ >= 0.5:
                continue  # skip walls
            for dx, dy in DIRS:
                if (gx + dx, gy + dy) not in self.graph_map.nodes:
                    frontiers.append((gx, gy))
                    break
        return frontiers

    def _get_frontier_regions(self, frontiers):
        """Group frontier cells into connected clusters (4-connected flood fill)."""
        frontier_set = set(frontiers)
        visited = set()
        regions = []
        DIRS4 = [(0,1),(0,-1),(1,0),(-1,0)]

        for cell in frontiers:
            if cell in visited:
                continue
            region = []
            queue  = [cell]
            while queue:
                cur = queue.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                if cur in frontier_set:
                    region.append(cur)
                    for dx, dy in DIRS4:
                        neighbor = (cur[0] + dx, cur[1] + dy)
                        if neighbor not in visited:
                            queue.append(neighbor)
            if region:
                regions.append(region)
        return regions

    def _bfs_distances(self, start):
        from collections import deque
        DIRS = [(0,1),(0,-1),(1,0),(-1,0)]
        dist = {start: 0}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in DIRS:
                nb = (cx+dx, cy+dy)
                if nb not in dist and self.graph_map.is_occupied(nb[0], nb[1]) is False:
                    dist[nb] = dist[(cx, cy)] + 1
                    q.append(nb)
        return dist

    def _choose_target(self, regions, robot_gx, robot_gy):
        ALPHA = 1.0
        BETA  = 0.5

        dist_map = self._bfs_distances((robot_gx, robot_gy))

        best_score  = -float('inf')
        best_target = None

        for region in regions:
            reachable = [(gx, gy) for gx, gy in region if (gx, gy) in dist_map]
            if not reachable:
                continue

            T = min(dist_map[(gx, gy)] for gx, gy in reachable)
            D = sum(math.dist((gx, gy), self.start_grid) for gx, gy in region)
            score = ALPHA * D - BETA * T

            if score > best_score:
                best_score  = score
                best_target = min(reachable, key=lambda c: dist_map[c])

        return best_target

    def _follow_path(self, rx, ry, ryaw):
        """Drive toward the next waypoint in the path."""
        if not self.current_path:
            return

        gx, gy = self.current_path[0]
        if self.graph_map.is_occupied(gx, gy) is True:
            self.current_path = []  # waypoint became a wall — replan
            return

        gx, gy     = self.current_path[0]
        wx, wy     = self.graph_map.grid_to_world(gx, gy)
        dx, dy     = wx - rx, wy - ry
        dist       = math.hypot(dx, dy)

        if dist < 0.08:
            self.current_path.pop(0)  # waypoint reached
            return

        target_yaw = math.atan2(-dy, dx)
        yaw_error  = math.atan2(math.sin(target_yaw - ryaw),
                                math.cos(target_yaw - ryaw))

        if abs(yaw_error) > 0.15:
            # Turn toward waypoint
            turn = self.MAX_SPEED * 0.4 * (1 if yaw_error > 0 else -1)
            self.left_motor.setVelocity(-turn)
            self.right_motor.setVelocity( turn)
        else:
            # Drive forward
            self.left_motor.setVelocity(self.MAX_SPEED * 0.6)
            self.right_motor.setVelocity(self.MAX_SPEED * 0.6)

    # ----------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------
    def run(self):
        print(f"Milestone 3 v2 started — mode: {self.mode}")
        print("Controls: W/A/S/D = move, M = toggle mode")
        step = 0

        while self.supervisor.step(self.time_step) != -1:
            step += 1

            # 0. Update odometry (encoders + compass)
            self.update_odometry()

            # 1. Camera → detect moving object columns
            img_raw = self.camera.getImage()
            if img_raw is None:
                continue
            img = np.frombuffer(img_raw, dtype=np.uint8).reshape((self.camera_h, self.camera_w, 4))
            pose_now = self.get_pose()   # (x, y, yaw)
            blocked_cols = get_blocked_columns(
                self.prev_frame, img,
                self.camera_w,
                pose1=self.prev_pose,
                pose2=pose_now,
                cam_hfov=self.camera.getFov(),
                lidar_ranges=self.lidar.getRangeImage(),
                lidar_fov=self.lidar.getFov()
            )
            self.prev_pose  = pose_now
            self.prev_frame = img.copy()

            # Union of last _BLOCK_HISTORY frames so a stationary ball stays blocked
            self._blocked_buffer.append(blocked_cols)
            if len(self._blocked_buffer) > self._BLOCK_HISTORY:
                self._blocked_buffer.pop(0)
            effective_blocked = set().union(*self._blocked_buffer)
            if step % 30 == 0:
                print(f"[DBG] step={step}  blocked={len(effective_blocked)} cols")

            # Debug view of motion detection output.
            self.draw_motion_display(img, blocked_cols, effective_blocked)

            # 2. Update map (camera-gated LiDAR)
            self.update_map(effective_blocked)

            # 3. Draw maps every 5 steps
            if step % 5 == 0:
                self.draw_map()
                self.draw_global_map()

            # 4. Navigation
            if   self.mode == "TELEOP":  self.handle_teleop()
            elif self.mode == "AUTO":    self.handle_auto()
            elif self.mode == "EXPLORE": self.handle_explore()


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    MODE = "TELEOP"   # Change to "EXPLORE" to enable A* frontier navigation
    MyRobot(mode=MODE).run()
