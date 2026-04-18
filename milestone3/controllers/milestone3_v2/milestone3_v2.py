import math
import heapq
import numpy as np
from controller import Supervisor
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

    def add_to_map(self, wx, wy, increment=0.1):
        key = self.world_to_grid(wx, wy)
        self.nodes[key] = min(1.0, self.nodes.get(key, 0.0) + increment)

    def decay(self, rate=0.003):
        """Subtract rate from every cell each step.
        Real walls are re-mapped continuously and stay near 1.0.
        Ball leaks (1-2 frames, probability 0.1-0.2) fade to 0 in ~70 steps (~2 s).
        """
        to_delete = [k for k, v in self.nodes.items() if v <= rate]
        for k in to_delete:
            del self.nodes[k]
        for k in self.nodes:
            self.nodes[k] = max(0.0, self.nodes[k] - rate)

    def is_occupied(self, gx, gy, threshold=0.5):
        val = self.nodes.get((gx, gy))
        if val is None:
            return None          # unknown cell
        return val >= threshold  # True = wall, False = free
#wat

# =============================================================
# PART 3: A* PATH FINDING
# =============================================================
def a_star(start, goal, graph_map, threshold=0.5, max_iter=5000):
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

        self.keyboard = self.supervisor.getKeyboard()
        self.keyboard.enable(self.time_step)

        self.display = self.supervisor.getDevice('display')
        self.display_w = self.display.getWidth()
        self.display_h = self.display.getHeight()

        self.left_motor  = self.supervisor.getDevice('left wheel motor')
        self.right_motor = self.supervisor.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # --- Odometry (Supervisor gives perfect pose) ---
        self.robot_node  = self.supervisor.getSelf()
        self.trans_field = self.robot_node.getField('translation')
        self.rot_field   = self.robot_node.getField('rotation')

        # --- Systems ---
        self.graph_map  = GraphMap(resolution=0.05)
        self.prev_frame      = None
        self._blocked_buffer = []
        self._BLOCK_HISTORY  = 5    # union of last 8 frames (~256 ms)

        # Store starting grid cell for utility scoring
        rx, ry, _ = self.get_pose()
        self.start_grid = self.graph_map.world_to_grid(rx, ry)

        # EXPLORE mode state
        self.current_path      = []
        self.target_grid       = None
        self.unreachable       = set()

        # AUTO mode: committed turn to prevent oscillation
        self._avoid_timer = 0
        self._avoid_dir   = 1   # +1 = turn left, -1 = turn right

    # ----------------------------------------------------------
    # POSE
    # ----------------------------------------------------------
    def get_pose(self):
        pos = self.trans_field.getSFVec3f()
        rot = self.rot_field.getSFRotation()   # [ax, ay, az, angle]
        yaw = -rot[3] if rot[2] > 0 else rot[3]
        return float(pos[0]), float(pos[1]), yaw

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

        # Convert blocked pixel columns → blocked angles (camera space)
        # px = (0.5 - angle/cam_hfov) * cam_w  →  angle = (0.5 - px/cam_w) * cam_hfov
        blocked_angles = [(0.5 - px / cam_w) * cam_hfov for px in blocked_cols]

        # Angular width of one camera pixel — used as match tolerance
        angle_per_pixel = cam_hfov / cam_w

        for i, dist in enumerate(ranges):
            # Angle of this ray relative to robot forward
            ray_angle = -lidar_fov / 2 + (i / (n - 1)) * lidar_fov

            # Only process rays the camera can see
            if abs(ray_angle) > cam_hfov / 2:
                continue

            # Skip if this ray angle matches a blocked (moving object) angle
            if any(abs(ray_angle - ba) < angle_per_pixel for ba in blocked_angles):
                continue

            # Skip bad readings
            if dist <= 0 or math.isinf(dist) or dist > 1.5:
                continue

            # Convert to world coordinates and add to map
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

        # White background
        self.display.setColor(0xFFFFFF)
        self.display.fillRectangle(0, 0, self.display_w, self.display_h)

        # Draw each known cell
        for (gx, gy), occ in self.graph_map.nodes.items():
            rel_x = gx - robot_gx
            rel_y = gy - robot_gy
            px = cx - rel_y
            py = cy - rel_x
            if not (0 <= px < self.display_w and 0 <= py < self.display_h):
                continue
            if occ >= 0.5:
                self.display.setColor(0x000000)  # wall = black
            else:
                gray = int(220 * (1 - occ / 0.5))
                self.display.setColor((gray << 16) | (gray << 8) | gray)
            self.display.drawPixel(px, py)

        # Robot = red dot at centre
        self.display.setColor(0xFF0000)
        self.display.fillRectangle(cx - 2, cy - 2, 4, 4)

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

    # ----------------------------------------------------------
    # NAVIGATION MODE: EXPLORE  (A* + frontier, press M to go TELEOP)
    # ----------------------------------------------------------
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

        # Pick best region using professor's utility formula
        target = self._choose_target(regions, robot_gx, robot_gy)
        if target is None:
            return

        # Plan path with A*
        path = a_star((robot_gx, robot_gy), target, self.graph_map)
        if path is None:
            self.unreachable.add(target)
        else:
            self.current_path = path
            self.target_grid  = target

    def _get_frontiers(self):
        """Free cells that have at least one unknown neighbour."""
        DIRS = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
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

    def _choose_target(self, regions, robot_gx, robot_gy):
        """
        Score each region using professor's utility formula:
            u = alpha * D - beta * T
            D = Euclidean distance from start (rewards deep exploration)
            T = Manhattan distance from robot  (penalises long travel)
        Pick nearest cell of best-scoring region.
        """
        ALPHA = 1.0  # depth weight
        BETA  = 0.5  # travel cost weight

        best_score  = -float('inf')
        best_region = None

        for region in regions:
            # Skip fully unreachable regions
            if all(cell in self.unreachable for cell in region):
                continue

            score = 0.0
            for gx, gy in region:
                D = math.dist((gx, gy), self.start_grid)
                T = abs(gx - robot_gx) + abs(gy - robot_gy)
                score += ALPHA * D - BETA * T

            if score > best_score:
                best_score  = score
                best_region = region

        if best_region is None:
            return None

        # Return the cell in the best region nearest to the robot
        return min(best_region,
                   key=lambda c: abs(c[0] - robot_gx) + abs(c[1] - robot_gy))

    def _follow_path(self, rx, ry, ryaw):
        """Drive toward the next waypoint in the path."""
        if not self.current_path:
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

            # 1. Camera → detect moving object columns
            img_raw      = self.camera.getImageArray()
            img          = np.array(img_raw, dtype=np.uint8).transpose(1, 0, 2)
            blocked_cols = get_blocked_columns(self.prev_frame, img, self.camera.getWidth())
            self.prev_frame = img

            # Union of last _BLOCK_HISTORY frames so a stationary ball stays blocked
            self._blocked_buffer.append(blocked_cols)
            if len(self._blocked_buffer) > self._BLOCK_HISTORY:
                self._blocked_buffer.pop(0)
            effective_blocked = set().union(*self._blocked_buffer)

            # 2. Update map (camera-gated LiDAR) then decay stale cells
            self.update_map(effective_blocked)
            self.graph_map.decay()

            # 3. Draw map every 5 steps
            if step % 5 == 0:
                self.draw_map()

            # 4. Navigation
            if   self.mode == "TELEOP":  self.handle_teleop()
            elif self.mode == "AUTO":    self.handle_auto()
            elif self.mode == "EXPLORE": self.handle_explore()


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    MODE = "AUTO"   # Change to "EXPLORE" to enable A* frontier navigation
    MyRobot(mode=MODE).run()
