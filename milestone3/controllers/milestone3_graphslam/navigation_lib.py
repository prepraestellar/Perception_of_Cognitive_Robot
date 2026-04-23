"""
Navigation modes: TELEOP, AUTO, and EXPLORE with frontier-based A* planning.
"""
import math
from collections import deque
from graph_map_lib import a_star


class Navigator:
    """Handles different navigation modes and path following."""

    MAX_SPEED = 6.28

    def __init__(self, graph_map, left_motor, right_motor, keyboard):
        """
        Args:
            graph_map: GraphMap instance
            left_motor: Left wheel motor device
            right_motor: Right wheel motor device
            keyboard: Keyboard input device
        """
        self.graph_map = graph_map
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.keyboard = keyboard

        # EXPLORE mode state
        self.current_path = []
        self.target_grid = None
        self.start_grid = None

        # AUTO mode: committed turn to prevent oscillation
        self._avoid_timer = 0
        self._avoid_dir = 1  # +1 = turn left, -1 = turn right

    def set_start_grid(self, start_grid):
        """Set starting grid cell for frontier scoring."""
        self.start_grid = start_grid

    def handle_teleop(self):
        """Handle TELEOP mode: W/A/S/D keyboard control."""
        key = self.keyboard.getKey()
        left, right = 0.0, 0.0

        if key == ord('W'):
            left = right = self.MAX_SPEED
        elif key == ord('S'):
            left = right = -self.MAX_SPEED
        elif key == ord('A'):
            left, right = -self.MAX_SPEED * 0.4, self.MAX_SPEED * 0.4
        elif key == ord('D'):
            left, right = self.MAX_SPEED * 0.4, -self.MAX_SPEED * 0.4

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)
        return key  # Return key for mode switching

    def handle_auto(self, ranges):
        """
        Handle AUTO mode: random walk with obstacle avoidance.
        
        Args:
            ranges: Full 360° LiDAR range data
        
        Returns:
            Key pressed (for mode switching)
        """
        n = len(ranges)

        front = min(ranges[n // 2 - 20 : n // 2 + 20])
        front_right = min(ranges[n // 2 + 20 : n // 2 + 80])
        front_left = min(ranges[n // 2 - 80 : n // 2 - 20])

        if self._avoid_timer > 0:
            self._avoid_timer -= 1
            spd = self.MAX_SPEED * 0.4
            self.left_motor.setVelocity(-self._avoid_dir * spd)
            self.right_motor.setVelocity(self._avoid_dir * spd)
        elif front < 0.25:
            self._avoid_dir = 1 if front_left >= front_right else -1
            self._avoid_timer = 25
            spd = self.MAX_SPEED * 0.4
            self.left_motor.setVelocity(-self._avoid_dir * spd)
            self.right_motor.setVelocity(self._avoid_dir * spd)
        elif front_right < 0.20:
            self.left_motor.setVelocity(self.MAX_SPEED * 0.3)
            self.right_motor.setVelocity(self.MAX_SPEED * 0.8)
        elif front_left < 0.20:
            self.left_motor.setVelocity(self.MAX_SPEED * 0.8)
            self.right_motor.setVelocity(self.MAX_SPEED * 0.3)
        else:
            self.left_motor.setVelocity(self.MAX_SPEED)
            self.right_motor.setVelocity(self.MAX_SPEED)

        return self.keyboard.getKey()

    def handle_explore(self, rx, ry, ryaw):
        """
        Handle EXPLORE mode: frontier-based autonomous exploration.
        
        Args:
            rx, ry, ryaw: Current robot pose
        
        Returns:
            Key pressed (for mode switching)
        """
        key = self.keyboard.getKey()
        if key != -1:
            return key

        robot_gx, robot_gy = self.graph_map.world_to_grid(rx, ry)

        # Follow existing path if we have one
        if self.current_path:
            self._follow_path(rx, ry, ryaw)
            return key

        # Find frontier cells
        frontiers = self._get_frontiers()
        if not frontiers:
            # Exploration complete — spin slowly
            self.left_motor.setVelocity(1.0)
            self.right_motor.setVelocity(-1.0)
            return key

        # Group frontiers and choose target
        regions = self._get_frontier_regions(frontiers)
        target = self._choose_target(regions, robot_gx, robot_gy)
        if target is None:
            self.left_motor.setVelocity(self.MAX_SPEED * 0.3)
            self.right_motor.setVelocity(-self.MAX_SPEED * 0.3)
            return key

        # Plan path with A*
        path = a_star((robot_gx, robot_gy), target, self.graph_map)
        if not path:
            self.current_path = [target]
            self.target_grid = target
        else:
            self.current_path = path
            self.target_grid = target

        return key

    def _get_frontiers(self):
        """Find frontier cells: free cells with unknown neighbours."""
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
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
        """Group frontier cells into connected clusters."""
        frontier_set = set(frontiers)
        visited = set()
        regions = []
        DIRS4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for cell in frontiers:
            if cell in visited:
                continue
            region = []
            queue = [cell]
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
        """Compute BFS distances from start to all reachable free cells."""
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dist = {start: 0}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in DIRS:
                nb = (cx + dx, cy + dy)
                if nb not in dist and self.graph_map.is_occupied(nb[0], nb[1]) is False:
                    dist[nb] = dist[(cx, cy)] + 1
                    q.append(nb)
        return dist

    def _choose_target(self, regions, robot_gx, robot_gy):
        """Choose frontier region target using utility function."""
        ALPHA = 1.0  # Discovery reward
        BETA = 0.5   # Travel cost penalty

        dist_map = self._bfs_distances((robot_gx, robot_gy))
        best_score = -float('inf')
        best_target = None

        for region in regions:
            reachable = [(gx, gy) for gx, gy in region if (gx, gy) in dist_map]
            if not reachable:
                continue

            T = min(dist_map[(gx, gy)] for gx, gy in reachable)
            D = sum(math.dist((gx, gy), self.start_grid) for gx, gy in region)
            score = ALPHA * D - BETA * T

            if score > best_score:
                best_score = score
                best_target = min(reachable, key=lambda c: dist_map[c])

        return best_target

    def _follow_path(self, rx, ry, ryaw):
        """Drive toward next waypoint in path."""
        if not self.current_path:
            return

        gx, gy = self.current_path[0]
        if self.graph_map.is_occupied(gx, gy) is True:
            self.current_path = []  # waypoint became wall — replan
            return

        wx, wy = self.graph_map.grid_to_world(gx, gy)
        dx, dy = wx - rx, wy - ry
        dist = math.hypot(dx, dy)

        if dist < 0.08:
            self.current_path.pop(0)
            return

        target_yaw = math.atan2(-dy, dx)
        yaw_error = math.atan2(math.sin(target_yaw - ryaw), math.cos(target_yaw - ryaw))

        if abs(yaw_error) > 0.15:
            turn = self.MAX_SPEED * 0.4 * (1 if yaw_error > 0 else -1)
            self.left_motor.setVelocity(-turn)
            self.right_motor.setVelocity(turn)
        else:
            self.left_motor.setVelocity(self.MAX_SPEED * 0.6)
            self.right_motor.setVelocity(self.MAX_SPEED * 0.6)
