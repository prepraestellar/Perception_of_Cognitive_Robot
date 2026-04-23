"""
Mapping functionality: camera-gated LiDAR integration with occupancy decay.
"""
import math


class Mapper:
    """
    Updates occupancy map from LiDAR readings gated by camera motion detection.
    Only maps LiDAR rays within camera FOV, skipping rays blocked by detected motion.
    """

    def __init__(self, graph_map):
        """
        Args:
            graph_map: GraphMap instance
        """
        self.graph_map = graph_map

    def reset(self):
        """Reset map state before rebuilding from stored keyframes."""
        return

    def update_map(self, blocked_cols, lidar_ranges, rx, ry, ryaw, camera_hfov, cam_w, lidar_fov):
        """
        Update occupancy map from camera-gated LiDAR data.
        
        Args:
            blocked_cols: Set of camera columns with detected motion
            lidar_ranges: List of LiDAR distance measurements
            rx, ry: Robot position in world coordinates
            ryaw: Robot heading in world coordinates
            camera_hfov: Camera horizontal FOV in radians
            cam_w: Camera image width in pixels
            lidar_fov: LiDAR FOV in radians
        """
        n = len(lidar_ranges)
        for i, dist in enumerate(lidar_ranges):
            # Angle of ray relative to robot forward
            ray_angle = -lidar_fov / 2 + (i / (n - 1)) * lidar_fov

            # Only process rays the camera can see
            if abs(ray_angle) > camera_hfov / 2:
                continue

            # Skip bad readings
            if dist <= 0 or math.isinf(dist) or dist > 1.5:
                continue

            # Map ray angle to camera pixel column
            px = (0.5 - ray_angle / camera_hfov) * cam_w

            # Skip if column belongs to detected moving object (±1 px tolerance)
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
                self.graph_map.nodes[key] = 0.0  # free

            # Mark wall cell at ray endpoint
            wx = rx + dist * math.cos(ryaw + ray_angle)
            wy = ry - dist * math.sin(ryaw + ray_angle)
            wall_key = self.graph_map.world_to_grid(wx, wy)
            self.graph_map.nodes[wall_key] = 1.0  # wall
