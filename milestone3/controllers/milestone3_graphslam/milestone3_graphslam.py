"""
Milestone 3: GraphSLAM robot controller with pose-graph optimization.

Main controller orchestrating:
- Pose-graph SLAM with loop closure detection
- Sensor fusion (encoders + IMU + LiDAR)
- Occupancy grid mapping
- TELEOP and AUTO navigation modes
- Real-time map visualization
"""
import numpy as np
from controller import Supervisor

# Import modular libraries
from graph_map_lib import GraphMap
from sensor_lib import SensorReader
from mapping_lib import Mapper
from display_lib import DisplayManager
from navigation_lib import Navigator
from graphslam import GraphSLAM
import math


class MyRobot:
    """Robot controller with GraphSLAM."""

    def __init__(self, mode="AUTO", optimize_enabled=True):
        """
        Args:
            mode: "TELEOP" or "AUTO"
            optimize_enabled: Whether to enable pose-graph optimization
        """
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())
        self.mode = mode
        self.optimize_enabled = optimize_enabled

        # Lidar
        self.lidar = self.supervisor.getDevice('LDS-01')
        self.lidar.enable(self.time_step)
        self._lidar_max_range = self.lidar.getMaxRange() if hasattr(self.lidar, 'getMaxRange') else 3.5

        # Camera
        self.camera = self.supervisor.getDevice('camera')
        self.camera.enable(self.time_step)
        self.camera_w = self.camera.getWidth()
        self.camera_h = self.camera.getHeight()

        # Keyboard
        self.keyboard = self.supervisor.getKeyboard()
        self.keyboard.enable(self.time_step)

        # Motors
        self.left_motor = self.supervisor.getDevice('left wheel motor')
        self.right_motor = self.supervisor.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Motor Encoders
        self.left_encoder = self.supervisor.getDevice('left wheel sensor')
        self.right_encoder = self.supervisor.getDevice('right wheel sensor')
        self.left_encoder.enable(self.time_step)
        self.right_encoder.enable(self.time_step)

        # Compass and IMU
        self.compass = self.supervisor.getDevice('compass')
        self.compass.enable(self.time_step)

        self.imu = self.supervisor.getDevice('inertial unit')
        if self.imu is not None:
            self.imu.enable(self.time_step)

        # Displays
        display = self.supervisor.getDevice('display')
        global_display = self.supervisor.getDevice('global_display')
        motion_display = self.supervisor.getDevice('motion_display')

        # ===== SUBSYSTEMS =====
        # Map and pathfinding
        self.graph_map = GraphMap(resolution=0.05)
        self.mapper = Mapper(self.graph_map)

        # Raw sensor reader (encoders + heading with noise, no fusion)
        self.sensor_reader = SensorReader(
            compass=self.compass,
            imu=self.imu,
            left_encoder=self.left_encoder,
            right_encoder=self.right_encoder,
            lidar=self.lidar,
            wheel_radius=0.033,
            axle_length=0.160
        )
        
        # Pose tracking (integrated from raw sensor readings)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_scan = np.array([])
        self.keyframes = [{"pose": np.array([0.0, 0.0, 0.0]), "scan": None}]
        
        # Keyframe thresholds
        self._optimization_threshold = 0.5  # metres
        self._rotation_threshold = 0.5      # radians
        self.last_keyframe_pose = np.array([0.0, 0.0, 0.0])
        self.keyframe_poses = [np.array([0.0, 0.0, 0.0])]
        self.keyframe_count = 1

        # Navigation
        self.navigator = Navigator(self.graph_map, self.left_motor, self.right_motor, self.keyboard)

        # Display
        self.display_manager = DisplayManager(display, global_display, motion_display)

        # Set starting grid for EXPLORE mode
        start_grid = self.graph_map.world_to_grid(self.x, self.y)
        self.navigator.set_start_grid(start_grid)

        # GraphSLAM backend
        self.graph_slam = GraphSLAM()
        
        # Add first pose to graph as reference
        self.graph_slam.add_pose(self.x, self.y, self.theta)

    
    def update_pose(self, d_left, d_right, heading):
        """Integrate wheel distance deltas and heading into pose estimate."""
        d_center = (d_left + d_right) / 2.0
        self.theta = heading
        self.x += d_center * math.cos(self.theta)
        self.y += d_center * math.sin(self.theta)
    
    def get_pose(self):
        """Return current pose (x, y, -theta) tuple."""
        return self.x, self.y, -self.theta

    @staticmethod
    def _wrap_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _relative_pose(self, from_pose, to_pose):
        """Return to_pose expressed as a relative motion in from_pose's frame."""
        dx_world = to_pose[0] - from_pose[0]
        dy_world = to_pose[1] - from_pose[1]
        from_theta = from_pose[2]

        cos_theta = math.cos(from_theta)
        sin_theta = math.sin(from_theta)
        dx_local = cos_theta * dx_world + sin_theta * dy_world
        dy_local = -sin_theta * dx_world + cos_theta * dy_world
        dtheta = self._wrap_angle(to_pose[2] - from_theta)
        return dx_local, dy_local, dtheta

    def _rebuild_map_from_keyframes(self):
        """Rebuild occupancy grid from stored keyframe scans and optimized poses."""
        self.graph_map.nodes.clear()
        self.mapper.reset()

        limit = min(len(self.keyframes), len(self.graph_slam.poses))
        for idx in range(1, limit):
            scan = self.keyframes[idx]["scan"]
            if scan is None:
                continue

            pose = self.graph_slam.poses[idx]
            self.mapper.update_map(
                set(),
                scan,
                pose[0],
                pose[1],
                -pose[2],
                self.camera.getFov(),
                self.camera_w,
                self.lidar.getFov(),
            )

        print(f"[MAP] Rebuilt from {max(0, limit - 1)} keyframe scans")



    def run(self):
        """Main control loop."""
        print(f"Milestone 3 GraphSLAM started — mode: {self.mode}")
        print(f"Controls: W/A/S/D = move, M = toggle mode, O = toggle optimization")
        step = 0
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        while self.supervisor.step(self.time_step) != -1:
            step += 1

            # 1. SENSOR ACQUISITION
            left_enc, right_enc = self.sensor_reader.read_encoders()
            heading = self.sensor_reader.read_imu()
            lidar_scan = self.sensor_reader.read_lidar()
            
            # 1a. ODOMETRY INTEGRATION
            self.update_pose(left_enc, right_enc, heading)
            self.current_pose = np.array([self.x, self.y, self.theta])

            # 2. KEYFRAME DECISION
            d_distance = np.sqrt((self.x - self.last_keyframe_pose[0])**2 + 
                                (self.y - self.last_keyframe_pose[1])**2)
            d_rotation = abs(self.theta - self.last_keyframe_pose[2])
            
            is_keyframe = (
                d_distance > self._optimization_threshold or
                d_rotation > self._rotation_threshold
            )
            
            if is_keyframe:
                self.keyframes.append({
                    "pose": np.array([self.x, self.y, self.theta]),
                    "scan": list(lidar_scan),
                })
                self.keyframe_poses.append(np.array([self.x, self.y, self.theta]))
                self.keyframe_count += 1
                print(
                    f"[SLAM] Keyframe {self.keyframe_count} at step {step}: "
                    f"({self.x:.2f}, {self.y:.2f}, {self.theta:.2f})"
                )

                #  3. ADD NODE ----
                node_id = len(self.graph_slam.poses)
                self.graph_slam.add_pose(self.x, self.y, self.theta)
                
                #  4. ADD ODOM EDGE ----
                if node_id > 0:
                    dx, dy, dtheta = self._relative_pose(
                        self.last_keyframe_pose,
                        np.array([self.x, self.y, self.theta]),
                    )
                    info_odom = np.diag([5.0, 1.0, 10.0])
                    self.graph_slam.add_odometry_edge(
                        node_id - 1, node_id, dx, dy, dtheta, info_odom
                    )
                
                # ---- 5. LOOP CLOSURE DETECTION (check last 10 poses) ----
                for prev_id in range(node_id):
                    prev_pose = self.graph_slam.poses[prev_id]
                    dist_to_prev = np.sqrt((self.x - prev_pose[0])**2 + 
                                          (self.y - prev_pose[1])**2)
                    
                    if dist_to_prev < 0.3 and prev_id != node_id - 1:
                        dx_loop, dy_loop, dtheta_loop = self._relative_pose(
                            prev_pose,
                            np.array([self.x, self.y, self.theta]),
                        )
                        info_loop = np.diag([10.0, 10.0, 10.0])
                        self.graph_slam.add_loop_closure_edge(
                            prev_id, node_id, dx_loop, dy_loop, dtheta_loop, info_loop
                        )
                
                # ---- 6. BACKEND OPTIMIZATION ----
                if self.optimize_enabled and self.keyframe_count % 5 == 0 and len(self.graph_slam.poses) > 1:
                    print(
                        f"[SLAM] Optimizing graph at step {step}: "
                        f"poses={len(self.graph_slam.poses)}, edges={len(self.graph_slam.edges)}"
                    )
                    result = self.graph_slam.optimize(verbose=True)
                    if result is not None:
                        print(
                            f"[SLAM] Optimization done: success={result.success}, "
                            f"cost={result.cost:.4f}, nfev={result.nfev}"
                        )
                        self._rebuild_map_from_keyframes()

                else:
                    self._rebuild_map_from_keyframes()
                
                # Update keyframe reference
                self.last_keyframe_pose = np.array([self.x, self.y, self.theta])
            
            # Store current scan for next iteration
            self.prev_scan = lidar_scan
            
            # ---- 8. VISUALIZATION ----
            if step % 5 == 0:
                frontier_set = set()
                rx, ry, ryaw = self.get_pose()
                self.display_manager.draw_local_map(self.graph_map, rx, ry, frontier_set)
                self.display_manager.draw_slam_nodes_local(self.graph_map, self.keyframe_poses, rx, ry)
                self.display_manager.draw_global_map(self.graph_map, rx, ry, frontier_set)
                self.display_manager.draw_slam_nodes_global(self.graph_map, self.keyframe_poses)
            
            # ---- 9. LOGGING ----
            if step % 50 == 0:
                n_poses = len(self.graph_slam.poses)
                n_edges = len(self.graph_slam.edges)
                n_loop_edges = sum(1 for e in self.graph_slam.edges if e.edge_type == "loop_closure")
                print(f"[Step {step}] Poses: {n_poses}, Edges: {n_edges} (loop: {n_loop_edges}), "
                      f"Pose: ({self.x:.2f}, {self.y:.2f}, {self.theta:.2f})")
            
            # ---- 10. NAVIGATION ----
            if self.mode == "TELEOP":
                key = self.navigator.handle_teleop()
                if key == ord('O'):
                    self.optimize_enabled = not self.optimize_enabled
                    print(f"Optimization {'enabled' if self.optimize_enabled else 'disabled'}")
                if key == ord('M'):
                    self.mode = "AUTO"
                    print("Switched to AUTO mode")
            
            elif self.mode == "AUTO":
                if len(lidar_scan) > 0:
                    key = self.navigator.handle_auto(lidar_scan)
                else:
                    key = self.navigator.handle_auto([0.0] * 360)

                if key == ord('O'):
                    self.optimize_enabled = not self.optimize_enabled
                    print(f"Optimization {'enabled' if self.optimize_enabled else 'disabled'}")
                
                if key == ord('M'):
                    self.mode = "TELEOP"
                    print("Switched to TELEOP mode")



# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    MODE = "AUTO"  # Change to "TELEOP" as needed
    OPTIMIZE_ENABLED = True  # Change to False to disable optimization
    MyRobot(mode=MODE, optimize_enabled=OPTIMIZE_ENABLED).run()

