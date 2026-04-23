"""
Milestone 3 v2: Modular robot controller with sensor fusion, mapping, and autonomous exploration.

Main controller orchestrating:
- Odometry with EKF fusion
- Sensor noise simulation
- Camera-gated LiDAR mapping
- Three navigation modes: TELEOP, AUTO, EXPLORE
- Real-time visualization
"""
import numpy as np
from controller import Supervisor
from motion_detection import get_blocked_columns

# Import modular libraries
from graph_map_lib import GraphMap, a_star
from odometry_lib import OdometryEKF
from sensor_noise_lib import LiDARNoise, CameraNoise
from mapping_lib import Mapper
from display_lib import DisplayManager
from navigation_lib import Navigator


class MyRobot:
    """Main robot controller."""

    def __init__(self, mode="AUTO", use_kalman=True):
        """
        Args:
            mode: "TELEOP", "AUTO", or "EXPLORE"
            use_kalman: Enable EKF sensor fusion
        """
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())
        self.mode = mode
        self.use_kalman = use_kalman

        # ===== DEVICES =====
        self.lidar = self.supervisor.getDevice('LDS-01')
        self.lidar.enable(self.time_step)
        self._lidar_max_range = self.lidar.getMaxRange() if hasattr(self.lidar, 'getMaxRange') else 3.5

        self.camera = self.supervisor.getDevice('camera')
        self.camera.enable(self.time_step)
        self.camera_w = self.camera.getWidth()
        self.camera_h = self.camera.getHeight()

        self.keyboard = self.supervisor.getKeyboard()
        self.keyboard.enable(self.time_step)

        self.left_motor = self.supervisor.getDevice('left wheel motor')
        self.right_motor = self.supervisor.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_encoder = self.supervisor.getDevice('left wheel sensor')
        self.right_encoder = self.supervisor.getDevice('right wheel sensor')
        self.left_encoder.enable(self.time_step)
        self.right_encoder.enable(self.time_step)

        self.compass = self.supervisor.getDevice('compass')
        self.compass.enable(self.time_step)

        self.imu = self.supervisor.getDevice('inertial unit')
        if self.imu is not None:
            self.imu.enable(self.time_step)

        display = self.supervisor.getDevice('display')
        global_display = self.supervisor.getDevice('global_display')
        motion_display = self.supervisor.getDevice('motion_display')

        # ===== SUBSYSTEMS =====
        # Map and pathfinding
        self.graph_map = GraphMap(resolution=0.05)
        self.mapper = Mapper(self.graph_map)

        # Odometry with optional EKF
        self.odometry = OdometryEKF(
            compass=self.compass,
            imu=self.imu,
            wheel_radius=0.033,
            axle_length=0.160
        )

        # Sensor noise
        self.lidar_noise = LiDARNoise(max_range=self._lidar_max_range)

        # Navigation
        self.navigator = Navigator(self.graph_map, self.left_motor, self.right_motor, self.keyboard)

        # Display
        self.display_manager = DisplayManager(display, global_display, motion_display)

        # Set starting grid for EXPLORE mode
        start_grid = self.graph_map.world_to_grid(self.odometry.x, self.odometry.y)
        self.navigator.set_start_grid(start_grid)

        # Motion detection buffer
        self.prev_frame = None
        self.prev_pose = None
        self._blocked_buffer = []
        self._BLOCK_HISTORY = 4

    def update_odometry(self):
        """Update robot pose from encoders and heading sensor."""
        left_enc = self.left_encoder.getValue()
        right_enc = self.right_encoder.getValue()
        self.odometry.update_odometry(left_enc, right_enc, use_ekf=self.use_kalman)

    def run(self):
        """Main control loop."""
        print(f"Milestone 3 v2 started — mode: {self.mode}")
        kalman_str = "ON (EKF fusion)" if self.use_kalman else "OFF (trust sensors)"
        print(f"Controls: W/A/S/D = move, M = toggle mode, K = toggle Kalman ({kalman_str})")
        print("Modules: graph_map_lib, odometry_lib, sensor_noise_lib, mapping_lib, display_lib, navigation_lib")
        step = 0

        while self.supervisor.step(self.time_step) != -1:
            step += 1

            # ===== 1. ODOMETRY =====
            self.update_odometry()

            # ===== 2. SENSORS =====
            img_raw = self.camera.getImage()
            if img_raw is None:
                continue

            img = np.frombuffer(img_raw, dtype=np.uint8).reshape((self.camera_h, self.camera_w, 4))
            img = CameraNoise.apply_noise(img)

            pose_now = self.odometry.get_pose()
            raw_lidar_ranges = self.lidar.getRangeImage()
            noisy_lidar_ranges = self.lidar_noise.apply_noise(raw_lidar_ranges)

            # ===== 3. MOTION DETECTION =====
            blocked_cols = get_blocked_columns(
                self.prev_frame, img,
                self.camera_w,
                pose1=self.prev_pose,
                pose2=pose_now,
                cam_hfov=self.camera.getFov(),
                lidar_ranges=noisy_lidar_ranges,
                lidar_fov=self.lidar.getFov()
            )
            self.prev_pose = pose_now
            self.prev_frame = img.copy()

            # Buffer motion detections
            self._blocked_buffer.append(blocked_cols)
            if len(self._blocked_buffer) > self._BLOCK_HISTORY:
                self._blocked_buffer.pop(0)
            effective_blocked = set().union(*self._blocked_buffer)

            if step % 30 == 0:
                print(f"[DBG] step={step}  blocked={len(effective_blocked)} cols")

            # ===== 4. VISUALIZATION =====
            self.display_manager.draw_motion_display(img, blocked_cols, effective_blocked, self.camera_w)

            # ===== 5. MAPPING =====
            rx, ry, ryaw = self.odometry.get_pose()
            self.mapper.update_map(
                effective_blocked, noisy_lidar_ranges,
                rx, ry, ryaw,
                self.camera.getFov(),
                self.camera_w,
                self.lidar.getFov()
            )

            # ===== 6. DISPLAY MAPS =====
            if step % 5 == 0:
                frontier_set = set(self.navigator._get_frontiers())
                self.display_manager.draw_local_map(self.graph_map, rx, ry, frontier_set)
                self.display_manager.draw_global_map(self.graph_map, rx, ry, frontier_set)

            # ===== 7. NAVIGATION =====
            if self.mode == "TELEOP":
                key = self.navigator.handle_teleop()
                if key == ord('M'):
                    self.mode = "AUTO"
                    print("Switched to AUTO mode")
                elif key == ord('K'):
                    self.use_kalman = not self.use_kalman
                    mode_str = "ON (EKF fusion)" if self.use_kalman else "OFF (trust sensors)"
                    print(f"Kalman filter toggled {mode_str}")

            elif self.mode == "AUTO":
                key = self.navigator.handle_auto(noisy_lidar_ranges)
                if key == ord('M'):
                    self.mode = "TELEOP"
                    print("Switched to TELEOP mode")
                elif key == ord('K'):
                    self.use_kalman = not self.use_kalman
                    mode_str = "ON (EKF fusion)" if self.use_kalman else "OFF (trust sensors)"
                    print(f"Kalman filter toggled {mode_str}")

            elif self.mode == "EXPLORE":
                key = self.navigator.handle_explore(rx, ry, ryaw)
                if key == ord('M'):
                    self.mode = "TELEOP"
                    print("Switched to TELEOP mode")
                elif key == ord('K'):
                    self.use_kalman = not self.use_kalman
                    mode_str = "ON (EKF fusion)" if self.use_kalman else "OFF (trust sensors)"
                    print(f"Kalman filter toggled {mode_str}")
# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    MODE = "AUTO"  # Change to "EXPLORE" or "TELEOP" as needed
    MyRobot(mode=MODE, use_kalman=False).run()

