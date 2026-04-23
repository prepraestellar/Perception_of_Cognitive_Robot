"""
Raw sensor readers for wheel encoders and IMU/compass with noise.
No pose estimation — just sensor readings with realistic noise.
"""
import math
import numpy as np


class SensorReader:
    """
    Raw sensor reader: encoders and heading (compass/IMU) with noise.
    Returns noisy measurements only, no pose integration.
    """

    # Noise parameters
    _ENC_NOISE_STD = 0.004  # metres — Gaussian noise on each wheel displacement
    _HEADING_NOISE_STD = 0.03   # radians (~1.7°) — Gaussian noise on heading sensor
    _HEADING_BIAS_RW_STD = 0.001  # radians/step — slow drifting heading bias
    _SLIP_STD = 0.003  # unitless — small per-step wheel slip factor

    def __init__(self, compass, imu=None, left_encoder=None, right_encoder=None, lidar=None, wheel_radius=0.033, axle_length=0.160):
        """
        Args:
            compass: Compass sensor object
            imu: IMU sensor object (optional)
            left_encoder: Left wheel encoder sensor object (optional)
            right_encoder: Right wheel encoder sensor object (optional)
            lidar: LiDAR sensor object (optional)
            wheel_radius: Robot wheel radius in metres
            axle_length: Distance between wheels in metres
        """
        self.compass = compass
        self.imu = imu
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.lidar = lidar
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length

        # IMU calibration (offset between compass and IMU frames)
        self._imu_yaw_offset = 0.0
        self._imu_calibrated = False
        self._heading_bias = 0.0
        
        # Encoder state tracking
        self.prev_left_enc = 0.0
        self.prev_right_enc = 0.0
        self._enc_initialized = False
        
        # Per-wheel calibration (systematic scale error)
        self._enc_scale_left = 1.0 + np.random.normal(0.0, 0.002)
        self._enc_scale_right = 1.0 + np.random.normal(0.0, 0.002)

    @staticmethod
    def _wrap_angle(angle):
        """Normalize angle to [-π, π]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def read_imu(self):
        """
        Read heading from compass/IMU with noise.
        
        Returns:
            Heading angle in radians (normalized to [-π, π])
        """
        if self.imu is not None and hasattr(self.imu, 'getRollPitchYaw'):
            rpy = self.imu.getRollPitchYaw()
            imu_yaw = self._wrap_angle(-rpy[2])

            if not self._imu_calibrated:
                cv = self.compass.getValues()
                compass_yaw = self._wrap_angle(math.atan2(cv[0], cv[1]))
                self._imu_yaw_offset = self._wrap_angle(compass_yaw - imu_yaw)
                self._imu_calibrated = True

            raw_heading = self._wrap_angle(imu_yaw + self._imu_yaw_offset)
        else:
            cv = self.compass.getValues()
            raw_heading = self._wrap_angle(math.atan2(cv[0], cv[1]))

        # Add measurement noise plus a slowly drifting bias to mimic real IMU/compass behavior.
        self._heading_bias += np.random.normal(0.0, self._HEADING_BIAS_RW_STD)
        self._heading_bias = float(np.clip(self._heading_bias, -0.15, 0.15))
        heading_with_noise = self._wrap_angle(
            raw_heading + self._heading_bias + np.random.normal(0.0, self._HEADING_NOISE_STD)
        )
        return heading_with_noise

    def read_encoders(self):
        """
        Read wheel encoder deltas and return as distance displacement with noise.
        
        Returns:
            (d_left, d_right) tuple — distance traveled by each wheel in metres
                with calibration, noise, and delta computation already applied
        """
        # Read actual raw encoder values from sensors
        if self.left_encoder is not None and self.right_encoder is not None:
            left_raw = self.left_encoder.getValue()
            right_raw = self.right_encoder.getValue()
        else:
            # Fallback if encoders not provided
            return 0.0, 0.0

        # First sample is only used to initialize encoder baseline.
        if not self._enc_initialized:
            self.prev_left_enc = left_raw
            self.prev_right_enc = right_raw
            self._enc_initialized = True
            return 0.0, 0.0
        
        # Compute raw delta (encoder ticks/units)
        d_left_raw = (left_raw - self.prev_left_enc)
        d_right_raw = (right_raw - self.prev_right_enc)
        
        # Store for next iteration
        self.prev_left_enc = left_raw
        self.prev_right_enc = right_raw
        
        # Convert to distance (metres) and apply per-wheel calibration
        d_left = d_left_raw * self.wheel_radius * self._enc_scale_left
        d_right = d_right_raw * self.wheel_radius * self._enc_scale_right

        # Add small correlated slip variation so odom drifts gradually over time.
        slip_left = 1.0 + np.random.normal(0.0, self._SLIP_STD)
        slip_right = 1.0 + np.random.normal(0.0, self._SLIP_STD)
        d_left *= slip_left
        d_right *= slip_right
        
        # Add measurement noise (Gaussian, independent per wheel)
        d_left += np.random.normal(0.0, self._ENC_NOISE_STD)
        d_right += np.random.normal(0.0, self._ENC_NOISE_STD)
        
        return d_left, d_right

    def read_lidar(self):
        """
        Read LiDAR range measurements with realistic noise.
        Applies distance-dependent noise, dropouts, and outliers matching real hardware.
        
        Returns:
            List of noisy range measurements (clipped to device max range)
        """
        # Get device parameters
        max_range = self.lidar.getMaxRange() if hasattr(self.lidar, 'getMaxRange') else 3.5
        fov = self.lidar.getFov() if hasattr(self.lidar, 'getFov') else 2 * np.pi
        
        # Get raw range image
        raw_ranges = self.lidar.getRangeImage()
        
        if raw_ranges is None or len(raw_ranges) == 0:
            return []
        
        # Apply realistic LiDAR noise (distance-dependent, dropouts, outliers)
        noisy_ranges = []
        for dist in raw_ranges:
            noisy_dist = self._apply_lidar_noise(dist, max_range)
            noisy_ranges.append(noisy_dist)
        
        return noisy_ranges
    
    def _apply_lidar_noise(self, dist, max_range=3.5):
        """
        Apply realistic LiDAR noise to a single measurement.
        Includes Gaussian noise, dropouts (longer range = higher chance), and rare outliers.
        
        Args:
            dist: Raw distance measurement
            max_range: Maximum sensor range
        
        Returns:
            Noisy distance (may be inf if dropout occurred)
        """
        # Noise parameters (tuned for realistic LiDAR behavior)
        NOISE_STD_BASE = 0.004
        NOISE_STD_PER_M = 0.010
        DROPOUT_BASE = 0.003
        DROPOUT_PER_M = 0.030
        OUTLIER_PROB = 0.008
        MIN_RANGE = 0.02
        
        # Invalid measurement
        if dist <= 0.0 or math.isinf(dist):
            return float('inf')
        
        # Clip to max range
        dist = min(dist, max_range)
        
        # Distance-dependent noise (farther = noisier)
        sigma = NOISE_STD_BASE + NOISE_STD_PER_M * dist
        noisy = dist + np.random.normal(0.0, sigma)
        
        # Dropout (missed detection) increases with distance
        dist_ratio = dist / max(max_range, 1e-6)
        p_dropout = DROPOUT_BASE + DROPOUT_PER_M * dist_ratio
        if np.random.random() < p_dropout:
            return float('inf')
        
        # Rare multipath/specular reflection (short-return outlier)
        if np.random.random() < OUTLIER_PROB:
            noisy *= np.random.uniform(0.55, 0.85)
        
        # Clip to valid range
        return min(max_range, max(MIN_RANGE, noisy))