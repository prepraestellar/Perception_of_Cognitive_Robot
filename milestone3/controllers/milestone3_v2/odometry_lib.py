"""
Odometry, pose estimation, and Extended Kalman Filter (EKF) fusion.
"""
import math
import numpy as np


class OdometryEKF:
    """
    Handles wheel encoder integration with IMU/compass heading correction.
    Supports raw sensor fusion or EKF-based estimation.
    """

    # Noise parameters
    _ENC_NOISE_STD = 0.002  # metres — Gaussian noise on each wheel displacement
    _IMU_NOISE_STD = 0.02   # radians (~1.1°) — Gaussian noise on heading sensor
    _HEADING_BIAS_RW_STD = 0.0008  # radians/step random-walk drift
    _HEADING_BIAS_MAX = 0.10  # clamp long-term drift (about 5.7 deg)

    # EKF tuning (variance terms)
    _KF_Q_XY_BASE = 2e-4
    _KF_Q_XY_SCALE = 4e-3
    _KF_Q_TH_BASE = 4e-4
    _KF_Q_TH_SCALE = 4e-3
    _KF_R_HEADING_MIN = 1e-4

    def __init__(self, compass, imu=None, wheel_radius=0.033, axle_length=0.160):
        """
        Args:
            compass: Compass sensor object
            imu: IMU sensor object (optional)
            wheel_radius: Robot wheel radius in metres
            axle_length: Distance between wheels in metres
        """
        self.compass = compass
        self.imu = imu
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length

        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left_enc = 0.0
        self.prev_right_enc = 0.0

        # Persistent sensor imperfections
        self._enc_scale_left = 1.0 + np.random.normal(0.0, 0.002)
        self._enc_scale_right = 1.0 + np.random.normal(0.0, 0.002)
        self._heading_bias = 0.0

        # EKF state [x, y, theta] and covariance
        self._ekf_state = np.array([self.x, self.y, self.theta], dtype=np.float64)
        self._ekf_cov = np.diag([1e-3, 1e-3, 2e-3]).astype(np.float64)

        # IMU calibration
        self._imu_yaw_offset = 0.0
        self._imu_calibrated = False

    @staticmethod
    def _wrap_angle(angle):
        """Normalize angle to [-π, π]."""
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

            raw_heading = self._wrap_angle(imu_yaw + self._imu_yaw_offset)
        else:
            cv = self.compass.getValues()
            raw_heading = self._wrap_angle(math.atan2(cv[0], cv[1]))

        # Slow heading drift (magnetic/inertial bias) with bounded random walk
        self._heading_bias += np.random.normal(0.0, self._HEADING_BIAS_RW_STD)
        self._heading_bias = float(np.clip(self._heading_bias, -self._HEADING_BIAS_MAX, self._HEADING_BIAS_MAX))
        return self._wrap_angle(raw_heading + self._heading_bias)

    def update_odometry(self, left_enc, right_enc, use_ekf=True):
        """
        Update pose from wheel encoders and heading sensor.
        
        Args:
            left_enc: Left wheel encoder value
            right_enc: Right wheel encoder value
            use_ekf: If True, use EKF fusion; if False, use raw sensor integration
        """
        d_left = (left_enc - self.prev_left_enc) * self.wheel_radius
        d_right = (right_enc - self.prev_right_enc) * self.wheel_radius
        self.prev_left_enc = left_enc
        self.prev_right_enc = right_enc

        # Per-wheel calibration mismatch (systematic scale error)
        d_left *= self._enc_scale_left
        d_right *= self._enc_scale_right

        # Encoder noise
        d_left += np.random.normal(0.0, self._ENC_NOISE_STD)
        d_right += np.random.normal(0.0, self._ENC_NOISE_STD)

        d_center = (d_right + d_left) / 2.0
        d_theta = (d_right - d_left) / self.axle_length

        if use_ekf:
            self._update_ekf(d_center, d_theta)
        else:
            self._update_raw(d_center, d_theta)

    def _update_ekf(self, d_center, d_theta):
        """Extended Kalman Filter update."""
        # 1) Prediction from wheel encoders (odometry model)
        x, y, theta = self._ekf_state
        theta_mid = theta + 0.5 * d_theta

        x_pred = x + d_center * math.cos(theta_mid)
        y_pred = y + d_center * math.sin(theta_mid)
        theta_pred = self._wrap_angle(theta + d_theta)
        state_pred = np.array([x_pred, y_pred, theta_pred], dtype=np.float64)

        F = np.array([
            [1.0, 0.0, -d_center * math.sin(theta_mid)],
            [0.0, 1.0, d_center * math.cos(theta_mid)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        q_xy = self._KF_Q_XY_BASE + abs(d_center) * self._KF_Q_XY_SCALE
        q_th = self._KF_Q_TH_BASE + abs(d_theta) * self._KF_Q_TH_SCALE
        Q = np.diag([q_xy, q_xy, q_th])

        cov_pred = F @ self._ekf_cov @ F.T + Q

        # 2) Correction from IMU/compass heading
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

    def _update_raw(self, d_center, d_theta):
        """Raw sensor integration without fusion."""
        theta_mid = self.theta + 0.5 * d_theta

        self.x += d_center * math.cos(theta_mid)
        self.y += d_center * math.sin(theta_mid)
        self.theta = self._wrap_angle(self.theta + d_theta)

        # Trust heading sensor directly (with noise but no fusion)
        heading_meas = self._read_heading_measurement()
        heading_meas = self._wrap_angle(heading_meas + np.random.normal(0.0, self._IMU_NOISE_STD))
        self.theta = heading_meas

    def get_pose(self):
        """Return (x, y, -theta) tuple."""
        return self.x, self.y, -self.theta
