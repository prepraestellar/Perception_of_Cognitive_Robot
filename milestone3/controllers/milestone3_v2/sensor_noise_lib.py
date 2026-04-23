"""
Sensor noise models for LiDAR and camera.
Simulates real-world sensor imperfections.
"""
import math
import numpy as np


class LiDARNoise:
    """LiDAR noise model (approximate real-world behavior)."""

    _NOISE_STD_BASE = 0.004  # metres
    _NOISE_STD_PER_M = 0.010  # metres per metre of distance
    _DROPOUT_BASE = 0.003  # baseline chance of invalid return
    _DROPOUT_PER_M = 0.030  # extra dropout toward max range
    _OUTLIER_PROB = 0.008  # occasional short-return outlier
    _MIN_RANGE = 0.02

    def __init__(self, max_range=3.5):
        self.max_range = max_range

    def _sample_noisy_distance(self, dist):
        """Apply noise to a single LiDAR distance measurement."""
        if dist <= 0.0 or math.isinf(dist):
            return float('inf')

        dist = min(dist, self.max_range)

        # Real sensors get noisier with distance
        sigma = self._NOISE_STD_BASE + self._NOISE_STD_PER_M * dist
        noisy = dist + np.random.normal(0.0, sigma)

        # Missed detections are more likely farther away
        dist_ratio = dist / max(self.max_range, 1e-6)
        p_dropout = self._DROPOUT_BASE + self._DROPOUT_PER_M * dist_ratio
        if np.random.random() < p_dropout:
            return float('inf')

        # Rare multipath/specular short return
        if np.random.random() < self._OUTLIER_PROB:
            noisy *= np.random.uniform(0.55, 0.85)

        return min(self.max_range, max(self._MIN_RANGE, noisy))

    def apply_noise(self, ranges):
        """
        Apply noise to all LiDAR range measurements.
        
        Args:
            ranges: List of distance values
        
        Returns:
            List of noisy distances
        """
        return [self._sample_noisy_distance(d) for d in ranges]


class CameraNoise:
    """Camera noise model including read noise, shot noise, and artifacts."""

    _READ_NOISE_STD = 3.0  # pixel DN (0..255)
    _SHOT_NOISE_SCALE = 0.020  # scales intensity-dependent shot noise
    _GAIN_JITTER_STD = 0.015  # frame-to-frame exposure/gain jitter
    _DEAD_PIXEL_PROB = 2e-4  # sparse stuck-dark pixels
    _FRAME_DROPOUT_PROB = 0.002  # rare dim frame

    @staticmethod
    def apply_noise(frame_bgra):
        """
        Apply camera noise to BGRA frame while preserving alpha channel.
        
        Args:
            frame_bgra: numpy array of shape (H, W, 4) with BGRA data
        
        Returns:
            Noisy BGRA frame as uint8
        """
        noisy = frame_bgra.copy().astype(np.float32)
        bgr = noisy[:, :, :3]

        # Global gain jitter simulates exposure/auto-gain variation
        gain = np.random.normal(1.0, CameraNoise._GAIN_JITTER_STD)
        bgr *= np.clip(gain, 0.85, 1.15)

        # Combine read noise and intensity-dependent shot noise
        shot_sigma = np.sqrt(np.maximum(bgr, 1.0)) * CameraNoise._SHOT_NOISE_SCALE
        bgr += np.random.normal(0.0, CameraNoise._READ_NOISE_STD, bgr.shape)
        bgr += np.random.normal(0.0, 1.0, bgr.shape) * shot_sigma

        # Rare low-illumination/dropout frame
        if np.random.random() < CameraNoise._FRAME_DROPOUT_PROB:
            bgr *= np.random.uniform(0.08, 0.25)

        # Sparse dead pixels
        h, w = bgr.shape[:2]
        dead_mask = np.random.random((h, w)) < CameraNoise._DEAD_PIXEL_PROB
        bgr[dead_mask] = 0.0

        noisy[:, :, :3] = np.clip(bgr, 0.0, 255.0)
        return noisy.astype(np.uint8)
