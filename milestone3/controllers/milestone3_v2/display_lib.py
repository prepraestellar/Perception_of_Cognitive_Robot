"""
Display and visualization utilities for local map, global map, and motion detection overlay.
"""
import math
import numpy as np
from controller import Display


class DisplayManager:
    """Manages all visualization displays."""

    def __init__(self, display, global_display, motion_display=None):
        """
        Args:
            display: Local occupancy map display device
            global_display: Global map display device
            motion_display: Motion detection overlay display (optional)
        """
        self.display = display
        self.display_w = display.getWidth()
        self.display_h = display.getHeight()

        self.global_display = global_display
        self.gdisplay_w = global_display.getWidth()
        self.gdisplay_h = global_display.getHeight()

        self.motion_display = motion_display
        if motion_display is not None:
            self.mdisplay_w = motion_display.getWidth()
            self.mdisplay_h = motion_display.getHeight()
        else:
            self.mdisplay_w = 0
            self.mdisplay_h = 0

    def draw_local_map(self, graph_map, rx, ry, frontier_set):
        """
        Draw local occupancy map centered on robot.
        
        Args:
            graph_map: GraphMap instance
            rx, ry: Robot position in world coordinates
            frontier_set: Set of frontier grid cells to highlight
        """
        robot_gx, robot_gy = graph_map.world_to_grid(rx, ry)
        cx, cy = self.display_w // 2, self.display_h // 2

        # Light grey background = undiscovered
        self.display.setColor(0xC0C0C0)
        self.display.fillRectangle(0, 0, self.display_w, self.display_h)

        # Draw each known cell
        for (gx, gy), occ in graph_map.nodes.items():
            rel_x = gx - robot_gx
            rel_y = gy - robot_gy
            px = cx - rel_y
            py = cy - rel_x
            if not (0 <= px < self.display_w and 0 <= py < self.display_h):
                continue
            if occ >= 0.5:
                self.display.setColor(0x000000)  # occupied = black
            elif (gx, gy) in frontier_set:
                self.display.setColor(0x00BFFF)  # frontier = light blue
            else:
                self.display.setColor(0xFFFFFF)  # free = white
            self.display.drawPixel(px, py)

        # Robot = red dot at centre
        self.display.setColor(0xFF0000)
        self.display.fillRectangle(cx - 2, cy - 2, 4, 4)

    def draw_global_map(self, graph_map, rx, ry, frontier_set):
        """
        Draw global occupancy map with fixed origin, showing entire explored area.
        
        Args:
            graph_map: GraphMap instance
            rx, ry: Robot position in world coordinates
            frontier_set: Set of frontier grid cells to highlight
        """
        if not graph_map.nodes:
            return

        keys = list(graph_map.nodes.keys())
        min_gx = min(k[0] for k in keys)
        max_gx = max(k[0] for k in keys)
        min_gy = min(k[1] for k in keys)
        max_gy = max(k[1] for k in keys)

        span_x = max(max_gx - min_gx + 1, 1)
        span_y = max(max_gy - min_gy + 1, 1)
        scale = min(self.gdisplay_w / span_y, self.gdisplay_h / span_x)
        scale = max(scale, 1.0)

        # Light grey background = undiscovered
        self.global_display.setColor(0xC0C0C0)
        self.global_display.fillRectangle(0, 0, self.gdisplay_w, self.gdisplay_h)

        sz = max(2, math.ceil(scale))
        for (gx, gy), occ in graph_map.nodes.items():
            px = int((gy - min_gy) * scale)
            py = int((gx - min_gx) * scale)
            if not (0 <= px < self.gdisplay_w and 0 <= py < self.gdisplay_h):
                continue
            if occ >= 0.5:
                self.global_display.setColor(0x000000)  # occupied = black
            elif (gx, gy) in frontier_set:
                self.global_display.setColor(0x00BFFF)  # frontier = light blue
            else:
                self.global_display.setColor(0xFFFFFF)  # free = white
            self.global_display.fillRectangle(px, py, sz, sz)

        # Robot position in red
        rgx, rgy = graph_map.world_to_grid(rx, ry)
        rpx = int((rgy - min_gy) * scale)
        rpy = int((rgx - min_gx) * scale)
        dot = max(sz * 3, 5)
        self.global_display.setColor(0xFF0000)
        self.global_display.fillRectangle(rpx - dot // 2, rpy - dot // 2, dot, dot)

    def draw_motion_display(self, frame_bgra, blocked_cols, effective_blocked, camera_width):
        """
        Draw live camera view with motion detection overlay.
        
        Args:
            frame_bgra: Camera frame as BGRA numpy array
            blocked_cols: Currently detected motion columns
            effective_blocked: Buffered motion columns used for mapping (red)
            camera_width: Original camera image width
        """
        if self.motion_display is None:
            return

        cam_h, cam_w = frame_bgra.shape[:2]
        if self.mdisplay_w == cam_w and self.mdisplay_h == cam_h:
            bgra_bytes = frame_bgra.tobytes()
        else:
            # Nearest-neighbor resize
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

        # Buffered detections used for mapping (red)
        self.motion_display.setColor(0xFF4040)
        for col in effective_blocked:
            x = int(col * self.mdisplay_w / cam_w)
            if 0 <= x < self.mdisplay_w:
                self.motion_display.drawLine(x, 0, x, self.mdisplay_h - 1)

        # Current-frame detections (green, in upper band)
        self.motion_display.setColor(0x00FF66)
        band_h = max(2, self.mdisplay_h // 6)
        for col in blocked_cols:
            x = int(col * self.mdisplay_w / cam_w)
            if 0 <= x < self.mdisplay_w:
                self.motion_display.drawLine(x, 0, x, band_h)

        # Camera center column marker (yellow)
        cx = self.mdisplay_w // 2
        self.motion_display.setColor(0xFFD700)
        self.motion_display.drawLine(cx, 0, cx, self.mdisplay_h - 1)

        # Status text
        self.motion_display.setColor(0x000000)
        self.motion_display.fillRectangle(0, 0, 130, 18)
        self.motion_display.setColor(0xFFFFFF)
        self.motion_display.drawText(
            f"R buf:{len(effective_blocked)} G now:{len(blocked_cols)}",
            2,
            2,
        )
