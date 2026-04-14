
# step 1: camera checking -> if not moving -> collect as coordinate in dictionary
# step 2: do the node - dictionary: key = coordinate (x,y), value = probability of being a wall
# - if the position of the calculated path is in the dictionary and value < threshold (not the wall), then you can move there -> this is implicitly the edge of the graph
# step 3: frontier detection: coordinate that is not in the dictionary  
# Step 4: Utility Calculation -> it needs to decide which frontier to go to -> calculate the utility of each frontier and choose the one with the highest utility as the goal
# - cost function: utility = information gain - distance cost
#   - distance cost = the distance from the current position to the frontier (the closer, the better)
#   - information gain = the amount of new area that can be explored from that frontier (the more, the better)
# step 5: calculate the algorithm = A* -> find the path from start to frontier for each position you are in 

import numpy as np
import math
import heapq
from controller import Supervisor

# ==========================================================
# PART 1: PERCEPTION SYSTEM
# ==========================================================
class PerceptionSystem:
    def __init__(self, threshold_motion=6.0):
        self.prev_gray = None
        self.threshold_motion = threshold_motion

    def process_frame(self, frame_rgb):
        gray = np.dot(frame_rgb[..., :3], [0.299, 0.587, 0.114])
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        diff = np.abs(gray - self.prev_gray)
        motion_mask = diff > self.threshold_motion
        self.prev_gray = gray
        return motion_mask

# ==========================================================
# PART 2: GRAPH-BASED MAP & A* PATHFINDING
# ==========================================================
class GraphMap:
    def __init__(self, resolution=0.05):
        self.nodes = {} 
        self.resolution = resolution

    def world_to_grid(self, wx, wy):
        gx = int(math.floor(wx / self.resolution))
        gy = int(math.floor(wy / self.resolution))
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.resolution + (self.resolution / 2)
        wy = gy * self.resolution + (self.resolution / 2)
        return (wx, wy)

    def add_to_map(self, wx, wy, proba_increase):
        pos = self.world_to_grid(wx, wy)
        current_val = self.nodes.get(pos, 0.0)
        self.nodes[pos] = min(1.0, current_val + proba_increase)

    def is_occupied(self, gx, gy, threshold=0.5):
        val = self.nodes.get((gx, gy))
        if val is None: return None
        return val >= threshold

def a_star(start_grid, goal_grid, graph_map, wall_threshold=0.5, max_iterations=1500):
    def get_neighbors(node):
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (node[0] + dx, node[1] + dy)
            if graph_map.is_occupied(neighbor[0], neighbor[1], wall_threshold) is not True:
                neighbors.append(neighbor)
        return neighbors

    open_set = []
    heapq.heappush(open_set, (0, start_grid))
    came_from = {}
    g_score = {start_grid: 0}
    iterations = 0
    while open_set:
        iterations += 1
        if iterations > 5: 
            return None
            
        _, current = heapq.heappop(open_set)
        
        if current == goal_grid:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + math.dist(current, neighbor)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + math.dist(neighbor, goal_grid)
                heapq.heappush(open_set, (f_score, neighbor))
        
    return None

# ==========================================================
# PART 3: ROBOT CONTROLLER
# ==========================================================
class MyRobot:
    def __init__(self, bot):
        self.supervisor = bot
        self.time_step = int(self.supervisor.getBasicTimeStep())

        self.lidar = self.supervisor.getDevice('LDS-01')
        self.lidar.enable(self.time_step)
        
        self.camera = self.supervisor.getDevice('camera')
        self.camera.enable(self.time_step)
        self.cam_width = self.camera.getWidth()
        self.cam_fov = self.camera.getFov()

        self.display = self.supervisor.getDevice('display')
        if self.display:
            self.display_width = self.display.getWidth()
            self.display_height = self.display.getHeight()

        self.left_motor = self.supervisor.getDevice('left wheel motor')
        self.right_motor = self.supervisor.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.graph_map = GraphMap(resolution=0.05)
        self.perception = PerceptionSystem()
        self.robot_node = self.supervisor.getSelf()
        
        self.current_path = []
        self.unreachable_frontiers = set() # บัญชีดำ
        self.target_grid = None # เก็บเป้าหมายไว้ใช้วาดบนจอ

    def get_pose(self):
        pos = self.robot_node.getField("translation").getSFVec3f()
        rot = self.robot_node.getField("rotation").getSFRotation()
        yaw = -rot[3] if rot[2] > 0 else rot[3]
        return pos[0], pos[1], yaw

    def filter_moving_lidar(self, ray_angle, motion_mask):
        if motion_mask is None: return False
        if abs(ray_angle) < (self.cam_fov / 2):
            pixel_x = int((0.5 - (ray_angle / self.cam_fov)) * self.cam_width)
            pixel_x = np.clip(pixel_x, 0, self.cam_width - 1)
            if np.any(motion_mask[:, pixel_x]): return True 
        return False

    def mapping(self):
        img_raw = self.camera.getImageArray()
        motion_mask = self.perception.process_frame(np.array(img_raw))
        lidar_ranges = self.lidar.getRangeImage()
        fov = self.lidar.getFov()
        rx, ry, ryaw = self.get_pose()

        num_points = len(lidar_ranges)
        for i, dist in enumerate(lidar_ranges):
            if dist <= 0 or np.isinf(dist) or dist > 3.0: continue
            ray_angle = -(fov / 2) + (i / (num_points - 1)) * fov
            if self.filter_moving_lidar(ray_angle, motion_mask): continue

            wx = rx + dist * math.cos(ryaw + ray_angle)
            wy = ry - dist * math.sin(ryaw + ray_angle) 
            self.graph_map.add_to_map(wx, wy, 0.1)

    def draw_map(self):
        if not self.display: return
        
        # 1. เทพื้นหลังสีขาว
        self.display.setColor(0xFFFFFF)
        self.display.fillRectangle(0, 0, self.display_width, self.display_height)

        center_x = self.display_width // 2
        center_y = self.display_height // 2
        pixels_per_cell = 2

        # 2. ดึงพิกัดหุ่นยนต์มาเป็น "ศูนย์กลางหน้าจอ" (แก้ปัญหาเดินตกขอบจอ)
        rx, ry, _ = self.get_pose()
        robot_gx, robot_gy = self.graph_map.world_to_grid(rx, ry)

        # 3. วาดกำแพง (สีดำ)
        self.display.setColor(0x000000)
        for (gx, gy), prob in self.graph_map.nodes.items():
            if prob >= 0.5:
                # คำนวณระยะห่างของกำแพงเทียบกับตัวหุ่นยนต์ (ทำให้แผนที่เลื่อนตามหุ่น)
                rel_x = gx - robot_gx
                rel_y = gy - robot_gy

                # 4. หมุนแกนหน้าจอ 90 องศาให้ตรงกับ 3D View (สลับแกน X, Y)
                px = center_x - (rel_y * pixels_per_cell) 
                py = center_y - (rel_x * pixels_per_cell)
                
                # วาดเฉพาะส่วนที่อยู่ในกรอบจอภาพ
                if 0 <= px < self.display_width and 0 <= py < self.display_height:
                    self.display.fillRectangle(int(px), int(py), pixels_per_cell, pixels_per_cell)

        # 5. วาดเป้าหมาย Frontier (สีเขียว) ให้เลื่อนตามด้วย
        if self.target_grid:
            rel_tx = self.target_grid[0] - robot_gx
            rel_ty = self.target_grid[1] - robot_gy
            tx = center_x - (rel_ty * pixels_per_cell)
            ty = center_y - (rel_tx * pixels_per_cell)
            
            if 0 <= tx < self.display_width and 0 <= ty < self.display_height:
                self.display.setColor(0x00FF00)
                self.display.fillRectangle(int(tx)-2, int(ty)-2, 5, 5)

        # 6. วาดหุ่นยนต์ (สีแดง) ให้อยู่ "ตรงกลางจอเสมอ"
        self.display.setColor(0xFF0000)
        self.display.fillRectangle(center_x-2, center_y-2, 4, 4)


    def find_best_frontier(self, rx, ry):
        frontiers = []
        for (gx, gy), val in self.graph_map.nodes.items():
            if val < 0.3:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (gx + dx, gy + dy)
                    if neighbor not in self.graph_map.nodes and neighbor not in self.unreachable_frontiers:
                        frontiers.append(neighbor)
        
        if not frontiers: return None

        best_frontier = None
        max_utility = -float('inf')
        for f in list(set(frontiers)):
            fx, fy = self.graph_map.grid_to_world(f[0], f[1])
            dist = math.dist((rx, ry), (fx, fy))
            utility = 1.0 - (0.5 * dist) 
            if utility > max_utility:
                max_utility = utility
                best_frontier = f
        return best_frontier

    def avoid_deadlock(self):
        print("[ACTION] ถอยหลังเพื่อเปิดมุมมอง...")
        self.left_motor.setVelocity(-2.0)
        self.right_motor.setVelocity(-1.0)
        for _ in range(15):
            if self.supervisor.step(self.time_step) == -1: break

    def move_along_path(self, rx, ry, ryaw):
        if self.current_path:
            target_node = self.current_path[0]
            tx, ty = self.graph_map.grid_to_world(target_node[0], target_node[1])
            dist = math.dist((rx, ry), (tx, ty))
            
            if dist < 0.15: 
                self.current_path.pop(0)
                return 

            angle_to_target = math.atan2(ty - ry, tx - rx)
            angle_diff = angle_to_target - ryaw
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            
            if abs(angle_diff) > 0.3:
                self.left_motor.setVelocity(-1.5 if angle_diff > 0 else 1.5)
                self.right_motor.setVelocity(1.5 if angle_diff > 0 else -1.5)
            else:
                self.left_motor.setVelocity(4.0)
                self.right_motor.setVelocity(4.0)
        else:
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)

    def run(self):
        print("🚀 M3 Smart Exploration Started!")
        recovery_counter = 0 # ตัวนับเพื่อแก้ติดหล่ม
        
        while self.supervisor.step(self.time_step) != -1:
            self.mapping()
            self.draw_map()
            
            rx, ry, ryaw = self.get_pose()
            start_grid = self.graph_map.world_to_grid(rx, ry)

            # --- ส่วนตัดสินใจหลัก ---
            if not self.current_path:
                self.target_grid = self.find_best_frontier(rx, ry)
                if self.target_grid:
                    path = a_star(start_grid, self.target_grid, self.graph_map)
                    if path:
                        self.current_path = path
                        recovery_counter = 0
                    else:
                        # ถ้าหาทางไปจุดที่ใกล้ที่สุดไม่ได้ ให้มาร์คจุดนั้นไว้และถอยหลัง
                        self.unreachable_frontiers.add(self.target_grid)
                        self.avoid_deadlock()
                else:
                    # ถ้าหา Frontier ไม่เจอเลย ให้หมุนตัวหาพื้นที่ใหม่
                    self.avoid_deadlock()

            # --- ส่วนควบคุมการเดิน ---
            if self.current_path:
                self.move_along_path(rx, ry, ryaw)
            else:
                # ถ้ายังนิ่งอยู่ ให้หมุนตัวเบาๆ เพื่อสแกนหาจุดใหม่
                self.left_motor.setVelocity(1.0)
                self.right_motor.setVelocity(-1.0)

if __name__ == "__main__":
    robot = MyRobot(Supervisor())
    robot.run()