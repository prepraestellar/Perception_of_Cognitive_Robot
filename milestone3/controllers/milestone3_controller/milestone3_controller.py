###############
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
from controller import Supervisor

# ==========================================================
# PART 1: PERCEPTION SYSTEM (From Milestone 1)
# ==========================================================

def compute_gradients(gray):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    return gx, gy

class PerceptionSystem:
    def __init__(self):
        self.prev_gray = None
        self.threshold_motion = 6.0

    def process_frame(self, frame_rgb):
        # convert to Gray scale
        gray = np.dot(frame_rgb[..., :3], [0.299, 0.587, 0.114])
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # check for motion (Temporal Difference)
        diff = np.abs(gray - self.prev_gray)
        motion_mask = diff > self.threshold_motion
        
        # find Blobs (in this case, using OpenCV or Simple Connected Components)
        # for simplicity in this example, we will process the motion_mask directly
        self.prev_gray = gray
        return motion_mask

# ==========================================================
# PART 2: GRAPH-BASED MAP (Milestone 3 Requirement #1)
# ==========================================================

class GraphMap:
    def __init__(self, resolution=0.05):
        # use Dictionary 
        # Key: (grid_x, grid_y), Value: occupancy probability
        self.nodes = {} 
        self.resolution = resolution

    def add_to_map(self, world_x, world_y, proba_increase):
        # convert to Grid Coordinates (Quantization)
        gx = int(math.floor(world_x / self.resolution))
        gy = int(math.floor(world_y / self.resolution))
        pos = (gx, gy)

        # update the occupancy probability
        current_val = self.nodes.get(pos, 0.0)
        self.nodes[pos] = min(1.0, current_val + proba_increase)

    def get_value(self, wx, wy):
        gx = int(math.floor(wx / self.resolution))
        gy = int(math.floor(wy / self.resolution))
        return self.nodes.get((gx, gy), 0.0)

# ==========================================================
# PART 3: ROBOT CONTROLLER (Sensor Fusion & Exploration)
# ==========================================================

class MyRobot:
    def __init__(self, bot):
        self.supervisor = bot
        self.time_step = int(self.supervisor.getBasicTimeStep())

        self.lidar = self.supervisor.getDevice('LDS-01')
        self.lidar.enable(self.time_step)
        
        self.camera = self.supervisor.getDevice('camera') 
        if self.camera:
            self.camera.enable(self.time_step)
            self.cam_width = self.camera.getWidth()
            self.cam_fov = self.camera.getFov()

        self.left_motor = self.supervisor.getDevice('left wheel motor')
        self.right_motor = self.supervisor.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.graph_map = GraphMap(resolution=0.05)
        self.perception = PerceptionSystem()
        
        # Perfect Odometry (Supervisor)
        self.robot_node = self.supervisor.getSelf()

    def get_robot_pose(self):
        pos = self.robot_node.getField("translation").getSFVec3f()
        rot = self.robot_node.getField("rotation").getSFRotation()
        yaw = rot[3] if rot[2] > 0 else -rot[3]
        return pos[0], pos[1], yaw

    def is_point_moving(self, ray_angle, motion_mask):
        """ 
        Milestone 3 Requirement #2: Combine Image Processing with LiDAR 
        ตรวจสอบว่ามุมของ LiDAR ลำนี้ ตรงกับพิกัดภาพที่กำลังเคลื่อนที่หรือไม่
        """
        if motion_mask is None or self.camera is None:
            return False
            
        # แปลงมุมของ Ray เป็นพิกัด X บนภาพ (Simplified Projection)
        # กล้องหน้ามองเห็นในช่วง FOV ของมัน
        if abs(ray_angle) > (self.cam_fov / 2):
            return False # อยู่นอกมุมมองกล้อง ให้ถือว่าไม่ขยับ (หรือใช้ข้อมูลเก่า)

        # หาตำแหน่ง Pixel X ที่สัมพันธ์กับมุม
        pixel_x = int(((-ray_angle / self.cam_fov) + 0.5) * self.cam_width)
        pixel_x = max(0, min(self.cam_width - 1, pixel_x))
        
        # เช็คใน motion_mask ว่าคอลัมน์นี้มีการขยับไหม
        return np.any(motion_mask[:, pixel_x])

    def mapping_step(self):
        # 1. รับข้อมูลภาพและตรวจจับการเคลื่อนที่ (M1)
        image = self.camera.getImageArray()
        motion_mask = self.perception.process_frame(np.array(image))

        # 2. รับข้อมูล LiDAR (M2)
        lidar_values = self.lidar.getRangeImage()
        fov = self.lidar.getFov()
        rx, ry, ryaw = self.get_robot_pose()

        # 3. Sensor Fusion & Graph Update (M3)
        for i, dist in enumerate(lidar_values):
            if dist <= 0 or np.isinf(dist) or dist > 3.5: continue
            
            # คำนวณมุมของ Ray ลำนี้
            ray_angle = -(fov / 2) + (i / (len(lidar_values) - 1)) * fov
            
            # --- FILTERING LOGIC ---
            if self.is_point_moving(ray_angle, motion_mask):
                # ถ้ากล้องบอกว่าตรงนี้ขยับ -> ข้ามการบันทึก (Discard)
                continue 
            
            # --- ADD TO GRAPH ---
            world_x = rx + dist * math.cos(ryaw + ray_angle)
            world_y = ry + dist * math.sin(ryaw + ray_angle)
            self.graph_map.add_to_map(world_x, world_y, 0.1)

    def get_frontiers(self):
        """ ค้นหาจุด Frontier (จุดว่างที่ติดกับพื้นที่ Unknown) """
        frontiers = []
        # วนลูปเช็ค Nodes ที่เป็นที่ว่างใน Dictionary
        for (gx, gy) in self.graph_map.nodes:
            if self.graph_map.nodes[(gx, gy)] < 0.3: # เป็นที่ว่าง
                # เช็คเพื่อนบ้าน 4 ทิศ
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    neighbor = (gx + dx, gy + dy)
                    if neighbor not in self.graph_map.nodes:
                        # นี่คือ Frontier! เพราะติดกับ Unknown
                        frontiers.append(neighbor)
        return list(set(frontiers))

    def calculate_utility(self, frontier, rx, ry):
        """ คำนวณความคุ้มค่า u = InformationGain - Distance """
        fx = frontier[0] * self.graph_map.resolution
        fy = frontier[1] * self.graph_map.resolution
        
        dist = math.sqrt((fx - rx)**2 + (fy - ry)**2)
        
        # สมมติ Information Gain คงที่ หรือคำนวณจากจำนวน Unknown รอบๆ
        info_gain = 1.0 
        
        # u = Gain - (Weight * Distance)
        utility = info_gain - (0.2 * dist)
        return utility

    def run(self):
        while self.supervisor.step(self.time_step) != -1:
            # 1. สร้างแผนที่และกรองวัตถุเคลื่อนที่
            self.mapping_step()
            
            # 2. หา Frontier และตัดสินใจ (Exploration)
            rx, ry, _ = self.get_robot_pose()
            frontiers = self.get_frontiers()
            
            if frontiers:
                # เลือกเป้าหมายที่มี Utility สูงสุด
                best_goal = max(frontiers, key=lambda f: self.calculate_utility(f, rx, ry))
                # ในที่นี้คุณสามารถนำ best_goal ไปเข้า A* ต่อได้เลยครับ
                
            # ขยับหุ่นยนต์ (ตัวอย่างเดินหน้าอย่างเดียว)
            self.left_motor.setVelocity(2.0)
            self.right_motor.setVelocity(2.0)

if __name__ == "__main__":
    supervisor = Supervisor()
    robot = MyRobot(supervisor)
    robot.run()