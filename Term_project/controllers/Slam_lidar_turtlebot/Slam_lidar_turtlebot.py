import numpy as np
from controller import Supervisor
from tqdm import tqdm
import math

class Map:
    def __init__(self, size: tuple, world_size_m: tuple):
        self.map = np.zeros(size)
        self.size = size
        self.world_size_m = world_size_m

    def add_to_map(self, item: int, position: tuple[int, int]):
        if 0 <= position[0] < self.size[0] and 0 <= position[1] < self.size[1]:
            self.map[tuple(position)] = item 
        # else:
        #     print(f"Warning: Position {position} is out of map bounds and will be ignored.")

    def print_map(self):
        for row in self.map:
            print(' '.join(str(int(cell)) for cell in row))
        print('==================================')

class Object: # undone
    def __init__(self, ideal: bool):
        self.ideal = ideal
    def is_wall(self) -> bool:
        if self.ideal:
            # assume that if the object is ideal, it is a wall for this example
            return True
        else:
            # In a real implementation, this would involve more complex logic to determine if the object is a wall
            return False 
        
class MyRobot:
    def __init__(self, wheel_radius=0.033, axle_length=0.16, bot=None, supervisor=None):
        self.robot = bot
        self.supervisor = supervisor
        self.time_step = int(self.supervisor.getBasicTimeStep())
        # assume perfect odom
        self.robot_node = self.supervisor.getSelf()
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")


        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.wheel_radius = wheel_radius
        self.axle_length = axle_length
        self.dir_dict = {
            'n':  0,        # no rotation
            's':  180,       # rotate 180 (direction doesn't really matter)
            'e':  -90,       # rotate right 90
            'w':  90,       # rotate left 90
            'nw': 45,   # rotate 45 left
            'ne': -45,   # rotate 45 right
            'sw': 135,   # rotate 135 left
            'se': -135    # rotate 135 right
        }

        # Lidar sensors
        self.lidar = self.robot.getDevice('LDS-01')
        self.lidar.enable(self.time_step)
        self.lidar.enablePointCloud()

        self.display = self.robot.getDevice('display')
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.time_step)

    def setmap(self, size: tuple, world_size_m: tuple) -> Map:
        self.map = Map(size, world_size_m)
        return self.map

    def draw_map(self):
        # 1. Safety check
        if self.display is None:
            return 
            
        dw = self.display.getWidth()
        dh = self.display.getHeight()
        
        # 2. Get the actual shape of your NumPy array directly
        rows, cols = self.map.map.shape 
        
        # Calculate how many display pixels each array cell represents
        cell_w = dw / cols
        cell_h = dh / rows

        # 3. Clear the display with a white background first (faster than drawing empty cells)
        self.display.setColor(0xFFFFFF) 
        self.display.fillRectangle(0, 0, dw, dh)

        # 4. Use the NumPy array as the reference to draw walls
        self.display.setColor(0x000000) # Black for walls
        
        for row in range(rows):
            for col in range(cols):
                # Read directly from the numpy array
                if self.map.map[row, col] == 1:
                    # Translate NumPy [row, col] to Display [x, y]
                    x_pixel = int(col * cell_w)
                    y_pixel = int(row * cell_h)
                    
                    self.display.fillRectangle(x_pixel, y_pixel, int(cell_w), int(cell_h))

        # 5. Draw the robot position (Red)
        self.display.setColor(0xFF0000)
        
        # Get robot coordinates (assuming your convert function returns [row, col])
        rx, ry = self.convert_to_map_coordinates(self.get_current_position())
        
        # Make sure the robot is inside the numpy array bounds
        if 0 <= rx < rows and 0 <= ry < cols:
            robot_x = int(ry * cell_w) # Map ry (col) to x_pixel
            robot_y = int(rx * cell_h) # Map rx (row) to y_pixel
            self.display.fillRectangle(robot_x, robot_y, int(cell_w), int(cell_h))

    def inverse_kinematic(self, angle_deg, time):
        angle = np.deg2rad(angle_deg)   # convert to radians
        omega = angle / time            # robot angular velocity
        vr = (omega * self.axle_length) / (2 * self.wheel_radius)
        vl = -vr
        return vr, vl
    
    def random_direction(self):
        return np.random.choice(['s', 'e', 'w', 'nw', 'ne', 'sw', 'se'])
    
    def mapping(self, iteration, verbose=100, teleop=False):
        if teleop:
            for _ in tqdm(range(iteration)):
                if self.supervisor.step(self.time_step) == -1:
                    break

                # Read all queued keys and keep the latest one for this step
                key = -1
                k = self.keyboard.getKey()
                while k != -1:
                    key = k
                    k = self.keyboard.getKey() # Updated inside the loop to avoid infinite loop

                left_speed = 0.0
                right_speed = 0.0
                speed = 6.28

                # Fixed indentation for the movement logic
                if key in (ord('W'), ord('w')):
                    left_speed = speed
                    right_speed = speed
                elif key in (ord('S'), ord('s')):
                    left_speed = -speed
                    right_speed = -speed
                elif key in (ord('A'), ord('a')):
                    left_speed = -speed
                    right_speed = speed
                elif key in (ord('D'), ord('d')):
                    left_speed = speed
                    right_speed = -speed

                self.left_motor.setVelocity(left_speed)
                self.right_motor.setVelocity(right_speed)

                # Fixed indentation for the mapping logic
                if key in (ord('S'), ord('s')) or key in (ord('W'), ord('w')) :
                    # If turning, we might want to skip mapping to avoid noisy data
                    self.read_distance_sensors()
                    wall_positions = self.get_wall_position()
                    for pos in wall_positions:
                        self.map.add_to_map(1, self.convert_to_map_coordinates(pos))
                    
                        self.draw_map()

            # Stop motors after the iterations are complete
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            
            return self.map
        else:
            for i in tqdm(range(iteration)):
                if self.supervisor.step(self.time_step) == -1:
                    break

                # 1. Read sensors first
                self.read_distance_sensors()
                Obstacle = self.check_obstacle()
                
                if Obstacle:
                    # --- TURNING STATE (NO MAPPING) ---
                    direction = self.random_direction()  
                    angle_deg = self.dir_dict[direction]
                    
                    vr, vl = self.inverse_kinematic(angle_deg, 1.5)
                    self.left_motor.setVelocity(vl)
                    self.right_motor.setVelocity(vr)
                    
                    # Execute the turn
                    steps_to_turn = int(1500 / self.time_step)
                    for _ in range(steps_to_turn):
                        self.supervisor.step(self.time_step)
                        
                    # Stop for just a few frames to let the robot's physics settle 
                    # before taking the next map reading.
                    self.left_motor.setVelocity(0.0)
                    self.right_motor.setVelocity(0.0)
                    for _ in range(100):
                        self.supervisor.step(self.time_step)
                        
                else:
                    # --- DRIVING STATE (MAPPING ENABLED) ---
                    self.left_motor.setVelocity(6.28)
                    self.right_motor.setVelocity(6.28)
                    
                    # Only calculate walls and map when moving straight
                    wall_positions = self.get_wall_position()
                    for pos in wall_positions:
                        self.map.add_to_map(1, self.convert_to_map_coordinates(pos))
                        
                    self.draw_map()

            # End of iterations
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
            return self.map

    def read_distance_sensors(self):
        # Read the 682 distance points from the LiDAR (returns values in exact meters)
        self.lidar_values = self.lidar.getRangeImage()

    def check_obstacle(self) -> bool:
        # Assuming the front of the robot is the center of the lidar_values array
        # Check only a narrow cone in front of the robot
        num_points = len(self.lidar_values)
        front_cone = self.lidar_values[num_points//2 - 50 : num_points//2 + 50] 
        return any(0 < v < 0.15 for v in front_cone if not np.isinf(v))

    def get_current_position(self) -> tuple[float, float]:
        position = self.trans_field.getSFVec3f()
        return (float(position[0]), float(position[1]))
    
    def get_wall_position(self) -> list[tuple[float, float]]:
        if not hasattr(self, 'lidar_values') or self.lidar_values is None:
            return []
    
        wall_positions = []
        robot_position = self.get_current_position() # Returns (World X, World Y)
        
        # Get robot yaw (rotation around the vertical Y-axis)
        rot = self.rot_field.getSFRotation()  # [ax, ay, az, angle]
        robot_yaw = -rot[3] if rot[2] > 0 else rot[3]  

        
        fov = self.lidar.getFov()
        print(f"LiDAR FOV: {np.rad2deg(fov):.2f} degrees")
        num_points = len(self.lidar_values)
        
        for i, distance in enumerate(self.lidar_values):
            # Only map valid hits (ignore infinity, zero, or sensor errors)
            if distance > 0 and not np.isinf(distance) and not np.isnan(distance):
                
                # 1. Find the angle of this specific ray relative to the robot
                ray_angle = -(fov / 2) + (i / (num_points - 1)) * fov
                global_angle = robot_yaw + ray_angle
                
                wall_x = robot_position[0] + distance * np.cos(global_angle)
                wall_y = robot_position[1] - distance * np.sin(global_angle) 
                
                # Append as (X, Y) to match your convert_to_map_coordinates function
                wall_positions.append((wall_x, wall_y))
                
        return wall_positions

    def convert_to_map_coordinates(self, position: tuple[float, float]) -> tuple[int, int]:
        scale_x = self.map.size[0] / self.map.world_size_m[0]
        scale_y = self.map.size[1] / self.map.world_size_m[1]
        map_x = int(math.floor(position[0] * scale_x)) + self.map.size[0] // 2
        map_y = int(math.floor(position[1] * scale_y)) + self.map.size[1] // 2
        return (map_x, map_y)    
  

if __name__ == "__main__":
    supervisor = Supervisor()
    turtle_bot = MyRobot(wheel_radius=0.033, axle_length=0.16, bot=supervisor, supervisor=supervisor)
    turtle_bot.setmap(size=(50, 50), world_size_m=(4.0, 4.0))
    world_map = turtle_bot.mapping(iteration=100000, verbose=0, teleop=True)

    print('done')