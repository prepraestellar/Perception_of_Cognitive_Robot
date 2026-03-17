import numpy as np
from controller import Supervisor
import math

class Map:
    def __init__(self, size: tuple, world_size_m: tuple):
        self.map = np.zeros(size)
        self.size = size
        self.world_size_m = world_size_m

    def add_to_map(self, item: float, position: tuple[int, int]):
        if 0 <= position[0] < self.size[0] and 0 <= position[1] < self.size[1]:
            self.map[tuple(position)] = min(1.0, self.map[tuple(position)] + item)

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
    def __init__(self, wheel_radius=0.033, axle_length=0.16, MAX_SPEED=6.28, bot=None, supervisor=None, use_supervisor = True):
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
        self.MAX_SPEED = MAX_SPEED  # Max speed for the motors (1 rotation per second)

        self.use_supervisor = use_supervisor

        # Encoders
        self.left_encoder = self.robot.getDevice('left wheel sensor')
        self.right_encoder = self.robot.getDevice('right wheel sensor')
        self.left_encoder.enable(self.time_step)
        self.right_encoder.enable(self.time_step)

        #Gyro
        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.time_step)

        # Compass
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.time_step)

        # Odom state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left_enc = 0.0
        self.prev_right_enc = 0.0

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

        # Cache for incremental display updates.
        self._display_cache_initialized = False
        self._last_map_binary = None
        self._prev_robot_cell = None
        self._last_display_size = None

    def setmap(self, size: tuple, world_size_m: tuple) -> Map:
        self.map = Map(size, world_size_m)
        self._display_cache_initialized = False
        self._last_map_binary = None
        self._prev_robot_cell = None
        return self.map

    def draw_map(self, threshold=0.5):
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

        # Use at least one pixel per cell to avoid empty draws on small displays.
        px_w = max(1, int(round(cell_w)))
        px_h = max(1, int(round(cell_h)))

        # Rebuild cache if first draw, map shape changed, or display size changed.
        needs_reset = (
            (not self._display_cache_initialized)
            or self._last_map_binary is None
            or self._last_map_binary.shape != self.map.map.shape
            or self._last_display_size != (dw, dh)
        )

        current_binary = self.map.map >= threshold

        if needs_reset:
            self.display.setColor(0xFFFFFF)
            self.display.fillRectangle(0, 0, dw, dh)
            self._last_map_binary = np.zeros_like(current_binary, dtype=bool)
            self._display_cache_initialized = True
            self._prev_robot_cell = None
            self._last_display_size = (dw, dh)

        # Draw only cells whose occupancy changed since last frame.
        changed = current_binary != self._last_map_binary
        changed_rows, changed_cols = np.where(changed)

        for row, col in zip(changed_rows, changed_cols):
            x_pixel = int(col * cell_w)
            y_pixel = int(row * cell_h)
            if current_binary[row, col]:
                self.display.setColor(0x000000)  # Wall
            else:
                self.display.setColor(0xFFFFFF)  # Free cell
            self.display.fillRectangle(x_pixel, y_pixel, px_w, px_h)

        self._last_map_binary = current_binary.copy()

        # Restore the previous robot cell before drawing the new one.
        if self._prev_robot_cell is not None:
            pr, pc = self._prev_robot_cell
            if 0 <= pr < rows and 0 <= pc < cols:
                prev_x = int(pc * cell_w)
                prev_y = int(pr * cell_h)
                if current_binary[pr, pc]:
                    self.display.setColor(0x000000)
                else:
                    self.display.setColor(0xFFFFFF)
                self.display.fillRectangle(prev_x, prev_y, px_w, px_h)

        # Draw the robot position (red).
        rx, ry = self.convert_to_map_coordinates(self.get_current_position())
        
        if 0 <= rx < rows and 0 <= ry < cols:
            robot_x = int(ry * cell_w) # Map ry (col) to x_pixel
            robot_y = int(rx * cell_h) # Map rx (row) to y_pixel
            self.display.setColor(0xFF0000)
            self.display.fillRectangle(robot_x, robot_y, px_w, px_h)
            self._prev_robot_cell = (rx, ry)
        else:
            self._prev_robot_cell = None

    def inverse_kinematic(self, angle_deg, time):
        angle = np.deg2rad(angle_deg)   # convert to radians
        omega = angle / time            # robot angular velocity
        vr = (omega * self.axle_length) / (2 * self.wheel_radius)
        vl = -vr
        return vr, vl
    
    def random_direction(self):
        # return np.random.choice(['s', 'e', 'w', 'nw', 'ne', 'sw', 'se'])
        return np.random.choice(['s', 'sw', 'se'])
    
    def mapping(self, iteration, verbose=100, teleop=False, show_display=True, proba_increase=0.1, threshold=0.5):
        if teleop:
            try:
                # Removed tqdm, added 'i' to track the current loop number
                for i in range(iteration):
                    if self.supervisor.step(self.time_step) == -1:
                        break
                    self.update_odometry()
                    # --- PROGRESS PRINT ---
                    if verbose > 0 and i % verbose == 0:
                        print(f"Teleop Mapping: {i}/{iteration} steps ({(i/iteration)*100:.1f}%)")

                    # Read all queued keys and keep the latest one for this step
                    key = -1
                    k = self.keyboard.getKey()
                    while k != -1:
                        key = k
                        k = self.keyboard.getKey() # Updated inside the loop to avoid infinite loop

                    # --- WEBOTS KEYBOARD INTERRUPT ---
                    # If 'Q' is the last key pressed, stop teleop mapping
                    if key in (ord('Q'), ord('q')):
                        print("\nTeleop mapping stopped manually via Webots (Q pressed). Continuing...")
                        break

                    left_speed = 0.0
                    right_speed = 0.0
                    speed = self.MAX_SPEED

                    # Movement logic
                    if key in (ord('W'), ord('w')):
                        left_speed = speed
                        right_speed = speed
                    elif key in (ord('S'), ord('s')):
                        left_speed = -speed
                        right_speed = -speed
                    elif key in (ord('A'), ord('a')):
                        left_speed = -0.3 * speed
                        right_speed = 0.3 * speed
                    elif key in (ord('D'), ord('d')):
                        left_speed = 0.3 * speed
                        right_speed = -0.3 * speed

                    self.left_motor.setVelocity(left_speed)
                    self.right_motor.setVelocity(right_speed)

                    # Mapping logic
                    self.read_distance_sensors()
                    wall_positions = self.get_wall_position()
                    for pos in wall_positions:
                        self.map.add_to_map(proba_increase, self.convert_to_map_coordinates(pos))
                    
                    if show_display:
                        self.draw_map(threshold=threshold) 

            # --- TERMINAL KEYBOARD INTERRUPT (CTRL+C) ---
            except KeyboardInterrupt:
                print("\nCtrl+C detected in terminal! Stopping teleop mapping early and continuing...")

            # Stop motors after the iterations are complete or interrupted
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            
            return self.map
        else:
            try:
                # Removed tqdm here as well
                for i in range(iteration):
                    if self.supervisor.step(self.time_step) == -1:
                        break
                    self.update_odometry()
                    # --- PROGRESS PRINT ---
                    if verbose > 0 and i % verbose == 0:
                        print(f"Auto Mapping: {i}/{iteration} steps ({(i/iteration)*100:.1f}%)")

                    # --- WEBOTS KEYBOARD INTERRUPT ---
                    k = self.keyboard.getKey()
                    while k != -1:
                        if k in (ord('Q'), ord('q')):
                            print("\nMapping stopped manually via Webots (Q pressed). Continuing...")
                            break 
                        k = self.keyboard.getKey()
                    
                    if k in (ord('Q'), ord('q')):
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
                            if self.supervisor.step(self.time_step) == -1:
                                break
                            self.update_odometry()
                            
                        # Stop for just a few frames to let the robot's physics settle 
                        self.left_motor.setVelocity(0.0)
                        self.right_motor.setVelocity(0.0)
                        for _ in range(5):
                            if self.supervisor.step(self.time_step) == -1:
                                break
                            self.update_odometry()
                    else:
                        # --- DRIVING STATE (MAPPING ENABLED) ---
                        self.left_motor.setVelocity(6.28)
                        self.right_motor.setVelocity(6.28)
                        
                        # Only calculate walls and map when moving straight
                        wall_positions = self.get_wall_position()
                        for pos in wall_positions:
                            self.map.add_to_map(proba_increase, self.convert_to_map_coordinates(pos))
                        
                        if show_display:
                            self.draw_map(threshold=threshold) 

            # --- TERMINAL KEYBOARD INTERRUPT (CTRL+C) ---
            except KeyboardInterrupt:
                print("\nCtrl+C detected in terminal! Stopping mapping early and continuing...")

            # End of iterations or Interrupted
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
        return any(0 < v < 0.3 for v in front_cone if not np.isinf(v))

    def get_current_position(self) -> tuple[float, float]:
        if self.use_supervisor:
            position = self.trans_field.getSFVec3f()
            return (float(position[0]), float(position[1]))
        else:
            return (self.x, self.y)

    def update_odometry(self):
        # Temporary debug — remove after
        # compass_vals = self.compass.getValues()
        # rot = self.rot_field.getSFRotation()
        # supervisor_yaw = -rot[3] if rot[2] > 0 else rot[3]
        # compass_yaw = -math.atan2(compass_vals[0], compass_vals[1])
        # print("-"*50)
        # print(f"compass raw:     {compass_vals}")
        # print(f"rot raw:         {rot}")
        # print(f"supervisor yaw:  {math.degrees(supervisor_yaw):.2f} deg")
        # print(f"compass yaw:     {math.degrees(compass_yaw):.2f} deg")

        # 1. Read encoders
        left_enc  = self.left_encoder.getValue()
        right_enc = self.right_encoder.getValue()

        # 2. Delta since last step → convert to meters
        d_left  = (left_enc  - self.prev_left_enc)  * self.wheel_radius
        d_right = (right_enc - self.prev_right_enc) * self.wheel_radius
        self.prev_left_enc  = left_enc
        self.prev_right_enc = right_enc

        # 3. Get heading from compass (no drift)
        compass_vals = self.compass.getValues()
        self.theta = math.atan2(compass_vals[0], compass_vals[1])

        # 4. Forward displacement
        d_center = (d_right + d_left) / 2.0

        # 5. Integrate position
        self.x += d_center * math.cos(self.theta)
        self.y += d_center * math.sin(self.theta)
    
    def get_wall_position(self) -> list[tuple[float, float]]:
        # if not hasattr(self, 'lidar_values') or self.lidar_values is None:
        #     return []
    
        # wall_positions = []
        # robot_position = self.get_current_position() # Returns (World X, World Y)
        
        # # Get robot yaw (rotation around the vertical Y-axis)
        # rot = self.rot_field.getSFRotation()  # [ax, ay, az, angle]
        # robot_yaw = -rot[3] if rot[2] > 0 else rot[3]  

        
        # fov = self.lidar.getFov()
        # num_points = len(self.lidar_values)
        
        
        # for i, distance in enumerate(self.lidar_values):
        #     # Only map valid hits (ignore infinity, zero, or sensor errors)
        #     if distance > 0 and not np.isinf(distance) and not np.isnan(distance):
                
        #         # 1. Find the angle of this specific ray relative to the robot
        #         ray_angle = -(fov / 2) + (i / (num_points - 1)) * fov
        #         global_angle = robot_yaw + ray_angle
                
        #         wall_x = robot_position[0] + distance * np.cos(global_angle)
        #         wall_y = robot_position[1] - distance * np.sin(global_angle) 
                
        #         # Append as (X, Y) to match your convert_to_map_coordinates function
        #         wall_positions.append((wall_x, wall_y))
        # return wall_positions
        if not hasattr(self, 'lidar_values') or self.lidar_values is None:
            return []
            
        # 1. Convert lidar readings to a numpy array for fast operations
        distances = np.array(self.lidar_values)
        num_points = len(distances)
        if num_points == 0:
            return []

        robot_position = self.get_current_position() # Returns (World X, World Y)
        
        # Get robot yaw with odom / supervisor
        #if self.use_supervisor:
        ##rot = self.rot_field.getSFRotation()
        ##robot_yaw = -rot[3] if rot[2] > 0 else rot[3]
        #else:
        #    robot_yaw = self.theta
        if self.use_supervisor:
            rot = self.rot_field.getSFRotation()
            robot_yaw = -rot[3] if rot[2] > 0 else rot[3]
        else:
            robot_yaw = -self.theta
        
        fov = self.lidar.getFov()
        
        # 2. Create a boolean mask of only the valid laser hits
        valid_mask = (distances > 0) & ~np.isinf(distances) & ~np.isnan(distances)
        
        # 3. Generate all ray angles at once using linspace
        # This replaces: -(fov / 2) + (i / (num_points - 1)) * fov
        all_ray_angles = np.linspace(-fov / 2, fov / 2, num_points)
        
        # 4. Filter the distances and angles using our mask
        valid_distances = distances[valid_mask]
        valid_angles = all_ray_angles[valid_mask]
        
        # 5. Calculate global angles in one go
        global_angles = robot_yaw + valid_angles
        
        # 6. Apply trigonometry to the entire array at once
        wall_x = robot_position[0] + valid_distances * np.cos(global_angles)
        wall_y = robot_position[1] - valid_distances * np.sin(global_angles) 
        
        # 7. Zip the X and Y arrays back into a list of tuples to match your existing code
        return list(zip(wall_x, wall_y))
    

    def convert_to_map_coordinates(self, position: tuple[float, float]) -> tuple[int, int]:
        scale_x = self.map.size[0] / self.map.world_size_m[0]
        scale_y = self.map.size[1] / self.map.world_size_m[1]
        map_x = int(math.floor(position[0] * scale_x)) + self.map.size[0] // 2
        map_y = int(math.floor(position[1] * scale_y)) + self.map.size[1] // 2
        return (map_x, map_y)    
  

if __name__ == "__main__":
    supervisor = Supervisor()
    turtle_bot = MyRobot(wheel_radius=0.033, axle_length=0.16, bot=supervisor, supervisor=supervisor, use_supervisor=True)
    turtle_bot.setmap(size=(1000, 1000), world_size_m=(4.0, 4.0))
    world_map = turtle_bot.mapping(iteration=100000, verbose=100, teleop=False, show_display=True, proba_increase=0.1, threshold=0.95)
    turtle_bot.draw_map(threshold=0.1) # Final map visualization with a threshold for occupied cells
    print('done')