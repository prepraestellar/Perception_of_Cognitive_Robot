import numpy as np
from controller import Robot, Supervisor
from tqdm import tqdm

class Map:
    def __init__(self, size: tuple, world_size_m: tuple):
        self.map = np.zeros(size)
        self.size = size
        self.world_size_m = world_size_m

    def add_to_map(self, item: int, position: tuple[int, int]):
        if 0 <= position[0] < self.size[0] and 0 <= position[1] < self.size[1]:
            self.map[tuple(position)] = item 
        else:
            print(f"Warning: Position {position} is out of map bounds and will be ignored.")

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
    def __init__(self, wheel_radius=0.0203, axle_length=0.056):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

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
        self.ds2deg = { 'ps0': -15, 'ps1':-49, 'ps6': 49, 'ps7': 15 }


        # Distance sensors
        self.ds_names = ["ps0", "ps1", "ps6", "ps7"]
        self.distance_sensors = []
        for name in self.ds_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.time_step)
            self.distance_sensors.append(sensor)
        self.ds_values = [0.0] * len(self.ds_names)

        # assume perfect odom
        self.supervisor = Supervisor() 
        self.time_step = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getSelf()
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        self.display = self.robot.getDevice('display')

    def setmap(self, size: tuple, world_size_m: tuple) -> Map:
        self.map = Map(size, world_size_m)
        return self.map

    def draw_map(self):
        dw = self.display.getWidth()
        dh = self.display.getHeight()
        cell_w = max(1, dw // self.map.size[0])
        cell_h = max(1, dh // self.map.size[1])

        for i in range(self.map.size[0]):
            for j in range(self.map.size[1]):
                if self.map.map[i, j] == 1:
                    self.display.setColor(0x000000)  # wall: black
                else:
                    self.display.setColor(0xFFFFFF)  # free: white
                self.display.fillRectangle(j * cell_w, i * cell_h, cell_w, cell_h)

        # Draw robot position in red
        rx, ry = self.convert_to_map_coordinates(self.get_current_position())
        self.display.setColor(0xFF0000)
        self.display.fillRectangle(ry * cell_w, rx * cell_h, cell_w, cell_h)

    def inverse_kinematic(self, angle_deg, time):
        angle = np.deg2rad(angle_deg)   # convert to radians
        omega = angle / time            # robot angular velocity
        vr = (omega * self.axle_length) / (2 * self.wheel_radius)
        vl = -vr
        return vr, vl
    
    def random_direction(self):
        direction = np.random.choice(['s', 'e', 'w', 'nw', 'ne', 'sw', 'se'])
        return direction
    def mapping(self, iteration, verbose=100):
        for i in tqdm(range(iteration)):
            self.left_motor.setVelocity(6.28)
            self.right_motor.setVelocity(6.28)
            self.robot.step(self.time_step)   # advance simulation
            self.read_distance_sensors()      # re-read sensors
            Obstacle = self.check_obstacle()
            if Obstacle:
                wall_positions = self.get_wall_position()
                for pos in wall_positions:
                    self.map.add_to_map(1, self.convert_to_map_coordinates(pos))  # add wall to map
                direction = self.random_direction()  # choose a random direction to turn
                angle_deg = self.dir_dict[direction]
                vr, vl = self.inverse_kinematic(angle_deg, 1.5)  # calculate wheel velocities for the turn
                self.left_motor.setVelocity(vl)
                self.right_motor.setVelocity(vr)
                self.robot.step(1500)

            self.draw_map()

            if verbose > 0 and (i + 1) % verbose == 0:
                self.map.print_map()  # print the map after exploration

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        return self.map

    def read_distance_sensors(self):
        for i, ds in enumerate(self.distance_sensors):
            if ds:
                # Scale the data to have a value between 0.0 and 1.0
                self.ds_values[i] = ds.getValue() / 4096.0


    def check_obstacle(self)->bool:
        return any(v > 0.1 for v in self.ds_values)

    def get_current_position(self) -> tuple[float, float]:
        position = self.trans_field.getSFVec3f()
        return (float(position[0]), float(position[1]))
    
    def get_wall_position(self) -> list[tuple[float, float]]:
        wall_positions = []
        for i, ds in enumerate(self.distance_sensors):
            if self.ds_values[i] > 0.1:  # If an obstacle is detected
                # Calculate the position of the wall based on the robot's current position and the sensor's orientation
                robot_position = self.get_current_position()
                rot = self.rot_field.getSFRotation()  # [ax, ay, az, angle]
                robot_yaw = rot[3] if rot[2] > 0 else -rot[3]  # Z-axis rotation
                angle_rad = robot_yaw + np.deg2rad(self.ds2deg[self.ds_names[i]])
                distance = self.ds_values[i] * 0.04  # Scale the distance ( max range is 4 cm)
                wall_x = robot_position[0] + distance * np.cos(angle_rad)
                wall_y = robot_position[1] + distance * np.sin(angle_rad)
                wall_positions.append((wall_x, wall_y))
        return wall_positions

    def convert_to_map_coordinates(self, position: tuple[float, float]) -> tuple[int, int]:
        scale_x = self.map.size[0] / self.map.world_size_m[0]
        scale_y = self.map.size[1] / self.map.world_size_m[1]
        map_x = int(position[0] * scale_x) + self.map.size[0] // 2
        map_y = int(position[1] * scale_y) + self.map.size[1] // 2
        return (map_x, map_y)    

if __name__ == "__main__":
    e_puck = MyRobot()
    e_puck.setmap((30, 30), (2.0, 2.0))
    world_map = e_puck.mapping(iteration=10000, verbose=100)

    print('done')