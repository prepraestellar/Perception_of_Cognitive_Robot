from controller import Robot, Display
import numpy as np
from PIL import Image
from motion_detection_lib_optim import find_moving_objects_in_latest_frame, blobs_on_orginalImage

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# get camera
camera = robot.getDevice("camera")
camera.enable(timestep)
width = camera.getWidth()
height = camera.getHeight()

display = robot.getDevice("display")

prev_frame = None

print("Camera:", width, height)
print("Display:", display.getWidth(), display.getHeight())

blobs = []
while robot.step(timestep) != -1:
    # get image from camera
    img = camera.getImage()

    # convert Webots image → NumPy RGB
    frame = np.frombuffer(img, dtype=np.uint8).reshape((height, width, 4))
    frame = frame[:, :, :3]   # drop alpha channel

    if prev_frame is not None:
        blobs = find_moving_objects_in_latest_frame(
            prev_frame,
            frame
        )

        print(f"Moving objects detected: {len(blobs)}")

    processed_img = blobs_on_orginalImage(blobs, frame,resized_shape=(150,150))
    img_data = processed_img.astype(np.uint8).tobytes()
    ir = display.imageNew(img_data, Display.RGB, width, height)
    display.imagePaste(ir, 0, 0, False)
    display.imageDelete(ir)
    prev_frame = frame.copy()