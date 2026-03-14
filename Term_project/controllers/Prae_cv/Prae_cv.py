from controller import Robot, Camera, Display
import numpy as np
import imageio
from milestone1_library import PerceptionSystem

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ===== CAMERA =====
camera = robot.getDevice("camera")
camera.enable(timestep)

width = camera.getWidth()
height = camera.getHeight()

# ===== DISPLAY =====
display = robot.getDevice("display")

# ===== PERCEPTION =====
perception = PerceptionSystem()

# LOAD GOAL IMAGE
goal_img = imageio.imread(
    "goal_reference.png"
)[:, :, :3]
perception.load_goal_image(goal_img)

print("Milestone 1 – Computer Vision Controller Started")


def draw_rectangle(img, bbox, color=(255, 0, 0), thickness=2):
    """Draw rectangle directly on RGB image"""
    min_r, min_c, max_r, max_c = bbox

    img[min_r:min_r+thickness, min_c:max_c] = color
    img[max_r-thickness:max_r, min_c:max_c] = color
    img[min_r:max_r, min_c:min_c+thickness] = color
    img[min_r:max_r, max_c-thickness:max_c] = color


while robot.step(timestep) != -1:
    image = camera.getImage()

    # Camera image is BGRA
    img = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))

    # Convert to RGB for processing
    frame_rgb = img[:, :, :3].copy()

    result = perception.process_frame(frame_rgb)
    if result is None:
        continue

    print("\n--- Frame analysis ---")

    for i, f in enumerate(result["features"]):
        cx, cy = f.center
        min_r, min_c, max_r, max_c = f.bbox

        # ===== DRAW RECTANGLE =====
        if f.is_goal:
            draw_rectangle(frame_rgb, f.bbox, (0, 255, 0))   # green
        elif f.moving:
            draw_rectangle(frame_rgb, f.bbox, (255, 255, 0)) # yellow
        else:
            draw_rectangle(frame_rgb, f.bbox, (255, 0, 0))   # red

        # ===== LABEL =====
        if f.is_goal and f.moving:
            label = "GOAL (DYNAMIC)"
        elif f.is_goal:
            label = "GOAL (STATIC)"
        elif f.moving:
            label = "DYNAMIC OBJECT"
        else:
            label = "STATIC OBJECT"

        print(
            f"[{label}] Object {i} "
            f"score={f.goal_score:.3f} "
            f"bbox=({min_c},{min_r})-({max_c},{max_r}) "
            f"center=({cx:.1f},{cy:.1f})"
        )

    # ===== SEND TO DISPLAY =====
    # Convert RGB → BGRA for Webots
    bgra = np.zeros((height, width, 4), dtype=np.uint8)
    bgra[:, :, 0] = frame_rgb[:, :, 2]  # B
    bgra[:, :, 1] = frame_rgb[:, :, 1]  # G
    bgra[:, :, 2] = frame_rgb[:, :, 0]  # R
    bgra[:, :, 3] = 255                 # Alpha

    image_ref = display.imageNew(
        bgra.tobytes(),
        Display.BGRA,
        width,
        height
    )

    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)