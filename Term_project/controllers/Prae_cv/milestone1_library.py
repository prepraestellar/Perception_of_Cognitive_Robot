import numpy as np
import math

# ===============================
# Utility functions
# ===============================

def center_of_mass(blob):
    xs = [p[1] for p in blob]
    ys = [p[0] for p in blob]
    return (np.mean(xs), np.mean(ys))


def bounding_box(blob):
    rows = [p[0] for p in blob]
    cols = [p[1] for p in blob]
    return min(rows), min(cols), max(rows), max(cols)


def color_histogram(blob, frame):
    hist = np.zeros((8, 8, 8), dtype=np.float32)
    for (r, c) in blob:
        R, G, B = frame[r, c]
        hist[R // 32, G // 32, B // 32] += 1
    hist /= np.sum(hist) + 1e-8
    return hist.flatten()


def hog_histogram(blob, gx, gy, bins=8):
    hist = np.zeros(bins, dtype=np.float32)
    for (r, c) in blob:
        mag = math.sqrt(gx[r, c]**2 + gy[r, c]**2)
        angle = (math.degrees(math.atan2(gy[r, c], gx[r, c])) + 180) % 180
        hist[int(angle // (180 / bins))] += mag
    hist /= np.sum(hist) + 1e-8
    return hist


def bhattacharyya_distance(h1, h2):
    return np.sqrt(1 - np.sum(np.sqrt(h1 * h2)))


def convolution(kernel: np.ndarray, array: np.ndarray) -> np.ndarray:
    k_rows, k_cols = kernel.shape
    rows, cols = array.shape

    pad_h, pad_w = k_rows // 2, k_cols // 2
    
    # Pad the image
    padded_image = np.pad(array, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros((rows, cols), dtype=np.float64)
    
    for i in range(k_rows):
        for j in range(k_cols):
            # Get the shifted view
            shifted_view = padded_image[i : i + rows, j : j + cols]
            output += shifted_view * kernel[i, j]
            
    return output


def compute_gradients(gray):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    gx = convolution(sobel_x, gray)
    gy = convolution(sobel_y, gray)
    return gx, gy


# ===============================
# Morphology
# ===============================

def binary_erosion(binary):
    padded = np.pad(binary, 1, mode='constant', constant_values=False)
    return (
        padded[1:-1, 1:-1] &
        padded[:-2, 1:-1] &
        padded[2:, 1:-1] &
        padded[1:-1, :-2] &
        padded[1:-1, 2:]
    )


def binary_dilation(binary):
    padded = np.pad(binary, 1, mode='constant', constant_values=False)
    return (
        padded[1:-1, 1:-1] |
        padded[:-2, 1:-1] |
        padded[2:, 1:-1] |
        padded[1:-1, :-2] |
        padded[1:-1, 2:]
    )


# ===============================
# Blob detection
# ===============================

def blob_ize(binary):
    visited = np.zeros_like(binary, dtype=bool)
    blobs = []

    for r in range(binary.shape[0]):
        for c in range(binary.shape[1]):
            if not binary[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            blob = set()

            while stack:
                x, y = stack.pop()
                if x < 0 or y < 0 or x >= binary.shape[0] or y >= binary.shape[1]:
                    continue
                if visited[x, y] or not binary[x, y]:
                    continue

                visited[x, y] = True
                blob.add((x, y))
                stack.extend([(x+1,y), (x-1,y), (x,y+1), (x,y-1)])

            if len(blob) > 50:
                blobs.append(blob)

    return blobs


def compute_blob_motion(blob, motion_mask):
    return sum(motion_mask[r, c] for (r, c) in blob) / (len(blob) + 1e-8)


# ===============================
# Blob feature
# ===============================

class BlobFeature:
    def __init__(self, blob, frame, gx, gy, motion_mask):
        self.blob = blob
        self.center = center_of_mass(blob)
        self.bbox = bounding_box(blob)  # 🔹 ADD ONLY THIS

        self.color = color_histogram(blob, frame)
        self.hog = hog_histogram(blob, gx, gy)

        self.moving_ratio = compute_blob_motion(blob, motion_mask)
        self.moving = self.moving_ratio > 0.05

        self.goal_score = None
        self.is_goal = False


# ===============================
# Perception System
# ===============================

class PerceptionSystem:
    def __init__(self):
        self.prev_gray = None
        self.goal_color = None
        self.goal_hog = None

        self.blur_kernel = np.array([
            [1,2,1],
            [2,4,2],
            [1,2,1]
        ]) / 16.0

    def load_goal_image(self, goal_img):
        gray = np.dot(goal_img[..., :3], [0.299, 0.587, 0.114])
        gx, gy = compute_gradients(gray)

        h, w = gray.shape
        blob = {(r, c) for r in range(h) for c in range(w)}

        self.goal_color = color_histogram(blob, goal_img)
        self.goal_hog = hog_histogram(blob, gx, gy)

        print("Goal image loaded")

    def process_frame(self, frame_rgb):
        gray = np.dot(frame_rgb[..., :3], [0.299, 0.587, 0.114])

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = convolution(self.blur_kernel, np.abs(gray - self.prev_gray))
        motion_mask = diff > 6

        gx, gy = compute_gradients(gray)
        mag = np.sqrt(gx**2 + gy**2)

        edges = mag > 80 #60
        edges = binary_erosion(edges)
        edges = binary_dilation(edges)

        blobs = blob_ize(edges)

        features = [
            BlobFeature(blob, frame_rgb, gx, gy, motion_mask)
            for blob in blobs
        ]

        if self.goal_color is not None:
            for f in features:
                Dc = bhattacharyya_distance(f.color, self.goal_color)
                Ds = bhattacharyya_distance(f.hog, self.goal_hog)
                f.goal_score = 0.6 * Dc + 0.4 * Ds
                f.is_goal = f.goal_score < 0.6

        self.prev_gray = gray
        return {"features": features}