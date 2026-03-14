import numpy as np
from PIL import Image

def resize_image(image_array, new_size):
    img = Image.fromarray(image_array)
    img_resized = img.resize(new_size)
    return np.array(img_resized)

def convolution(kernel: np.ndarray, array: np.ndarray) -> np.ndarray:
    shape = kernel.shape
    window = shape[0]  # must be square kernel
    # pad 0 around
    row, column = array.shape
    pad_width = window // 2
    padded_array = np.pad(array, pad_width=pad_width, mode='constant', constant_values=0)
    new_img_array = np.zeros((row, column))
    for i in range(row): 
        for j in range(column):
            region = padded_array[i:i+window, j:j+window]
            new_img_array[i, j] = np.sum(region * kernel)
    return new_img_array

def edge_detection(frame_blurred, vertical, horizontal):
    vertical_edges = convolution(vertical, frame_blurred)
    horizontal_edges = convolution(horizontal, frame_blurred)
    edge = np.sqrt(np.square(vertical_edges) + np.square(horizontal_edges))
    return edge

def blob_ize(array):
    visited = np.zeros_like(array, dtype=np.uint8)
    rows, cols = array.shape
    blobs = []
    for i in range(rows):
        for j in range(cols):
            if array[i, j] != 0 or visited[i, j]:
                continue
            # start a new blob
            blob = set()
            assign_blob(array, visited, i, j, blob)
            blobs.append(blob)
    return blobs

def assign_blob(array, visited, i, j, blob):
    rows, cols = array.shape
    stack = [(i, j)]
    while stack:
        x, y = stack.pop()
        # bounds
        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue
        # stop at edges or visited pixels
        if visited[x, y] or array[x, y] != 0:
            continue
        visited[x, y] = 1
        blob.add((x, y))
        # 4-connected neighbors
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

def blobs_on_orginalImage(
    blobs,
    originalImage,
    resized_shape=(255, 255),
    color=(255, 0, 0),
    thickness=1
):
    img = originalImage.copy()
    orig_h, orig_w, _ = img.shape
    res_h, res_w = resized_shape

    scale_r = orig_h / res_h
    scale_c = orig_w / res_w

    for blob in blobs:
        for r, c in blob:
            rr = int(r * scale_r)
            cc = int(c * scale_c)

            # fill with thickness
            for t_r in range(-thickness, thickness + 1):
                for t_c in range(-thickness, thickness + 1):
                    rrr = rr + t_r
                    ccc = cc + t_c
                    if 0 <= rrr < orig_h and 0 <= ccc < orig_w:
                        img[rrr, ccc] = color

    return img



def visualize_blobs(blobs, shape, background=(0, 0, 0), seed=42):
    """
    blobs: list of sets {(row, col)}
    shape: (rows, cols)
    """
    rows, cols = shape

    img = np.zeros((rows, cols, 3), dtype=np.uint8)
    img[:] = background

    rng = np.random.default_rng(seed)

    for blob in blobs:
        # random color per blob
        color = rng.integers(0, 256, size=3, dtype=np.uint8)

        for (r, c) in blob:
            img[r, c] = color

    return Image.fromarray(img)

def center_of_blob(blob):
    rows = [coord[0] for coord in blob]
    cols = [coord[1] for coord in blob]
    center_row = sum(rows) / len(rows)
    center_col = sum(cols) / len(cols)
    return (center_row, center_col)

def center_of_blobs(blobs):
    return [center_of_blob(blob) for blob in blobs]

def find_moving_objects_in_latest_frame(frame1, frame2, resize_to=(150, 150)):
    # frame1 and frame2 are RGB images as numpy arrays
    # resize
    frame1_resized = resize_image(frame1, resize_to)
    frame2_resized = resize_image(frame2, resize_to)
    

    # convert to grayscale
    frame1_gray = 0.299 * frame1_resized[:,:,0] + 0.587 * frame1_resized[:,:,1] + 0.114 * frame1_resized[:,:,2]
    frame2_gray = 0.299 * frame2_resized[:,:,0] + 0.587 * frame2_resized[:,:,1] + 0.114 * frame2_resized[:,:,2]
    
    #r,g,b channels
    frame2R = frame2_resized[:,:,0]
    frame2G = frame2_resized[:,:,1]
    frame2B = frame2_resized[:,:,2]

    #blur
    gaussian_blurred = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]]) / 16.0
    frame1_blurred = convolution(gaussian_blurred, frame1_gray)
    frame2_blurred = convolution(gaussian_blurred, frame2_gray)
    frame2R_blurred = convolution(gaussian_blurred, frame2R)
    frame2G_blurred = convolution(gaussian_blurred, frame2G)    
    frame2B_blurred = convolution(gaussian_blurred, frame2B)
    
    #subtract
    frame2subframe1 = frame2_blurred - frame1_blurred
    
    #threshold
    threshold = 120 
    thresholded_2sub1 = (frame2subframe1 > threshold).astype(np.uint8) * 255

    #edge detection
    vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    edge2B = edge_detection(frame2B_blurred, vertical, horizontal)
    edge2 = edge_detection(frame2_blurred, vertical, horizontal)
    edge2R = edge_detection(frame2R_blurred, vertical, horizontal)
    edge2G = edge_detection(frame2G_blurred, vertical, horizontal)
    threshold = 75
    edge2_thresholded = (edge2 > threshold).astype(np.uint8) * 255
    edge2B_thresholded = (edge2B > threshold).astype(np.uint8) * 255
    edge2R_thresholded = (edge2R > threshold).astype(np.uint8) * 255
    edge2G_thresholded = (edge2G > threshold).astype(np.uint8) * 255
    edge2_full = edge2_thresholded + edge2B_thresholded + edge2R_thresholded + edge2G_thresholded
    
    #blob detection
    blobs2 = blob_ize(edge2_full)

    #filter blobs by subtraction threshold
    coor_surviving_sub = []
    for r in range(thresholded_2sub1.shape[0]):
        for c in range(thresholded_2sub1.shape[1]):
            if thresholded_2sub1[r, c] != 0:
                coor_surviving_sub.append((r, c))
    coor_surviving_sub = set(coor_surviving_sub)
    moving_blobs2 = []
    static_blobs2 = []
    for blob in blobs2:
        if any(coor in coor_surviving_sub for coor in blob):
            moving_blobs2.append(blob)
        else:
            static_blobs2.append(blob)
    # visualize_blobs(moving_blobs2, edge2_full.shape).show()
    
    return moving_blobs2
