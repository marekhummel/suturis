import numpy as np
import cv2


def compute_target_canvas(img1_dim, img2_dim, homography):
    def get_corner_pixels(img_dim) -> list[list[int]]:
        height, width = img_dim
        return [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]

    # Set corners
    img1_corners = np.array(get_corner_pixels(img1_dim), dtype=np.float32).reshape(4, 1, 2)
    img2_corners = np.array(get_corner_pixels(img2_dim), dtype=np.float32).reshape(4, 1, 2)

    # Transform second image corners with homography
    img2_corners_transformed = cv2.perspectiveTransform(img2_corners, homography)

    # Find min and max of all corners
    all_corners = np.concatenate((img1_corners, img2_corners_transformed), axis=0)
    x_min, y_min = np.around(all_corners.min(axis=0).ravel()).astype(np.int32)
    x_max, y_max = np.around(all_corners.max(axis=0).ravel()).astype(np.int32)

    # Set translation and output size
    translation = (-x_min, -y_min)
    canvas_size = (x_max - x_min, y_max - y_min)
    return translation, canvas_size


img1 = cv2.imread("data/out/debug/img1.jpg")
img2 = cv2.imread("data/out/debug/img1.jpg")
homography = np.load("./data/examples/files/homography_bowstern.npy")


translation, canvas_size = compute_target_canvas(img1.shape[:2], img2.shape[:2], homography)


np.savez(
    "./data/examples/files/homography_bowstern.npz", canvas=canvas_size, translation=translation, homography=homography
)
