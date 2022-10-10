import numpy as np
import cv2


def find_crop(img_shape, homography, translation):
    # Define corners
    height, width = img_shape[:2]
    corners = np.array([[[0, 0]], [[0, height - 1]], [[width - 1, height - 1]], [[width - 1, 0]]], dtype=np.float32)

    # Compute corners after transformation for both img1 and img2
    tx, ty = translation
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    img1_corners_transformed = cv2.perspectiveTransform(corners, translation_matrix).astype(np.int32)
    img2_corners_transformed = cv2.perspectiveTransform(corners, translation_matrix @ homography).astype(np.int32)

    # Find min and max corner (not necessarily a corner, just min/max x and y as a point) in each image
    img1_corners_min = np.min(img1_corners_transformed, axis=0)
    img1_corners_max = np.max(img1_corners_transformed, axis=0)
    img2_corners_min = np.min(img2_corners_transformed, axis=0)
    img2_corners_max = np.max(img2_corners_transformed, axis=0)

    # For left top use the max of the min, for right bot use min of max
    x_start, y_start = np.max(np.concatenate([img1_corners_min, img2_corners_min]), axis=0)
    x_end, y_end = np.min(np.concatenate([img1_corners_max, img2_corners_max]), axis=0)
    crop_size = (y_end - y_start + 1, x_end - x_start + 1)
    return (((y_start, x_start)), ((y_end, x_end))), crop_size


def _compute_target_canvas(img1_dim, img2_dim, homography):
    def get_corner_pixels(img_dim) -> list[list[int]]:
        height, width = img_dim
        return [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]

    # Set corners
    img1_corners = np.array(get_corner_pixels(img1_dim[:2]), dtype=np.float32).reshape(4, 1, 2)
    img2_corners = np.array(get_corner_pixels(img2_dim[:2]), dtype=np.float32).reshape(4, 1, 2)

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


def analyze_transformed_canvas(img_shape, homography):
    # Set corners
    height, width = img_shape[:2]
    corners = np.array(
        [[[0, 0]], [[0, height - 1]], [[width - 1, height - 1]], [[width - 1, 0]]], dtype=np.float32
    ).reshape(4, 1, 2)

    # Transform second image corners with homography
    corners_homography = cv2.perspectiveTransform(corners, homography)

    # Find min and max of all corners
    all_corners = np.concatenate((corners, corners_homography), axis=0)
    x_min, y_min = np.around(all_corners.min(axis=0).ravel()).astype(np.int32)
    x_max, y_max = np.around(all_corners.max(axis=0).ravel()).astype(np.int32)

    # Compute translation and canvas size
    translation = (-x_min, -y_min)
    canvas_size = (x_max - x_min, y_max - y_min)

    # Apply transformation to find crop
    img1_corners_transformed = corners + np.array([translation[0], translation[1]])
    img2_corners_transformed = corners_homography + np.array([translation[0], translation[1]])

    # Find min and max corner (not necessarily a corner, just min/max x/y as a point) in each image
    img1_corners_min = np.min(img1_corners_transformed, axis=0)
    img1_corners_max = np.max(img1_corners_transformed, axis=0)
    img2_corners_min = np.min(img2_corners_transformed, axis=0)
    img2_corners_max = np.max(img2_corners_transformed, axis=0)

    # For left top use the max of the min, for right bot use min of max
    x_start, y_start = np.floor(np.max(np.concatenate([img1_corners_min, img2_corners_min]), axis=0)).astype(np.int32)
    x_end, y_end = np.ceil(np.min(np.concatenate([img1_corners_max, img2_corners_max]), axis=0)).astype(np.int32)
    crop_area = (y_start, x_start), (y_end, x_end)
    crop_size = (y_end - y_start + 1, x_end - x_start + 1)

    # Return
    return canvas_size, translation, crop_area, crop_size


img_shape = (720, 1280, 3)
homography = np.array(
    [
        [9.72207305e-01, -2.39784214e-02, 4.52830749e01],
        [-2.65546526e-02, 9.42927680e-01, -6.70054479e00],
        [-1.03685511e-05, -1.01653987e-04, 1.00000000e00],
    ]
)

translation, canvas_size = _compute_target_canvas(img_shape[:2], img_shape[:2], homography)
(crop_top, crop_bot), crop_size = find_crop(img_shape, homography, translation)

print(translation)
print(canvas_size)
print(crop_top, crop_bot)
print(crop_size)
print()

translation2, canvas_size2, (crop_top2, crop_bot2), crop_size2 = analyze_transformed_canvas(img_shape, homography)

print(translation)
print(canvas_size)
print(crop_top, crop_bot)
print(crop_size)
