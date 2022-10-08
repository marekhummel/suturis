import numpy as np
import cv2


def find_crop(img_shape, homography, translation):
    height, width = img_shape[:2]
    points = np.array([[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1], [width - 1, height - 1, 1]])
    hom_points: np.ndarray = np.array([])
    trans_points: np.ndarray = np.array([])
    h_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
    # Apply transformations to all of those corner points
    for pts in points:
        # Warp the points
        tmp = cv2.perspectiveTransform(np.array([[[pts[0], pts[1]]]], dtype=np.float32), homography)
        # Add the translation
        tmp = np.matmul(h_translation, np.array([tmp[0][0][0], tmp[0][0][1], 1]))
        hom_points = np.concatenate((hom_points, tmp))
        trans_points = np.concatenate((trans_points, np.matmul(h_translation, pts)))
    # Calculating the perfect corner points
    start = (
        int(round(max(min(hom_points[1::3]), min(trans_points[1::3])))),
        int(round(max(min(hom_points[0::3]), min(trans_points[0::3])))),
    )
    end = (
        int(round(min(max(hom_points[1::3]), max(trans_points[1::3])))),
        int(round(min(max(hom_points[0::3]), max(trans_points[0::3])))),
    )
    crop_size = (end[0] - start[0] + 1, end[1] - start[1] + 1)
    return (start, end), crop_size


def find_crop2(img_shape, homography, translation):
    height, width = img_shape[:2]
    corners = np.array([[[0, 0]], [[0, height - 1]], [[width - 1, height - 1]], [[width - 1, 0]]], dtype=np.float32)

    tx, ty = translation
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    img1_corners_transformed = cv2.perspectiveTransform(corners, translation_matrix).astype(np.int32)
    img2_corners_transformed = cv2.perspectiveTransform(corners, translation_matrix @ homography).astype(np.int32)

    img1_corners_min = np.min(img1_corners_transformed, axis=0)
    img1_corners_max = np.max(img1_corners_transformed, axis=0)
    img2_corners_min = np.min(img2_corners_transformed, axis=0)
    img2_corners_max = np.max(img2_corners_transformed, axis=0)

    left = np.concatenate([img1_corners_min, img2_corners_min])
    right = np.concatenate([img1_corners_max, img2_corners_max])

    x_start, y_start = np.max(left, axis=0)
    x_end, y_end = np.min(right, axis=0)
    crop_size = (y_end - y_start + 1, x_end - x_start + 1)
    return ((y_start, x_start), (y_end, x_end)), crop_size


img1 = cv2.imread("data/out/Image1.png")
img2 = cv2.imread("data/out/Image2.png")
homography = np.array(
    [
        [9.72207305e-01, -2.39784214e-02, 4.52830749e01],
        [-2.65546526e-02, 9.42927680e-01, -6.70054479e00],
        [-1.03685511e-05, -1.01653987e-04, 1.00000000e00],
    ]
)
translation = [0, 41]

print(find_crop(img1.shape, homography, translation))
print(find_crop2(img1.shape, homography, translation))
