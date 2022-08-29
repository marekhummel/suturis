import cv2
import numpy as np


def find_homography_matrix(img1, img2):
    """
    Finds a homography matrix for two images that can be used to warp one image to the other
    """
    orb = cv2.ORB_create(nfeatures=50_000)
    query_keypoints, query_descriptors = orb.detectAndCompute(img1, None)  # queryImage
    train_keypoints, train_descriptors = orb.detectAndCompute(img2, None)  # trainImage

    bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = bfm.knnMatch(query_descriptors, train_descriptors, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append((m, n))

    min_matches = 10
    if len(good) > min_matches:
        # Convert keypoints to an argument for findHomography
        dst_pts = np.float32(
            [query_keypoints[m.queryIdx].pt for (m, _) in good]
        ).reshape(-1, 1, 2)
        src_pts = np.float32(
            [train_keypoints[m.trainIdx].pt for (m, _) in good]
        ).reshape(-1, 1, 2)

        # Establish a homography
        m = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)[0]
    else:
        m = np.zeros((3, 3))
    return m, good, query_keypoints, train_keypoints


def find_transformation(image1, image2, m):
    """
    Finds a translation matrix and other attributes.
    """
    rows1, cols1 = image1.shape[:2]
    rows2, cols2 = image2.shape[:2]

    # Error must be somewhere around here||||||||||||||||||
    # Get a valid input value for the perspective transform
    list_of_points_1 = np.float32(
        [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]
    ).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(
        -1, 1, 2
    )

    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, m)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    # Get the translation parameters
    translation_dist = [-x_min, -y_min]
    # and here ||||||||||||||||||||||||||||||||||||||
    # Put translation matrix together
    h_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
    )
    return h_translation, x_max, x_min, y_max, y_min, translation_dist, rows1, cols1


def translate_and_warp(
    base,
    query,
    h_translation,
    x_delta,
    y_delta,
    M,
    translation_dist,
    rows1,
    cols1,
):
    query_warped = cv2.warpPerspective(query, h_translation.dot(M), (x_delta, y_delta))

    base_translated = np.zeros_like(query_warped)
    base_translated[
        translation_dist[1] : rows1 + translation_dist[1],
        translation_dist[0] : cols1 + translation_dist[0],
    ] = base

    return base_translated, query_warped
