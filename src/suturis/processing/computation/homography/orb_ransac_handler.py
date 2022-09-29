import logging as log

import cv2
import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import CvSize, Homography, Image, NpSize, Point, TranslationVector


class OrbRansacHandler(BaseHomographyHandler):
    orb_features: int
    min_matches: int

    def __init__(self, continous_recomputation: bool, orb_features: int = 50000, min_matches: int = 10):
        log.debug("Init Orb Ransac Homography Handler")
        super().__init__(continous_recomputation)
        self.orb_features = orb_features
        self.min_matches = min_matches

    def find_homography(self, img1: Image, img2: Image) -> tuple[TranslationVector, CvSize, Homography]:
        homography = self._compute_homography_matrix(img1, img2)
        translation, target_size = self._compute_target_canvas(img1.shape[:2], img2.shape[:2], homography)
        return translation, target_size, homography

    def find_crop(self, img1: Image, homography: Homography, translation: TranslationVector) -> tuple[Point, Point]:
        points = np.array(
            [
                [0, 0, 1],
                [img1.shape[1] - 1, 0, 1],
                [0, img1.shape[0] - 1, 1],
                [img1.shape[1] - 1, img1.shape[0] - 1, 1],
            ]
        )
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
        return start, end

    def _compute_homography_matrix(self, img1: Image, img2: Image):
        # Create ORB and compute
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        query_keypoints, query_descriptors = orb.detectAndCompute(img1, None)  # queryImage
        train_keypoints, train_descriptors = orb.detectAndCompute(img2, None)  # trainImage

        # Match
        bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches = bfm.knnMatch(query_descriptors, train_descriptors, k=2)
        good = [(m, n) for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) <= self.min_matches:
            return np.zeros((3, 3))

        # Convert keypoints to an argument for findHomography
        dst_pts = np.array([query_keypoints[m.queryIdx].pt for m, _ in good], dtype=np.float32).reshape(-1, 1, 2)
        src_pts = np.array([train_keypoints[m.trainIdx].pt for m, _ in good], dtype=np.float32).reshape(-1, 1, 2)

        # Establish a homography
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        return h

    def _compute_target_canvas(self, img1_dim: NpSize, img2_dim: NpSize, homography: Homography) -> tuple[Point, Point]:
        # Set corners
        img1_corners = np.array(self._get_corner_pixels(img1_dim), dtype=np.float32).reshape(4, 1, 2)
        img2_corners = np.array(self._get_corner_pixels(img2_dim), dtype=np.float32).reshape(4, 1, 2)

        # Transform second image corners with homography
        img2_corners_transformed = cv2.perspectiveTransform(img2_corners, homography)

        # Find min and max of all corners
        all_corners = np.concatenate((img1_corners, img2_corners_transformed), axis=0)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Set translation and output size
        translation = (-x_min, -y_min)
        target_size = (x_max - x_min, y_max - y_min)
        return translation, target_size

    def _get_corner_pixels(self, img_dim: NpSize) -> list[list[int]]:
        height, width = img_dim
        return [[0, 0], [0, height], [width, height], [width, 0]]
