import logging as log

import cv2
import numpy as np
import numpy.typing as npt
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import CvPoint, CvSize, Homography, Image, TranslationVector, WarpingInfo, NpShape


class OrbRansacHandler(BaseHomographyHandler):
    orb_features: int
    min_matches: int
    relevant_area_one: tuple[CvPoint, CvPoint] | None
    relevant_area_two: tuple[CvPoint, CvPoint] | None
    _mask_img1: npt.NDArray | None
    _mask_img2: npt.NDArray | None

    def __init__(
        self,
        continous_recomputation: bool,
        orb_features: int = 50000,
        min_matches: int = 10,
        relevant_area_one: tuple[CvPoint, CvPoint] | None = None,
        relevant_area_two: tuple[CvPoint, CvPoint] | None = None,
    ):
        log.debug("Init Orb Ransac Homography Handler")
        super().__init__(continous_recomputation)
        self.orb_features = orb_features
        self.min_matches = min_matches
        self.relevant_area_one = relevant_area_one
        self.relevant_area_two = relevant_area_two
        self._mask_img1 = None
        self._mask_img2 = None

    def _find_homography(self, img1: Image, img2: Image) -> WarpingInfo:
        homography = self._compute_homography_matrix(img1, img2)
        translation, canvas_size = self._compute_target_canvas(img1.shape, img2.shape, homography)
        return translation, canvas_size, homography

    def _compute_homography_matrix(self, img1: Image, img2: Image) -> Homography:
        # Create masks if needed to restrict detection to relevant areas
        if self.relevant_area_one is not None and self._mask_img1 is None:
            (xs, ys), (xe, ye) = self.relevant_area_one
            self._mask_img1 = np.zeros(shape=img1.shape[:2], dtype=np.uint8)
            self._mask_img1[ys : ye + 1, xs : xe + 1] = 255

        if self.relevant_area_two is not None and self._mask_img2 is None:
            (xs, ys), (xe, ye) = self.relevant_area_two
            self._mask_img2 = np.zeros(shape=img2.shape[:2], dtype=np.uint8)
            self._mask_img2[ys : ye + 1, xs : xe + 1] = 255

        # Create ORB and compute
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        kpts_img1, descs_img1 = orb.detectAndCompute(img1, mask=self._mask_img1)  # queryImage
        kpts_img2, descs_img2 = orb.detectAndCompute(img2, mask=self._mask_img2)  # trainImage

        # Match and return default if not enough matches
        bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches = bfm.match(descs_img1, descs_img2)
        if len(matches) <= self.min_matches:
            return Homography(np.zeros((3, 3), dtype=np.float64))

        # Convert keypoints to an argument for findHomography
        dst_pts = np.array([kpts_img1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        src_pts = np.array([kpts_img2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

        # Establish a homography
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        return Homography(h)

    def _compute_target_canvas(
        self, img1_dim: NpShape, img2_dim: NpShape, homography: Homography
    ) -> tuple[TranslationVector, CvSize]:
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
        translation = TranslationVector((-x_min, -y_min))
        canvas_size = CvSize((x_max - x_min, y_max - y_min))
        return translation, canvas_size
