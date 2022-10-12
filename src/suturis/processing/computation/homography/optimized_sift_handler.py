import logging as log

import cv2
import numpy as np
import numpy.typing as npt
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import CvRect, Homography, Image


class OptimizedSiftHandler(BaseHomographyHandler):
    sift_features: int
    min_matches: int
    relevant_areas_one: list[CvRect]
    relevant_areas_two: list[CvRect]
    enable_debug_output: bool
    _mask_img1: npt.NDArray | None
    _mask_img2: npt.NDArray | None

    def __init__(
        self,
        continous_recomputation: bool,
        save_to_file: bool = False,
        sift_features: int = 50000,
        min_matches: int = 10,
        relevant_areas_one: list[CvRect] = [],
        relevant_areas_two: list[CvRect] = [],
        enable_debug_output: bool = False,
    ):
        log.debug("Init Orb Ransac Homography Handler")
        super().__init__(continous_recomputation, save_to_file)
        self.sift_features = sift_features
        self.min_matches = min_matches
        self.relevant_areas_one = relevant_areas_one
        self.relevant_areas_two = relevant_areas_two
        self._mask_img1 = None
        self._mask_img2 = None
        self.enable_debug_output = enable_debug_output

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        # Create masks if needed to restrict detection to relevant areas
        if len(self.relevant_areas_one) > 0 and self._mask_img1 is None:
            self._mask_img1 = np.zeros(shape=img1.shape[:2], dtype=np.uint8)
            for (xs, ys), (xe, ye) in self.relevant_areas_one:
                self._mask_img1[ys : ye + 1, xs : xe + 1] = 255

        if len(self.relevant_areas_two) > 0 and self._mask_img2 is None:
            self._mask_img2 = np.zeros(shape=img2.shape[:2], dtype=np.uint8)
            for (xs, ys), (xe, ye) in self.relevant_areas_two:
                self._mask_img2[ys : ye + 1, xs : xe + 1] = 255

        # Create SIFT and compute
        sift = cv2.SIFT_create(nfeatures=self.sift_features)
        kpts_img1, descs_img1 = sift.detectAndCompute(img1, mask=self._mask_img1)  # queryImage
        kpts_img2, descs_img2 = sift.detectAndCompute(img2, mask=self._mask_img2)  # trainImage

        # Match and return default if not enough matches
        bfm = cv2.BFMatcher_create(cv2.NORM_L2)
        matches = bfm.match(descs_img1, descs_img2)
        good_matches = self._filter_good_matches(matches, kpts_img1, kpts_img2)

        if self.enable_debug_output:
            self._output_debug_images(img1, img2, kpts_img1, kpts_img2, good_matches)

        if len(good_matches) < self.min_matches:
            return Homography(np.identity(3, dtype=np.float64))

        # Convert keypoints to an argument for findHomography
        dst_pts = np.array([kpts_img1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
        src_pts = np.array([kpts_img2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

        # Establish a homography
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        return Homography(h)

    def _filter_good_matches(self, matches, kpts_img1, kpts_img2):
        def loc_distance(m):
            pt1 = kpts_img1[m.queryIdx].pt
            pt2 = kpts_img2[m.trainIdx].pt
            return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2

        max_loc_distance = max(loc_distance(m) for m in matches)
        max_desc_distance = max(m.distance for m in matches)

        def comb_distance(m):
            return (2 * m.distance / max_desc_distance + loc_distance(m) / max_loc_distance) / 3

        return [m for m in matches if comb_distance(m) < 0.25]

    def _output_debug_images(self, img1: Image, img2: Image, kpts_img1: list, kpts_img2: list, matches: list) -> None:
        matches_img = cv2.drawMatches(
            img1, kpts_img1, img2, kpts_img2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("data/out/osift_matches.jpg", matches_img)

        kpts1_img = cv2.drawKeypoints(img1, kpts_img1, None, color=(72, 144, 233))
        cv2.imwrite("data/out/osift_keypoints1.jpg", kpts1_img)

        kpts2_img = cv2.drawKeypoints(img2, kpts_img2, None, color=(72, 144, 233))
        cv2.imwrite("data/out/osift_keypoints2.jpg", kpts2_img)
