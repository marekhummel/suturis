import logging as log

import cv2
import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class BasicOrbHandler(BaseHomographyHandler):
    orb_features: int
    min_matches: int
    enable_debug_output: bool

    def __init__(
        self,
        continous_recomputation: bool,
        save_to_file: bool = False,
        orb_features: int = 50000,
        min_matches: int = 10,
        enable_debug_output: bool = False,
    ):
        log.debug("Init Orb Ransac Homography Handler")
        super().__init__(continous_recomputation, save_to_file)
        self.orb_features = orb_features
        self.min_matches = min_matches
        self.enable_debug_output = enable_debug_output

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        # Create ORB and compute
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        kpts_img1, descs_img1 = orb.detectAndCompute(img1)  # queryImage
        kpts_img2, descs_img2 = orb.detectAndCompute(img2)  # trainImage

        # Match and return default if not enough matches
        bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches = bfm.match(descs_img1, descs_img2)

        if self.enable_debug_output:
            self._output_debug_images(img1, img2, kpts_img1, kpts_img2, matches)

        if len(matches) <= self.min_matches:
            return Homography(np.identity(3, dtype=np.float64))

        # Convert keypoints to an argument for findHomography
        dst_pts = np.array([kpts_img1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        src_pts = np.array([kpts_img2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

        # Establish a homography
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        return Homography(h)

    def _output_debug_images(self, img1: Image, img2: Image, kpts_img1: list, kpts_img2: list, matches: list) -> None:
        matches_img = cv2.drawMatches(
            img1, kpts_img1, img2, kpts_img2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("data/out/osift_matches.jpg", matches_img)

        kpts1_img = cv2.drawKeypoints(img1, kpts_img1, None, color=(72, 144, 233))
        cv2.imwrite("data/out/osift_keypoints1.jpg", kpts1_img)

        kpts2_img = cv2.drawKeypoints(img2, kpts_img2, None, color=(72, 144, 233))
        cv2.imwrite("data/out/osift_keypoints2.jpg", kpts2_img)
