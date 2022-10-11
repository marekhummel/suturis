import logging as log

import cv2
import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class BasicOrbHandler(BaseHomographyHandler):
    orb_features: int
    min_matches: int

    def __init__(
        self,
        continous_recomputation: bool,
        save_to_file: bool = False,
        orb_features: int = 50000,
        min_matches: int = 10,
    ):
        log.debug("Init Orb Ransac Homography Handler")
        super().__init__(continous_recomputation, save_to_file)
        self.orb_features = orb_features
        self.min_matches = min_matches

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        # Create ORB and compute
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        kpts_img1, descs_img1 = orb.detectAndCompute(img1, mask=self._mask_img1)  # queryImage
        kpts_img2, descs_img2 = orb.detectAndCompute(img2, mask=self._mask_img2)  # trainImage

        # Match and return default if not enough matches
        bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches = bfm.match(descs_img1, descs_img2)

        if len(matches) <= self.min_matches:
            return Homography(np.identity(3, dtype=np.float64))

        # Convert keypoints to an argument for findHomography
        dst_pts = np.array([kpts_img1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        src_pts = np.array([kpts_img2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

        # Establish a homography
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        return Homography(h)
