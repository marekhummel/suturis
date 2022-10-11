import logging as log

import cv2
import numpy as np
import numpy.typing as npt
from suturis.processing.computation.homography import NoWarpingHandler
from suturis.typing import CvRect, Homography, Image


class OrbRansacHandler(NoWarpingHandler):
    orb_features: int
    min_matches: int
    relevant_area_one: CvRect | None
    relevant_area_two: CvRect | None
    _mask_img1: npt.NDArray | None
    _mask_img2: npt.NDArray | None

    def __init__(
        self,
        continous_recomputation: bool,
        save_to_file: bool = False,
        orb_features: int = 50000,
        min_matches: int = 10,
        relevant_area_one: CvRect | None = None,
        relevant_area_two: CvRect | None = None,
    ):
        log.debug("Init Orb Ransac Homography Handler")
        super().__init__(continous_recomputation, save_to_file)
        self.orb_features = orb_features
        self.min_matches = min_matches
        self.relevant_area_one = relevant_area_one
        self.relevant_area_two = relevant_area_two
        self._mask_img1 = None
        self._mask_img2 = None

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
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
            return super()._find_homography(img1, img2)

        # Convert keypoints to an argument for findHomography
        dst_pts = np.array([kpts_img1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        src_pts = np.array([kpts_img2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

        matches = sorted(matches, key=lambda m: m.distance, reverse=True)[:40]
        matches_img = cv2.drawMatches(
            img1, kpts_img1, img2, kpts_img2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("data/out/matches_img.jpg", matches_img)

        # Establish a homography
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        return Homography(h)
