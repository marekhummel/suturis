import logging as log
from typing import Any

import cv2
import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class BasicOrbHandler(BaseHomographyHandler):
    """Simple homography handler which uses ORB and RANSAC to find features"""

    orb_features: int
    min_matches: int

    def __init__(self, orb_features: int = 50000, min_matches: int = 10, **kwargs: Any):
        """Creates new basic ORB handler instance.

        Parameters
        ----------
        orb_features : int, optional
            Number of max features for ORB instance, by default 50000
        min_matches : int, optional
            Number of min found matches, otherwise identity will be returned, by default 10
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug(f"Init ORB Ransac Homography Handler with {orb_features} features and {min_matches} min matches")
        super().__init__(**kwargs)
        self.orb_features = orb_features
        self.min_matches = min_matches

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        """Algorithm to find homography with ORB.

        Parameters
        ----------
        img1 : Image
            First input image (query)
        img2 : Image
            Second input image (train)

        Returns
        -------
        Homography
            Homography to warp the second image onto the first, based on ORB features
        """
        log.debug("Compute new homography with ORB for images")
        # Create ORB and compute
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        kpts_img1, descs_img1 = orb.detectAndCompute(img1, None)
        kpts_img2, descs_img2 = orb.detectAndCompute(img2, None)

        # Match and return default if not enough matches
        bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches = bfm.knnMatch(descs_img1, descs_img2, k=2)
        good_matches = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]  # Lowes ratio test

        # Output some debug results
        if self._debugging_enabled:
            log.debug("Write debug images (keypoints and matches)")
            self._output_debug_images(img1, img2, kpts_img1, kpts_img2, good_matches)

        # Return identity if not enough matches
        if len(good_matches) <= self.min_matches:
            log.debug("Not enough matches found, return identity")
            return Homography(np.identity(3, dtype=np.float64))

        # Convert keypoints to an argument for findHomography
        dst_pts = np.array([kpts_img1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
        src_pts = np.array([kpts_img2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

        # Establish a homography
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        log.debug(f"Computed homography {h.tolist()}")
        return Homography(h)

    def _output_debug_images(self, img1: Image, img2: Image, kpts_img1: list, kpts_img2: list, matches: list) -> None:
        """Writes debug images to disk to visualize matches and keypoints.

        Parameters
        ----------
        img1 : Image
            First image
        img2 : Image
            Second image
        kpts_img1 : list
            Keypoints in first image
        kpts_img2 : list
            Keypoints in second image
        matches : list
            Set of matches found by ORB and a brute force matcher
        """
        matches_img = cv2.drawMatches(
            img1, kpts_img1, img2, kpts_img2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(f"{self._path}orb_matches.jpg", matches_img)

        kpts1_img = cv2.drawKeypoints(img1, kpts_img1, None, color=(72, 144, 233))
        cv2.imwrite(f"{self._path}img1_orb_keypoints.jpg", kpts1_img)

        kpts2_img = cv2.drawKeypoints(img2, kpts_img2, None, color=(72, 144, 233))
        cv2.imwrite(f"{self._path}img2_orb_keypoints.jpg", kpts2_img)
