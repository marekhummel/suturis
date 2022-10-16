import logging as log
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import CvRect, Homography, Image


class OptimizedSiftHandler(BaseHomographyHandler):
    """Homography handler which uses SIFT and RANSAC to find features in masked image and
    filters matches intelligently
    """

    sift_features: int
    min_matches: int
    relevant_areas_one: list[CvRect]
    relevant_areas_two: list[CvRect]
    enable_debug_output: bool
    _mask_img1: npt.NDArray | None
    _mask_img2: npt.NDArray | None

    def __init__(
        self,
        sift_features: int = 50000,
        min_matches: int = 10,
        relevant_areas_one: list[CvRect] | None = None,
        relevant_areas_two: list[CvRect] | None = None,
        enable_debug_output: bool = False,
        **kwargs: Any,
    ):
        """_summary_

        Parameters
        ----------
        sift_features : int, optional
            Number of max features for SIFT instance, by default 50000
        min_matches : int, optional
            Number of min found matches, otherwise identity will be returned, by default 10
        relevant_areas_one : list[CvRect] | None, optional
            List of rectangles, which defines the set of areas used for feature finding in first image, by default []
        relevant_areas_two : list[CvRect] | None, optional
            List of rectangles, which defines the set of areas used for feature finding in second image, by default []
        enable_debug_output : bool, optional
            If true, computed keypoints and matches will be visualized and saved to "data/out/debug/osift_*.jpg",
            by default False
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug(
            f"Init Optimized SIFT Homography Handler with {sift_features} features, {min_matches} min matches and "
            f"{len(relevant_areas_one or [])} / {len(relevant_areas_two or [])} defined relevant areas"
        )
        super().__init__(**kwargs)
        self.sift_features = sift_features
        self.min_matches = min_matches
        self.relevant_areas_one = relevant_areas_one or []
        self.relevant_areas_two = relevant_areas_two or []
        self._mask_img1 = None
        self._mask_img2 = None
        self.enable_debug_output = enable_debug_output

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        """Algorithm to find homography with SIFT.
        Matches are filtered by descriptor distance (obviously) but also by spatial distance.

        Parameters
        ----------
        img1 : Image
            First input image (query)
        img2 : Image
            Second input image (train)

        Returns
        -------
        Homography
            Homography to warp the second image onto the first, based on SIFT features
        """
        log.debug("Compute new homography with SIFT for images")
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
        log.debug("Find features")
        sift = cv2.SIFT_create(nfeatures=self.sift_features)
        kpts_img1, descs_img1 = sift.detectAndCompute(img1, mask=self._mask_img1)  # queryImage
        kpts_img2, descs_img2 = sift.detectAndCompute(img2, mask=self._mask_img2)  # trainImage

        # Match and return default if not enough matches
        log.debug("Match features and filter by descriptor distance and spatial distance")
        bfm = cv2.BFMatcher_create(cv2.NORM_L2)
        matches = bfm.match(descs_img1, descs_img2)
        good_matches = self._filter_good_matches(matches, kpts_img1, kpts_img2)

        # Write debug images
        if self.enable_debug_output:
            log.debug("Write debug images (keypoints and matches)")
            self._output_debug_images(img1, img2, kpts_img1, kpts_img2, good_matches)

        # Return identity if not enough matches
        if len(good_matches) < self.min_matches:
            log.debug("Not enough matches found, return identity")
            return Homography(np.identity(3, dtype=np.float64))

        # Convert keypoints to an argument for findHomography
        dst_pts = np.array([kpts_img1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
        src_pts = np.array([kpts_img2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

        # Establish a homography
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        log.debug(f"Computed homography {h.tolist()}")
        return Homography(h)

    def _filter_good_matches(
        self, matches: list[cv2.DMatch], kpts_img1: list[cv2.KeyPoint], kpts_img2: list[cv2.KeyPoint]
    ) -> list[cv2.DMatch]:
        """Find relevant matches. Not only have the descriptors to be similar,
        but the keypoints need to be close to each other.

        Parameters
        ----------
        matches : list
            List of matches found by a brute force matcher
        kpts_img1 : list
            List of keypoints in first image
        kpts_img2 : list
            List of keypoints in second image

        Returns
        -------
        list
            Subset of given matches, filtered by criteria mentioned above.
        """

        def loc_distance(m: cv2.DMatch) -> float:
            pt1 = kpts_img1[m.queryIdx].pt
            pt2 = kpts_img2[m.trainIdx].pt
            return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2

        max_loc_distance = max(loc_distance(m) for m in matches)
        max_desc_distance = max(m.distance for m in matches)

        def comb_distance(m: cv2.DMatch) -> float:
            return (2 * m.distance / max_desc_distance + loc_distance(m) / max_loc_distance) / 3

        return [m for m in matches if comb_distance(m) < 0.25]

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
            Set of matches found by SIFT and a brute force matcher
        """

        matches_img = cv2.drawMatches(
            img1, kpts_img1, img2, kpts_img2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("data/out/debug/osift_matches.jpg", matches_img)

        kpts1_img = cv2.drawKeypoints(img1, kpts_img1, None, color=(72, 144, 233))
        cv2.imwrite("data/out/debug/osift_keypoints1.jpg", kpts1_img)

        kpts2_img = cv2.drawKeypoints(img2, kpts_img2, None, color=(72, 144, 233))
        cv2.imwrite("data/out/debug/osift_keypoints2.jpg", kpts2_img)
