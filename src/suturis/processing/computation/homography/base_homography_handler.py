import cv2
import numpy as np
import logging as log
from suturis.typing import CvSize, Homography, Image, NpPoint, NpSize, TranslationVector, WarpingInfo, CropArea


class BaseHomographyHandler:
    continous_recomputation: bool
    _cached_homography: WarpingInfo | None

    def __init__(self, continous_recomputation: bool):
        self.continous_recomputation = continous_recomputation
        self._cached_homography = None

    def find_homography(self, img1: Image, img2: Image) -> WarpingInfo:
        log.debug("Find homography")
        if self.continous_recomputation or self._cached_homography is None:
            log.debug("Recomputation of homography is requested")
            self._cached_homography = self._find_homography(img1, img2)
        return self._cached_homography

    def _find_homography(self, img1: Image, img2: Image) -> WarpingInfo:
        raise NotImplementedError("Abstract method needs to be overriden")

    def find_crop(self, img1: Image, homography: Homography, translation: TranslationVector) -> tuple[CropArea, NpSize]:
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

        crop_size = (end[0] - start[0] + 1, end[1] - start[1] + 1)
        return (start, end), crop_size

    def apply_transformations(
        self,
        img1: Image,
        img2: Image,
        translation: TranslationVector,
        target_size: CvSize,
        homography_matrix: Homography,
    ) -> tuple[Image, Image]:
        target_width, target_height = target_size
        tx, ty = translation

        # Translate image 1 (could apply warpPerspective or warpAffine as well with respective
        # translation matrices but this is faster)
        img1_height, img1_width = img1.shape[:2]
        img1_translated = np.zeros(shape=(target_height, target_width, 3))
        img1_translated[ty : img1_height + ty, tx : img1_width + tx] = img1

        # Perspective transform on image 2
        # This warp is the reason for the new image size, but will create negative pixel coordinates,
        # hence the translation
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        img2_warped = cv2.warpPerspective(img2, translation_matrix @ homography_matrix, target_size)

        # Return images with same sized canvas for easy overlay
        return img1_translated, img2_warped

    def apply_crop(self, img1: Image, img2: Image, start: NpPoint, end: NpPoint) -> tuple[Image, Image]:
        ystart, xstart = start
        yend, xend = end

        img1_crop = img1[ystart : yend + 1, xstart : xend + 1, :]
        img2_crop = img2[ystart : yend + 1, xstart : xend + 1, :]

        return img1_crop, img2_crop
