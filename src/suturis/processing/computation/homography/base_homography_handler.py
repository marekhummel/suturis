import cv2
import numpy as np
import logging as log
from suturis.typing import CvSize, Homography, Image, NpPoint, NpSize, TranslationVector, WarpingInfo, CropArea, NpShape


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

    def find_crop(
        self, img_shape: NpShape, homography: Homography, translation: TranslationVector
    ) -> tuple[CropArea, NpSize]:
        # Define corners
        height, width = img_shape[:2]
        corners = np.array([[[0, 0]], [[0, height - 1]], [[width - 1, height - 1]], [[width - 1, 0]]], dtype=np.float32)

        # Compute corners after transformation for both img1 and img2
        tx, ty = translation
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        img1_corners_transformed = cv2.perspectiveTransform(corners, translation_matrix).astype(np.int32)
        img2_corners_transformed = cv2.perspectiveTransform(corners, translation_matrix @ homography).astype(np.int32)

        # Find min and max corner (not necessarily a corner, just min/max x and y as a point) in each image
        img1_corners_min = np.min(img1_corners_transformed, axis=0)
        img1_corners_max = np.max(img1_corners_transformed, axis=0)
        img2_corners_min = np.min(img2_corners_transformed, axis=0)
        img2_corners_max = np.max(img2_corners_transformed, axis=0)

        # For left top use the max of the min, for right bot use min of max
        x_start, y_start = np.max(np.concatenate([img1_corners_min, img2_corners_min]), axis=0)
        x_end, y_end = np.min(np.concatenate([img1_corners_max, img2_corners_max]), axis=0)
        crop_size = NpSize((y_end - y_start + 1, x_end - x_start + 1))
        return (NpPoint((y_start, x_start)), NpPoint((y_end, x_end))), crop_size

    def apply_transformations(
        self,
        img1: Image,
        img2: Image,
        translation: TranslationVector,
        canvas_size: CvSize,
        homography_matrix: Homography,
    ) -> tuple[Image, Image]:
        target_width, target_height = canvas_size
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
        img2_warped = cv2.warpPerspective(img2, translation_matrix @ homography_matrix, canvas_size)

        # Return images with same sized canvas for easy overlay
        return Image(img1_translated), Image(img2_warped)

    def apply_crop(self, img1: Image, img2: Image, start: NpPoint, end: NpPoint) -> tuple[Image, Image]:
        ystart, xstart = start
        yend, xend = end

        img1_crop = img1[ystart : yend + 1, xstart : xend + 1, :]
        img2_crop = img2[ystart : yend + 1, xstart : xend + 1, :]

        return Image(img1_crop), Image(img2_crop)
