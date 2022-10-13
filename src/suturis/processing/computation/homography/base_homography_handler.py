import logging as log

import cv2
import numpy as np
from suturis.typing import CanvasInfo, CanvasSize, Homography, Image, NpPoint, NpShape, TranslationVector


class BaseHomographyHandler:
    continous_recomputation: bool
    save_to_file: bool
    _cached_homography: Homography | None

    def __init__(self, continous_recomputation: bool, save_to_file: bool):
        log.debug(
            f"Init homography handler, with continous recomputation set to {continous_recomputation}"
            "and file output set to {save_to_file}"
        )
        self.continous_recomputation = continous_recomputation
        self.save_to_file = save_to_file
        self._cached_homography = None

    def find_homography(self, img1: Image, img2: Image) -> Homography:
        log.debug("Find homography")

        if self.continous_recomputation or self._cached_homography is None:
            log.debug("Recomputation of homography is requested")
            self._cached_homography = self._find_homography(img1, img2)

            if self.save_to_file:
                log.debug("Save computed mask to file")
                np.save("data/out/homography.npy", self._cached_homography, allow_pickle=False)

        return self._cached_homography

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        raise NotImplementedError("Abstract method needs to be overriden")

    def analyze_transformed_canvas(self, img_shape: NpShape, homography: Homography) -> CanvasInfo:
        log.debug("Find dimensions of transformed image space")

        # Set corners
        height, width = img_shape[:2]
        corners_basic = [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]
        corners = np.array(corners_basic, dtype=np.float32).reshape(4, 1, 2)

        # Transform second image corners with homography
        corners_homography = cv2.perspectiveTransform(corners, homography)

        # Find min and max of all corners
        all_corners = np.concatenate((corners, corners_homography), axis=0)
        x_min, y_min = np.around(all_corners.min(axis=0).ravel()).astype(np.int32)
        x_max, y_max = np.around(all_corners.max(axis=0).ravel()).astype(np.int32)

        # Compute translation and canvas size
        translation = TranslationVector((-x_min, -y_min))
        canvas_size = CanvasSize((x_max - x_min + 1, y_max - y_min + 1))
        log.debug(f"Computed translation {translation} and canvas size {canvas_size}")

        # Apply transformation to find crop
        img1_corners_transformed = corners + np.array(translation)
        img2_corners_transformed = corners_homography + np.array(translation)

        # Find min and max corner (not necessarily a corner, just min/max x/y as a point) in each image
        img1_corners_min = np.min(img1_corners_transformed, axis=0)
        img1_corners_max = np.max(img1_corners_transformed, axis=0)
        img2_corners_min = np.min(img2_corners_transformed, axis=0)
        img2_corners_max = np.max(img2_corners_transformed, axis=0)

        # For left top use the max of the min, for right bot use min of max
        min_corners = np.concatenate([img1_corners_min, img2_corners_min])
        max_corners = np.concatenate([img1_corners_max, img2_corners_max])
        x_start, y_start = np.floor(np.max(min_corners, axis=0)).astype(np.int32)
        x_end, y_end = np.ceil(np.min(max_corners, axis=0)).astype(np.int32)
        crop_area = NpPoint((y_start, x_start)), NpPoint((y_end, x_end))
        log.debug(f"Computed crop area from {crop_area[0]} to {crop_area[1]}")

        # Return
        return canvas_size, translation, crop_area

    def apply_transformations(
        self,
        img1: Image,
        img2: Image,
        canvas_size: CanvasSize,
        translation: TranslationVector,
        homography_matrix: Homography,
    ) -> tuple[Image, Image]:
        log.debug("Apply transformations to input images")
        target_width, target_height = canvas_size
        tx, ty = translation

        # Translate image 1 (could apply warpPerspective or warpAffine as well with respective
        # translation matrices but this is faster)
        img1_height, img1_width = img1.shape[:2]
        img1_translated = np.zeros(shape=(target_height, target_width, 3), dtype=np.uint8)
        img1_translated[ty : img1_height + ty, tx : img1_width + tx] = img1

        # Perspective transform on image 2
        # This warp is the reason for the new image size, but will create negative pixel coordinates,
        # hence the translation
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        img2_warped = cv2.warpPerspective(img2, translation_matrix @ homography_matrix, canvas_size)

        # Return images with same sized canvas for easy overlay
        return Image(img1_translated), Image(img2_warped)

    def apply_crop(self, img1: Image, img2: Image, start: NpPoint, end: NpPoint) -> tuple[Image, Image]:
        log.debug("Crop transformed images to computed area")
        ystart, xstart = start
        yend, xend = end

        img1_crop = img1[ystart : yend + 1, xstart : xend + 1, :]
        img2_crop = img2[ystart : yend + 1, xstart : xend + 1, :]

        return Image(img1_crop), Image(img2_crop)
