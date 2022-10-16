import logging as log

import cv2
import numpy as np
from suturis.typing import CanvasInfo, CanvasSize, Homography, Image, NpPoint, NpShape, TranslationVector


class BaseHomographyHandler:
    """Base class for homography computation."""

    continous_recomputation: bool
    save_to_file: bool
    disable_cropping: bool
    _cached_homography: Homography | None

    def __init__(self, continous_recomputation: bool, save_to_file: bool = False, disable_cropping: bool = False):
        """Create new base homography handler instance, should not be called explicitly only from subclasses.

        Parameters
        ----------
        continous_recomputation : bool
            If set, homography will be recomputed each time, otherwise the first result will be reused
        save_to_file : bool, optional
            If set, the homography matrix will be saved to a .npy file in "data/out/matrix/", by default False
        disable_cropping : bool, optional
            If set, the target canvas won't be cropped to the relevant parts (this will likely create black areas),
            by default False
        """
        log.debug(
            f"Init homography handler, with continous recomputation set to {continous_recomputation}, "
            f"file output set to {save_to_file} and disabled cropping set to {disable_cropping}"
        )
        self.continous_recomputation = continous_recomputation
        self.save_to_file = save_to_file
        self.disable_cropping = disable_cropping
        self._cached_homography = None

    def find_homography(self, img1: Image, img2: Image) -> Homography:
        """Return homography for input images, recomputed if needed.

        Parameters
        ----------
        img1 : Image
            First input image (source)
        img2 : Image
            Second input image (destination)

        Returns
        -------
        Homography
            The homography computed for those images, or the cached one
        """
        log.debug("Find homography")

        if self.continous_recomputation or self._cached_homography is None:
            log.debug("Recomputation of homography is requested")
            self._cached_homography = self._find_homography(img1, img2)

            if self.save_to_file:
                log.debug("Save computed homography to file")
                np.save("data/out/matrix/homography.npy", self._cached_homography, allow_pickle=False)

        return self._cached_homography

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        """Abstract method to compute homography.

        Parameters
        ----------
        img1 : Image
            First input image (source)
        img2 : Image
            Second input image (destination)

        Returns
        -------
        Homography
            The homography computed for those images

        Raises
        ------
        NotImplementedError
            Unless overriden, this method will raise an error.
        """
        raise NotImplementedError("Abstract method needs to be overriden")

    def analyze_transformed_canvas(self, img_shape: NpShape, homography: Homography) -> CanvasInfo:
        """Analyzes the dimensions of the input and the homography, to compute dimensions and crop area in target space.

        Parameters
        ----------
        img_shape : NpShape
            Numpy shape of the input images
        homography : Homography
            Homography which will be applied to warp the images

        Returns
        -------
        CanvasInfo
            Set of information about the target space. Includes the necessary translation to avoid negative coordinates,
            the size of the target space, and the rectangle to which the target space can be cropped to.
        """
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

        if not self.disable_cropping:
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
        else:
            # Crop area == Canvas size
            width, height = canvas_size
            crop_area = NpPoint((0, 0)), NpPoint((height + 1, width + 1))

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
        """Applies computed transformations to the source images.

        Parameters
        ----------
        img1 : Image
            First input image
        img2 : Image
            Second input image
        canvas_size : CanvasSize
            Dimensions of the target space
        translation : TranslationVector
            Translation needed to move out of negative coordinates
        homography_matrix : Homography
            Homography which warpes the second image

        Returns
        -------
        tuple[Image, Image]
            Transformed images
        """
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
        """Applies cropping to transformed images.

        Parameters
        ----------
        img1 : Image
            First image, transformed (only translation here)
        img2 : Image
            Second image, transformed (translation + warping)
        start : NpPoint
            Top left corner of the crop rectangle
        end : NpPoint
            Bottom right corner of the crop rectabngle

        Returns
        -------
        tuple[Image, Image]
            Cropped images
        """
        log.debug("Crop transformed images to computed area")
        ystart, xstart = start
        yend, xend = end

        img1_crop = img1[ystart : yend + 1, xstart : xend + 1, :]
        img2_crop = img2[ystart : yend + 1, xstart : xend + 1, :]

        return Image(img1_crop), Image(img2_crop)
