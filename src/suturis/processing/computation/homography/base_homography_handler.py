import logging as log
from typing import Any

import cv2
import numpy as np

from suturis.processing.computation.base_computation_handler import BaseComputationHandler
from suturis.timer import track_timings
from suturis.typing import (
    CanvasInfo,
    CanvasSize,
    Homography,
    Image,
    ImagePair,
    NpShape,
    TransformationInfo,
    TranslationVector,
)


class BaseHomographyHandler(BaseComputationHandler[TransformationInfo]):
    """Base class for homography computation."""

    save_to_file: bool

    def __init__(self, save_to_file: bool = False, **kwargs: Any):
        """Create new base homography handler instance, should not be called explicitly only from subclasses.

        Parameters
        ----------
        save_to_file : bool, optional
            If set, the homography matrix will be saved to a .npy file in "data/out/matrix/", by default False
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug(f"Init homography handler file output set to {save_to_file}")
        super().__init__(**kwargs)
        self.save_to_file = save_to_file

    def find_transformation(self, img1: Image, img2: Image) -> TransformationInfo:
        """Return homography for input images, recomputed if needed.

        Parameters
        ----------
        img1 : Image
            First input image (source)
        img2 : Image
            Second input image (destination)

        Returns
        -------
        TransformationInfo
            Full transformation parameters (canvas size, translation and homography), or the cached ones
        """
        log.debug("Find homography")

        if not self._caching_enabled or self._cache is None:
            log.debug("Recomputation of homography is requested")
            homography = self._find_homography(img1, img2)
            canvas_size, translation = self._analyze_transformed_canvas(img1.shape, homography)
            self._cache = canvas_size, translation, homography

            if self.save_to_file:
                log.debug("Save computed transformation params to file")
                np.savez(
                    "data/out/matrix/homography.npz", canvas=canvas_size, translation=translation, homography=homography
                )
        return self._cache

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

    def _analyze_transformed_canvas(self, img_shape: NpShape, homography: Homography) -> CanvasInfo:
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

        # Return
        return canvas_size, translation

    @track_timings(name="Homography Application")
    def apply_transformations(self, img1: Image, img2: Image, transformation_info: TransformationInfo) -> ImagePair:
        """Applies computed transformations to the source images.

        Parameters
        ----------
        img1 : Image
            First input image
        img2 : Image
            Second input image
        transformation_info : TransformationInfo
            Parameters needed to apply transformation (canvas size, translation and homography)

        Returns
        -------
        ImagePair
            Transformed images
        """
        log.debug("Apply transformations to input images")
        canvas_size, translation, homography_matrix = transformation_info
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
