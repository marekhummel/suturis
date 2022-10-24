import logging as log
from typing import Any

import cv2
import numpy as np
from suturis.processing.computation.postprocessing.base_postprocessor import BasePostprocessor
from suturis.typing import CvSize, Image, Matrix


class Rotation(BasePostprocessor[tuple[Matrix, CvSize]]):
    """Postprocessor which rotates the image"""

    angle_deg: float

    def __init__(self, *args: Any, angle_deg: float, **kwargs: Any) -> None:
        """Creates new rotation postprocessor.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        angle_deg : float
            Rotation in degrees (positive means ccw)
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug(f"Init Rotation postprocessor with rotation of {angle_deg}deg")
        super().__init__(*args, **kwargs)

        self.angle_deg = angle_deg

    def process(self, image: Image) -> Image:
        """Rotate image (and adjust array size to avoid cutting of image parts).

        Parameters
        ----------
        image: Image
            Stitched image, may be modified by previous postprocessors

        Returns
        -------
        Image
            Modified image.
        """
        log.debug("Rotate image")

        if not self._caching_enabled or self._cache is None:
            # Compute rotation matrix
            height, width = image.shape[:2]
            center = width // 2, height // 2
            rot_matrix: Matrix = cv2.getRotationMatrix2D(center, self.angle_deg, 1.0)

            # Compute new canvas size and translation
            cos_angle, sin_angle = rot_matrix[0, 0:2]
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            translation = np.array([new_width, new_height]) / 2 - np.array(center)

            # Adjust affine matrix and compute rotated image
            rot_matrix[0:2, 2] += translation

            # Store
            self._cache = rot_matrix, CvSize((new_width, new_height))

        matrix, dimensions = self._cache
        rotated = cv2.warpAffine(image, matrix, dimensions)
        return Image(rotated.astype(np.uint8))
