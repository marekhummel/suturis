import logging as log
from typing import Any

import cv2
import numpy as np
from suturis.processing.computation.preprocessing.base_preprocessor import BasePreprocessor
from suturis.typing import Image


class Rotation(BasePreprocessor):
    """Preprocessor which rotates images"""

    degrees_img1: float
    degrees_img2: float

    def __init__(self, *args: Any, degrees_img1: float, degrees_img2: float, **kwargs: Any) -> None:
        """Creates new rotation preprocessor.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        degrees_img1 : float
            Rotation in degrees for first image (positive means ccw)
        degrees_img2 : float
            Rotation in degrees for second image (positive means ccw)
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug(
            f"Init Rotation preprocessor with rotations of "
            f"{degrees_img1}deg for image one and {degrees_img2}deg for image two"
        )
        super().__init__(*args, **kwargs)

        self.degrees_img1 = degrees_img1
        self.degrees_img2 = degrees_img2

    def process(self, img1: Image, img2: Image) -> tuple[Image, Image]:
        """Rotate images.

        Parameters
        ----------
        img1 : Image
            First input image
        img2 : Image
            Second input image

        Returns
        -------
        tuple[Image, Image]
            Rotated images.
        """
        log.debug("Rotate images")
        return self._rotate(img1, self.degrees_img1), self._rotate(img2, self.degrees_img2)

    def _rotate(self, img: Image, degrees: float) -> Image:
        """Rotate an image.

        Parameters
        ----------
        img : Image
            Image to rotate
        degrees : float
            Degrees to rotate the image by (positive means ccw)

        Returns
        -------
        Image
            Rotated image.
        """
        height, width = img.shape[:2]
        center = width // 2, height // 2
        rot_matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (width, height))

        return Image(rotated.astype(np.uint8))
