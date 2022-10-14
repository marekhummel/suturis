import logging as log

import cv2
import numpy as np
from suturis.processing.computation.preprocessing.base_preprocessor import BasePreprocessor
from suturis.typing import Image


class DebugColoring(BasePreprocessor):
    """Preprocessor which adds slight coloring to images for easier debugging."""

    color_img1: tuple[int, int, int]
    color_img2: tuple[int, int, int]

    def __init__(
        self,
        index: int,
        /,
        needed_for_computation: bool = False,
        color_img1: tuple[int, int, int] = (0, 0, 127),
        color_img2: tuple[int, int, int] = (127, 0, 0),
    ) -> None:
        """Create new debug coloring preprocessor.

        Parameters
        ----------
        index : int
            0-based index of this preprocessor. Given implicitly by list index in config
        needed_for_computation : bool, optional
            Flag to indicate of this preprocessor should be used for computation, by default False
        color_img1 : tuple[int, int, int], optional
            BGR color to use for first image, by default (0, 0, 127)
        color_img2 : tuple[int, int, int], optional
            BGR color to use for second image, by default (127, 0, 0)
        """
        log.debug(f"Init Coloring preprocessor at index #{index}")
        super().__init__(index, needed_for_computation)

        self.color_img1 = color_img1
        self.color_img2 = color_img2

    def process(self, img1: Image, img2: Image) -> tuple[Image, Image]:
        """Add faded colors to images.

        Parameters
        ----------
        img1 : Image
            First input image
        img2 : Image
            Second input image

        Returns
        -------
        tuple[Image, Image]
            Tainted images.
        """
        log.debug("Add debugging colors to images")
        return self._color(img1, self.color_img1), self._color(img2, self.color_img2)

    def _color(self, img: Image, color: tuple[int, int, int]) -> Image:
        """Colorize image.

        Parameters
        ----------
        img : Image
            Image to colorize
        color : tuple[int, int, int]
            Color to use

        Returns
        -------
        Image
            Tainted image.
        """
        colored_plane = np.zeros_like(img)
        colored_plane[:, :] = color

        colored_img = cv2.addWeighted(img, 0.7, colored_plane, 0.3, 0)
        return colored_img.astype(np.uint8)
