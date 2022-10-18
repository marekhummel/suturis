import logging as log
from typing import Any

import cv2
import numpy as np
from suturis.processing.computation.preprocessing.base_preprocessor import BasePreprocessor
from suturis.typing import Image, ImagePair


class DebugColoring(BasePreprocessor):
    """Preprocessor which adds slight coloring to images for easier distiction."""

    color_img1: tuple[int, int, int]
    color_img2: tuple[int, int, int]

    def __init__(
        self,
        *args: Any,
        color_img1: tuple[int, int, int] = (0, 0, 127),
        color_img2: tuple[int, int, int] = (127, 0, 0),
        **kwargs: Any,
    ) -> None:
        """Create new debug coloring preprocessor.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        color_img1 : tuple[int, int, int], optional
            BGR color to use for first image, by default (0, 0, 127)
        color_img2 : tuple[int, int, int], optional
            BGR color to use for second image, by default (127, 0, 0)
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init Debug Coloring preprocessor")
        super().__init__(*args, **kwargs)

        self.color_img1 = color_img1
        self.color_img2 = color_img2

    def process(self, img1: Image, img2: Image) -> ImagePair:
        """Add faded colors to images.

        Parameters
        ----------
        img1 : Image
            First input image
        img2 : Image
            Second input image

        Returns
        -------
        ImagePair
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
        return Image(colored_img.astype(np.uint8))
