import logging as log

import cv2
import numpy as np
from suturis.processing.computation.preprocessing.base_preprocessor import BasePreprocessor
from suturis.typing import Image


class DebugColoring(BasePreprocessor):
    color_img1: tuple[int, int, int]
    color_img2: tuple[int, int, int]

    def __init__(
        self,
        index: int,
        /,
        needed_for_computation: bool = True,
        color_img1: tuple[int, int, int] = (0, 0, 127),
        color_img2: tuple[int, int, int] = (127, 0, 0),
    ) -> None:
        log.debug(f"Init Coloring preprocessor at index #{index}")
        super().__init__(index, needed_for_computation)

        self.color_img1 = color_img1
        self.color_img2 = color_img2

    def process(self, img1: Image, img2: Image) -> tuple[Image, Image]:
        return self._color(img1, self.color_img1), self._color(img2, self.color_img2)

    def _color(self, img: Image, color: tuple[int, int, int]) -> Image:
        colored_plane = np.zeros_like(img)
        colored_plane[:, :] = color

        colored_img = cv2.addWeighted(img, 0.7, colored_plane, 0.3, 0)
        return colored_img.astype(np.uint8)
