import logging as log

import cv2
import numpy as np
from suturis.processing.computation.preprocessing.base_preprocessor import BasePreprocessor
from suturis.typing import Image


class Rotation(BasePreprocessor):
    degrees_img1: float
    degrees_img2: float

    def __init__(
        self, index: int, /, needed_for_computation: bool = True, *degrees_img1: float, degrees_img2: float
    ) -> None:
        log.debug(f"Init Rotation preprocessor at index #{index}")
        super().__init__(index, needed_for_computation)

        self.degrees_img1 = degrees_img1
        self.degrees_img2 = degrees_img2

    def process(self, img1: Image, img2: Image) -> tuple[Image, Image]:
        return self._rotate(img1, self.degrees_img1), self._rotate(img2, self.degrees_img2)

    def _rotate(self, img: Image, degrees: float) -> Image:
        height, width = img.shape[:2]
        center = width // 2, height // 2
        rot_matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (width, height))

        return rotated.astype(np.uint8)
