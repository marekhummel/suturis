import logging as log

import cv2
import numpy as np
import numpy.typing as npt
from suturis.processing.computation.preprocessing.base_preprocessor import BasePreprocessor
from suturis.typing import CvRect, Image


class TextRemoval(BasePreprocessor):
    text_areas_one: list[CvRect]
    text_areas_two: list[CvRect]
    _mask_img1: npt.NDArray | None
    _mask_img2: npt.NDArray | None

    def __init__(
        self,
        index: int,
        /,
        needed_for_computation: bool = False,
        *,
        text_areas_one: list[CvRect] | None = None,
        text_areas_two: list[CvRect] | None = None,
    ) -> None:
        log.debug(f"Init Text Removal preprocessor at index #{index}")
        super().__init__(index, needed_for_computation)

        self.text_areas_one = text_areas_one or []
        self.text_areas_two = text_areas_two or []
        self._mask_img1 = None
        self._mask_img2 = None

    def process(self, img1: Image, img2: Image) -> tuple[Image, Image]:
        if len(self.text_areas_one) > 0 and self._mask_img1 is None:
            self._mask_img1 = np.zeros(shape=img1.shape[:2], dtype=np.uint8)
            for (xs, ys), (xe, ye) in self.text_areas_one:
                self._mask_img1[ys : ye + 1, xs : xe + 1] = 255

        if len(self.text_areas_two) > 0 and self._mask_img2 is None:
            self._mask_img2 = np.zeros(shape=img2.shape[:2], dtype=np.uint8)
            for (xs, ys), (xe, ye) in self.text_areas_one:
                self._mask_img2[ys : ye + 1, xs : xe + 1] = 255

        return self._remove_text(img1, self._mask_img1), self._remove_text(img2, self._mask_img2)

    def _remove_text(self, img: Image, mask: npt.NDArray | None) -> Image:
        return cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA) if mask is not None else img
