import numpy as np
from suturis.typing import CvSize, Image, Mask, Point, TranslationVector
import logging as log


class BaseMaskingHandler:
    continous_recomputation: bool
    _cached_mask = Mask | None

    def __init__(self, continous_recomputation: bool):
        self.continous_recomputation = continous_recomputation
        self._cached_mask = None

    def compute_mask(
        self,
        img1: Image,
        img2: Image,
        target_size: CvSize,
        translation: TranslationVector,
        crop_area: tuple[Point, Point],
    ) -> Mask:
        log.debug("Find mask")
        if self.continous_recomputation or self._cached_mask is None:
            log.debug("Recomputation of mask is requested")
            self._cached_mask = self._compute_mask(img1, img2, target_size, translation, crop_area)

        return self._cached_mask

    def _compute_mask(
        self,
        img1: Image,
        img2: Image,
        target_size: CvSize,
        translation: TranslationVector,
        crop_area: tuple[Point, Point],
    ) -> Mask:
        raise NotImplementedError("Abstract method needs to be overriden")

    def apply_mask(self, img1: Image, img2: Image, mask: Mask) -> Image:
        img1_masked = img1.astype(np.float64) * mask
        img2_masked = img2.astype(np.float64) * (1 - mask)
        return img1_masked + img2_masked
