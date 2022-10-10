import logging as log

import numpy as np
from suturis.typing import Image, Mask


class BaseMaskingHandler:
    continous_recomputation: bool
    _cached_mask: Mask | None

    def __init__(self, continous_recomputation: bool):
        self.continous_recomputation = continous_recomputation
        self._cached_mask = None

    def compute_mask(self, img1: Image, img2: Image) -> Mask:
        assert img1.shape[:2] == img2.shape[:2]

        log.debug("Find mask")
        if self.continous_recomputation or self._cached_mask is None:
            log.debug("Recomputation of mask is requested")
            self._cached_mask = self._compute_mask(img1, img2)

        return self._cached_mask

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        raise NotImplementedError("Abstract method needs to be overriden")

    def apply_mask(self, img1: Image, img2: Image, mask: Mask) -> Image:
        img1_masked = img1.astype(np.float64) * mask
        img2_masked = img2.astype(np.float64) * (1 - mask)
        final = (img1_masked + img2_masked).astype(np.uint8)
        return Image(final)
