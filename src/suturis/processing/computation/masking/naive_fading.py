import logging as log

import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask


class NaiveFading(BaseMaskingHandler):
    def __init__(self, continous_recomputation: bool, save_to_file: bool = False):
        log.debug("Init Naive Fading Handler")
        super().__init__(continous_recomputation, save_to_file)

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        faded_mask = np.full_like(img1, 0.5, dtype=np.float64)
        return Mask(faded_mask)
