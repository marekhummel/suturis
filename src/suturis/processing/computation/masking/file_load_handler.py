import logging as log

import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask


class FileLoadHandler(BaseMaskingHandler):
    _loaded_mask: Mask

    def __init__(
        self,
        continous_recomputation: bool = False,
        save_to_file: bool = False,
        invert: bool = False,
        path: str = "data/out/mask.npy",
    ):
        log.debug("Init File Load Handler")
        super().__init__(continous_recomputation, save_to_file, invert)
        self._loaded_mask = Mask(np.load(path, allow_pickle=False))

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        return self._loaded_mask
