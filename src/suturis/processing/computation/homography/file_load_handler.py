import logging as log

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class FileLoadHandler(BaseHomographyHandler):
    _loaded_homography: Homography

    def __init__(
        self, continous_recomputation: bool = False, save_to_file: bool = False, path: str = "data/out/homography.npy"
    ):
        log.debug(f"Init File Load Handler looking at {path}")
        super().__init__(continous_recomputation, save_to_file)
        self._loaded_homography = Homography(np.load(path, allow_pickle=False))

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        log.debug("Return loaded homography")
        return self._loaded_homography
