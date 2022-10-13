import logging as log

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class NoWarpingHandler(BaseHomographyHandler):
    def __init__(self, continous_recomputation: bool = False, save_to_file: bool = False):
        log.debug("Init No Warping Handler")
        super().__init__(continous_recomputation, save_to_file)

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        log.debug("Return identity as homography")
        return Homography(np.identity(3, dtype=np.float64))
