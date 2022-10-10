import logging as log

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class NoWarpingHandler(BaseHomographyHandler):
    def __init__(self, continous_recomputation: bool):
        log.debug("Init No Warping Handler")
        super().__init__(continous_recomputation)

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        return Homography(np.identity(3, dtype=np.float64))
