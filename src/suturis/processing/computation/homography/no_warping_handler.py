import logging as log

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class NoWarpingHandler(BaseHomographyHandler):
    """Dummy homography handler, just returns identity."""

    def __init__(self, save_to_file: bool = False):
        """Creates new no warping handler.

        Parameters
        ----------
        save_to_file : bool, optional
            If set, the homography matrix will be saved to a .npy file in "data/out/debug", by default False
        """
        log.debug("Init No Warping Handler")
        super().__init__(False, save_to_file)

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        """Homography finding, just retuns identity.

        Parameters
        ----------
        img1 : Image
            First input image (won't be used here)
        img2 : Image
            Second input image (won't be used here)

        Returns
        -------
        Homography
            Identity matrix
        """

        log.debug("Return identity as homography")
        return Homography(np.identity(3, dtype=np.float64))
