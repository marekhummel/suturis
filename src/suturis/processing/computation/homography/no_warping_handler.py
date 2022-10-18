import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class NoWarpingHandler(BaseHomographyHandler):
    """Dummy homography handler, just returns identity."""

    def __init__(self, **kwargs: Any):
        """Creates new no warping handler.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init No Warping Handler")

        if "continous_recomputation" in kwargs:
            log.warn("continous_recomputation flag in config will be ignored and overwritten with False")

        kwargs["continous_recomputation"] = False
        super().__init__(**kwargs)

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
