import logging as log

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class FileLoadHandler(BaseHomographyHandler):
    """File homography handler, meaning homography will simply be loaded from a .npy. file."""

    _loaded_homography: Homography

    def __init__(self, path: str = "data/out/debug/homography.npy"):
        """Creates new file load handler.

        Parameters
        ----------
        path : str, optional
            Path to the homography file, by default "data/out/debug/homography.npy"
        """
        log.debug(f"Init File Load Handler looking at {path}")
        super().__init__(False, False)
        self._loaded_homography = Homography(np.load(path, allow_pickle=False))

    def _find_homography(self, img1: Image, img2: Image) -> Homography:
        """Homography finding, just returns matrix loaded from file.

        Parameters
        ----------
        img1 : Image
            First input image (won't be used here)
        img2 : Image
            Second input image (won't be used here)

        Returns
        -------
        Homography
            Matrix loaded from file
        """
        log.debug("Return loaded homography")
        return self._loaded_homography
