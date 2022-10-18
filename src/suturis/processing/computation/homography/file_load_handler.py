import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import Homography, Image


class FileLoadHandler(BaseHomographyHandler):
    """File homography handler, meaning homography will simply be loaded from a .npy file."""

    _loaded_homography: Homography

    def __init__(self, path: str = "data/out/matrix/homography.npy", **kwargs: Any):
        """Creates new file load handler.

        Parameters
        ----------
        path : str, optional
            Path to the homography file, by default "data/out/matrix/homography.npy"
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug(f"Init File Load Handler looking at {path}")

        if "caching_enabled" in kwargs:
            log.warn("caching_enabled flag in config will be ignored and overwritten with False")
        if "save_to_file" in kwargs:
            log.warn("save_to_file flag in config will be ignored and overwritten with False")

        kwargs["caching_enabled"] = False
        kwargs["save_to_file"] = False
        super().__init__(**kwargs)
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
