import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import CvSize, Homography, TranslationVector


class FileLoadHandler(BaseHomographyHandler):
    """File homography handler, meaning transformation info will simply be loaded from a .npz file."""

    def __init__(self, path: str = "data/out/matrix/homography.npz", **kwargs: Any):
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
            log.warning("caching_enabled flag in config will be ignored and overwritten with True")
        if "save_to_file" in kwargs:
            log.warning("save_to_file flag in config will be ignored and overwritten with False")

        kwargs["caching_enabled"] = True
        kwargs["save_to_file"] = False
        super().__init__(**kwargs)

        file = np.load(path)
        canvas_size = file["canvas"]
        translation = file["translation"]
        homography = file["homography"]
        file.close()

        self._cache = CvSize(tuple(canvas_size)), TranslationVector(tuple(translation)), Homography(homography)
