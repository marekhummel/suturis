import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.typing import CvSize, Homography, Matrix, TranslationVector


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

        # Load
        try:
            file = np.load(path)
            canvas_size = file["canvas"]
            translation = file["translation"]
            homography = file["homography"]
            file.close()
        except KeyError as e:
            raise KeyError("Invalid .npz file, any of canvas, translation or homography is missing") from e

        # Verify
        if type(canvas_size) is not Matrix or canvas_size.shape != (2,) or canvas_size.dtype != np.int32:
            raise ValueError("Invalid .npz file, canvas is wrongly formatted")

        if type(translation) is not Matrix or translation.shape != (2,) or translation.dtype != np.int32:
            raise ValueError("Invalid .npz file, translation is wrongly formatted")

        if type(homography) is not Matrix or homography.shape != (3, 3):
            raise ValueError("Invalid .npz file, homography is wrongly formatted")

        # Store
        canvas_size_tuple = (canvas_size[0], canvas_size[1])
        translation_tuple = (translation[0], translation[1])
        self._cache = CvSize(canvas_size_tuple), TranslationVector(translation_tuple), Homography(homography)
