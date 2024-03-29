import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Mask


class FileLoadHandler(BaseMaskingHandler):
    """File mask handler, meaning mask will simply be loaded from a .npy. file.
    Note that this should only be used if the homography is constant too due to the changing image dimensions."""

    def __init__(self, path: str = "data/out/matrix/mask.npy", **kwargs: Any):
        """Creates new file load handler.

        Parameters
        ----------
        path : str, optional
            Path to the mask file, by default "data/out/matrix/mask.npy"
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init File Load Handler")

        if "caching_enabled" in kwargs:
            log.warning("caching_enabled flag in config will be ignored and overwritten with True")
        if "save_to_file" in kwargs:
            log.warning("save_to_file flag in config will be ignored and overwritten with False")

        kwargs["caching_enabled"] = True
        kwargs["save_to_file"] = False
        super().__init__(**kwargs)
        mask = Mask(np.load(path, allow_pickle=False))
        self._cache = mask if not self.invert else Mask(1 - mask)
