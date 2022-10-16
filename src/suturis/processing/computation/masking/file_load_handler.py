import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask


class FileLoadHandler(BaseMaskingHandler):
    """File mask handler, meaning mask will simply be loaded from a .npy. file.
    Note that this should only be used if the homography is constant too due to the changing image dimensions."""

    _loaded_mask: Mask

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

        if "continous_recomputation" in kwargs:
            log.warn("continous_recomputation flag in config will be ignored and overwritten with False")
        if "save_to_file" in kwargs:
            log.warn("save_to_file flag in config will be ignored and overwritten with False")

        kwargs["continous_recomputation"] = False
        kwargs["save_to_file"] = False
        super().__init__(**kwargs)
        self._loaded_mask = Mask(np.load(path, allow_pickle=False))

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        """Mask finding, just returns matrix loaded from file.

        Parameters
        ----------
        img1 : Image
            First input image (won't be used here)
        img2 : Image
            Second input image (won't be used here)

        Returns
        -------
        Mask
            Matrix loaded from file
        """
        log.debug("Return loaded mask")
        return self._loaded_mask
