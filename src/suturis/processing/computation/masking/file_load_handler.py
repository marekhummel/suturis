import logging as log

import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask


class FileLoadHandler(BaseMaskingHandler):
    """File mask handler, meaning mask will simply be loaded from a .npy. file.
    Note that this should only be used if the homography is constant too due to the changing image dimensions."""

    _loaded_mask: Mask

    def __init__(
        self,
        invert: bool = False,
        path: str = "data/out/debug/mask.npy",
    ):
        """Creates new file load handler.

        Parameters
        ----------
        invert : bool, optional
            If set, the mask will be inverted before applying, by default False
        path : str, optional
            Path to the mask file, by default "data/out/debug/mask.npy"
        """

        log.debug("Init File Load Handler")
        super().__init__(False, False, invert)
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
