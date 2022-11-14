import logging as log
from typing import Any

import cv2
import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask


class StraightSeam(BaseMaskingHandler):
    """Simple straight seam (either vertical or horizontal)."""

    index: int
    vertical: bool

    def __init__(self, *, index: int, vertical: bool = False, **kwargs: Any):
        """Create simple straight seam as a mask

        Parameters
        ----------
        index : int
            Index where to place seam (x when vertical, y when horizontal)
        vertical : bool, optional
            If set, the seam will be vertical instead of horizontal, by default False
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init Naive Fading Handler")

        super().__init__(**kwargs)
        self.index = index
        self.vertical = vertical

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        """Creation of the mask which just splits the mask in two halves.

        Parameters
        ----------
        img1 : Image
            Transformed and cropped first image
        img2 : Image
            Transformed and cropped second image

        Returns
        -------
        Mask
            The mask matrix used to combine the images.
        """
        log.debug("Return straight seam mask")
        mask = np.zeros_like(img1, dtype=np.float64)

        if self.vertical:
            mask[:, self.index :, :] = 1
        else:
            mask[self.index :, :, :] = 1

        mask = cv2.GaussianBlur(mask, (27, 27), 0, borderType=cv2.BORDER_REPLICATE)
        return Mask(mask)
