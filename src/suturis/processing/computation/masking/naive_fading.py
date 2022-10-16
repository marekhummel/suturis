import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask


class NaiveFading(BaseMaskingHandler):
    """Simple masking handler which just overlays both images with half transperancy."""

    def __init__(self, **kwargs: Any):
        """Create new naive fading handler.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init Naive Fading Handler")

        if "continous_recomputation" in kwargs:
            log.warn("continous_recomputation flag in config will be ignored and overwritten with False")
        if "invert" in kwargs:
            log.warn("invert flag in config will be ignored and overwritten with False")

        kwargs["continous_recomputation"] = False
        kwargs["invert"] = False
        super().__init__(**kwargs)

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        """Creation of the mask with 50% transparency everywhere.

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
        log.debug("Return basic fading mask (50%% of each image)")
        faded_mask = np.full_like(img1, 0.5, dtype=np.float64)
        return Mask(faded_mask)
