import logging as log

import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask


class NaiveFading(BaseMaskingHandler):
    """Simple masking handler which just overlays both images with half transperancy."""

    def __init__(self, continous_recomputation: bool = False, save_to_file: bool = False):
        """Create new naive fading handler.

        Parameters
        ----------
        continous_recomputation : bool, optional
            If set, homography will be recomputed each time, otherwise the first result will be reused, by default False
        save_to_file : bool, optional
            If set, the homography matrix will be saved to a .npy file in "data/out/matrix/", by default False
        """
        log.debug("Init Naive Fading Handler")
        super().__init__(continous_recomputation, save_to_file, False)

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
