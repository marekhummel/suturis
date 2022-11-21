import logging as log
from typing import Any

import cv2
import numpy as np

from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.timer import track_timings
from suturis.typing import Image, Mask

GAUSS_BLUR_SIZE = 27


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
        mask = np.zeros_like(img1, dtype=np.float32)

        if self.vertical:
            mask[:, self.index :, :] = 1
        else:
            mask[self.index :, :, :] = 1

        mask = cv2.GaussianBlur(mask, (GAUSS_BLUR_SIZE, GAUSS_BLUR_SIZE), 0, borderType=cv2.BORDER_REPLICATE)
        return Mask(mask)

    @track_timings(name="Mask Application")
    def apply_mask(self, img1: Image, img2: Image, mask: Mask) -> Image:
        """Applies mask to transformed images to create stitched result.

        Parameters
        ----------
        img1 : Image
            First input image, transformed and cropped
        img2 : Image
            Second input image, transformed and cropped
        mask : Mask
            The mask to use. Values correspond to the percentage to be used of first image.

        Returns
        -------
        Image
            Stitched image created by the mask.
        """
        log.debug("Apply horizontal seam carving mask to images")

        # Custom method, because big areas can easily be copied
        first = self.index - GAUSS_BLUR_SIZE
        second = self.index + GAUSS_BLUR_SIZE
        final = np.zeros_like(mask)

        first_img = img1 if self.invert else img2
        second_img = img2 if self.invert else img1

        if self.vertical:
            # Copy whole areas
            final[:, :first] = first_img[:, :first]
            final[:, second:] = second_img[:, second:]

            # Only compute around seam
            mask_section = mask[:, first:second]
            img1_section = img1[:, first:second] * mask_section
            img2_section = img2[:, first:second] * (1 - mask_section)
            final[:, first:second] = img1_section + img2_section
        else:
            # Copy whole areas
            final[:first, :] = first_img[:first, :]
            final[second:, :] = second_img[second:, :]

            # Only compute around seam
            mask_section = mask[first:second, :]
            img1_section = img1[first:second, :] * mask_section
            img2_section = img2[first:second, :] * (1 - mask_section)
            final[first:second, :] = img1_section + img2_section

        return Image(final.astype(np.uint8))
