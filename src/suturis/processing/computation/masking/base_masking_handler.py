import logging as log

import numpy as np
from suturis.typing import Image, Mask


class BaseMaskingHandler:
    """Base class for mask computation."""

    continous_recomputation: bool
    save_to_file: bool
    invert: bool
    _cached_mask: Mask | None

    def __init__(self, continous_recomputation: bool, save_to_file: bool, invert: bool):
        """Create new base mask handler instance, should not be called explicitly only from subclasses.

        Parameters
        ----------
        continous_recomputation : bool
            If set, homography will be recomputed each time, otherwise the first result will be reused
        save_to_file : bool
            If set, the homography matrix will be saved to a .npy file in "data/out/matrix/"
        invert : bool
            If set, the mask will be inverted before applying
        """
        log.debug(
            f"Init masking handler, with continous recomputation set to {continous_recomputation}, "
            f"file output set to {save_to_file} and invert set to {invert}"
        )
        self.continous_recomputation = continous_recomputation
        self.save_to_file = save_to_file
        self.invert = invert
        self._cached_mask = None

    def compute_mask(self, img1: Image, img2: Image) -> Mask:
        """Return mask for (transformed and cropped) input images, recomputed if needed.

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
        assert img1.shape[:2] == img2.shape[:2]

        log.debug("Find mask")
        if self.continous_recomputation or self._cached_mask is None:
            log.debug("Recomputation of mask is requested")
            self._cached_mask = self._compute_mask(img1, img2)

            if self.save_to_file:
                log.debug("Save computed mask to file")
                np.save("data/out/matrix/mask.npy", self._cached_mask, allow_pickle=False)

        return self._cached_mask

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        """Abstract method to compute mask.

        Parameters
        ----------
        img1 : Image
            First input image, transformed and cropped
        img2 : Image
            Second input image, transformed and cropped

        Returns
        -------
        Mask
            The mask computed for these images.

        Raises
        ------
        NotImplementedError
            Unless overriden, this method will raise an error.
        """
        raise NotImplementedError("Abstract method needs to be overriden")

    def apply_mask(self, img1: Image, img2: Image, mask: Mask) -> Image:
        """Applies mask to transformed images to create stitched result.

        Parameters
        ----------
        img1 : Image
            First input image, transformed and cropped
        img2 : Image
            Second input image, transformed and cropped
        mask : Mask
            The mask to use. Will be inverted if instance attribute "invert" is set.

        Returns
        -------
        Image
            Stitched image created by the mask.
        """
        log.debug("Apply mask to images")
        mask1, mask2 = (1 - mask, mask) if self.invert else (mask, 1 - mask)
        img1_masked = img1.astype(np.float64) * mask1
        img2_masked = img2.astype(np.float64) * mask2
        final = (img1_masked + img2_masked).astype(np.uint8)
        return Image(final)
