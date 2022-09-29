import numpy as np
from suturis.typing import CvSize, Image, Mask, Point, TranslationVector


class BaseMaskingHandler:
    def __init__(self, continous_recomputation: bool):
        self.continous_recomputation = continous_recomputation

    def compute_mask(
        self,
        img1: Image,
        img2: Image,
        target_size: CvSize,
        translation: TranslationVector,
        crop_area: tuple[Point, Point],
    ) -> Mask:
        raise NotImplementedError("Abstract method needs to be overriden")

    def apply_mask(self, img1: Image, img2: Image, mask: Mask) -> Image:
        img1_masked = img1.astype(np.float64) * mask
        img2_masked = img2.astype(np.float64) * (1 - mask)
        return img1_masked + img2_masked
