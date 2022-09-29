import cv2
import numpy as np
from suturis.typing import CvSize, Homography, Image, Point, TranslationVector


class BaseHomographyHandler:
    def __init__(self, continous_recomputation: bool):
        self.continous_recomputation = continous_recomputation

    def find_homography(self, img1: Image, img2: Image) -> tuple[TranslationVector, CvSize, Homography]:
        raise NotImplementedError("Abstract method needs to be overriden")

    def find_crop(self, img1: Image, homography: Homography, translation: TranslationVector) -> tuple[Point, Point]:
        raise NotImplementedError("Abstract method needs to be overriden")

    def apply_transformations(
        self,
        img1: Image,
        img2: Image,
        target_size: CvSize,
        translation: TranslationVector,
        homography_matrix: Homography,
    ) -> tuple[Image, Image]:
        target_width, target_height = target_size
        tx, ty = translation

        # Translate image 1 (could apply warpPerspective or warpAffine as well with respective
        # translation matrices but this is faster)
        img1_height, img1_width = img1.shape[:2]
        img1_translated = np.zeros(shape=(target_height, target_width, 3))
        img1_translated[ty : img1_height + ty, tx : img1_width + tx] = img1

        # Perspective transform on image 2
        # This warp is the reason for the new image size, but will create negative pixel coordinates,
        # hence the translation
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        img2_warped = cv2.warpPerspective(img2, translation_matrix @ homography_matrix, target_size)

        # Return images with same sized canvas for easy overlay
        return img1_translated, img2_warped
