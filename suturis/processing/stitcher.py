import cv2
import numpy as np

stitcher = cv2.Stitcher_create()


async def compute(image1, image2):
    stitched = _stitch(image1, image2)
    return _combine_with_input(image1, image2, stitched)


def _stitch(image1, image2):
    success, image = stitcher.stitch(image1, image2)
    print(success)
    return image


def _test_stitch(image1, image2):
    stitched = cv2.rotate(image2, rotateCode=cv2.ROTATE_180)
    return stitched


def _combine_with_input(image1, image2, stitch):
    image1 = cv2.resize(image1, (0, 0), None, 0.5, 0.5)
    image2 = cv2.resize(image2, (0, 0), None, 0.5, 0.5)
    stitch = cv2.resize(stitch, (0, 0), None, 0.5, 0.5)
    w = image1.shape[1]
    stitch = np.pad(
        stitch, [(0, 0), (w // 2, w // 2), (0, 0)], mode="constant", constant_values=0
    )
    image = np.concatenate((image1, image2), axis=1)
    image = np.concatenate((image, stitch), axis=0)
    return image
