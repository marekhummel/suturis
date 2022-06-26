import cv2
import numpy as np
from suturis.processing.stitching.current_stitcher import CurrentStitcher
from suturis.processing.stitching.video_stitcher import VideoStitcher
# import suturis.processing.stitching.opencv_stitcher


async def compute(image1, image2):
    # stitcher = CurrentStitcher()
    stitcher = VideoStitcher()
    stitched = stitcher.stitch(image1, image2)
    # print(stitched.shape)
    # stitched = current_stitcher.stitch(image1, image2)
    return _combine_with_input(image1, image2, stitched)


def _test_stitch(image1, image2):
    stitched = cv2.rotate(image2, rotateCode=cv2.ROTATE_180)
    return stitched


def _combine_with_input(image1, image2, stitch):
    image1 = cv2.resize(image1, (0, 0), None, 0.5, 0.5)
    image2 = cv2.resize(image2, (0, 0), None, 0.5, 0.5)
    h, w = image1.shape[:2]
    sw = stitch.shape[1]
    stitch = cv2.resize(stitch, (min(sw, w + h), h), interpolation=cv2.INTER_CUBIC)
    pad = (2 * w - stitch.shape[1]) // 2
    if pad > 0:
        stitch = np.pad(
            stitch, [(0, 0), (pad, pad), (0, 0)], mode="constant", constant_values=255
        )
    image = np.concatenate((image1, image2), axis=1)
    image = np.concatenate((image, stitch), axis=0)
    return image
