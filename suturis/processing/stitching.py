import suturis.processing.preprocessing as pre
from suturis.processing.stitcher.current_stitcher import CurrentStitcher
from suturis.processing.stitcher.opencv_stitcher import OpenCvStitcher
from suturis.processing.stitcher.video_stitcher import VideoStitcher

stitcher = VideoStitcher()


async def compute(*images):
    mod_images = pre.preprocess(*images)
    stitched = stitcher.stitch(mod_images)
    return stitched


def get_sticher():
    return stitcher
