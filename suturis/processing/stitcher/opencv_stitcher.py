import cv2


class OpenCvStitcher:
    def __init__(self) -> None:
        self.stitcher = cv2.Stitcher_create()

    def stitch(self, images):
        success, image = self.stitcher.stitch(*images)
        return image
