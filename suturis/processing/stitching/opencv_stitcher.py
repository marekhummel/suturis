import cv2


class OpenCvStitcher:
    def __init__(self) -> None:
        self.stitcher = cv2.Stitcher_create()

    def stitch(self, image1, image2):
        success, image = self.stitcher.stitch(image1, image2)
        return image
