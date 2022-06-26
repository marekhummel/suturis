import cv2

stitcher = cv2.Stitcher_create()

def stitch(image1, image2):
    success, image = stitcher.stitch(image1, image2)
    return image
