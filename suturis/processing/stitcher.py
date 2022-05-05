import cv2
import numpy as np


def stitch(image1, image2):
    image1 = cv2.resize(image1, (0, 0), None, .5, .5)
    image2 = cv2.resize(image2, (0, 0), None, .5, .5)
    image2 = cv2.rotate(image2, rotateCode=cv2.ROTATE_180)
    image = np.concatenate((image1, image2), axis=0)
    return image
