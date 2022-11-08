import cv2
import numpy as np

import sys
import os

sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/../../src")

from suturis.processing.computation.masking.seam_carving import SeamCarving

image1 = cv2.imread("data/out/debug/img1_transformed.jpg")
image2 = cv2.imread("data/out/debug/img2_transformed.jpg")
blocked_area1 = [[540, 70], [1400, 360]]
blocked_area2 = [[0, 420], [825, 700]]

sc = SeamCarving(blocked_area1, blocked_area2, caching_enabled=True)
mask = sc._compute_mask(image1, image2) * 255

cv2.imwrite("data/out/debug/test.jpg", mask.astype(np.uint8))
