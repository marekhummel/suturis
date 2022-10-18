import numpy as np
import cv2

img1 = cv2.imread("data/out/debug/img1_transformed.jpg")
img2 = cv2.imread("data/out/debug/img2_transformed.jpg")
y = 400

bool_mask = np.zeros(shape=img1.shape[:2], dtype=bool)
bool_mask[:y, :] = True


is_black_img1 = np.all(img1[:, :] == [0, 0, 0], axis=2)
is_black_img2 = np.all(img2[:, :] == [0, 0, 0], axis=2)


bool_mask[bool_mask & is_black_img2] = False
bool_mask[~bool_mask & is_black_img1] = True

stacked = np.stack([bool_mask for _ in range(3)], axis=-1).astype(np.float64)

# stacked = cv2.GaussianBlur(stacked, (27, 27), 0, borderType=cv2.BORDER_REPLICATE)
output = img1 * (1 - stacked) + img2 * stacked

# cv2.imshow("m", bool_mask.astype(np.uint8) * 255)
cv2.imwrite("data/out/thesis/extend_borders_crop.jpg", output.astype(np.uint8))
