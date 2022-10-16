import numpy as np
import cv2

# x = 458
# size = (720, 667)
x = 1079
size = (1024, 1946)

mask = np.zeros(shape=(*size, 3), dtype=np.float64)
mask[:, :x, :] = 1
mask = cv2.GaussianBlur(mask, (27, 27), 0, borderType=cv2.BORDER_REPLICATE)

np.save("data/examples/files/mask_bowstern.npy", mask)
