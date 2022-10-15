import cv2
import numpy as np
from time import perf_counter


def _get_energy(img1, img2, xstart, ystart, xend, yend):
    width = xend - xstart + 1
    height = yend - ystart + 1
    dif = np.zeros((height, width))
    img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2Lab)
    img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2Lab)

    for row in range(height):
        for col in range(width):
            dif[row][col] = np.sqrt(
                sum(
                    abs(int(x) - int(y))
                    for x, y in zip(
                        img1[row + ystart][col + xstart],
                        img2[row + ystart][col + xstart],
                    )
                )
            )

    return dif


def _get_energy2(img1, img2, xstart, ystart, xend, yend):
    img1_crop = img1[ystart : yend + 1, xstart : xend + 1, :]
    img2_crop = img2[ystart : yend + 1, xstart : xend + 1, :]

    img1_lab = cv2.cvtColor(img1_crop.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.int32)
    img2_lab = cv2.cvtColor(img2_crop.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.int32)

    diff = np.abs(img1_lab - img2_lab)
    diff = np.sum(diff, axis=2)
    diff = np.sqrt(diff)
    return diff


img1 = cv2.imread("data/out/debug/img1_transformed.jpg")
img2 = cv2.imread("data/out/debug/img2_transformed.jpg")
ystart = 57
yend = 774
xstart = 37
xend = 1279


start = perf_counter()
diff1 = _get_energy(img1, img2, xstart, ystart, xend, yend)
print(perf_counter() - start)
start = perf_counter()
diff2 = _get_energy2(img1, img2, xstart, ystart, xend, yend)
print(perf_counter() - start)


print(np.average(diff1), np.average(diff2))
