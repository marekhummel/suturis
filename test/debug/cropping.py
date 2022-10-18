import cv2
import numpy as np
import time

image = cv2.imread("data/out/debug/result.jpg")

start = time.perf_counter()
height, width = image.shape[:2]
xmin, xmax = 0, width - 1
ymin, ymax = 0, height - 1


is_black = np.all(image[:, :] == [0, 0, 0], axis=2)
black_ratio_rows = np.average(is_black.astype(np.float64), axis=1)
black_ratio_cols = np.average(is_black.astype(np.float64), axis=0)


borders = [[xmin, 1, 1], [xmax, 1, -1], [ymin, 0, 1], [ymax, 0, -1]]


ratios = [black_ratio_rows, black_ratio_cols]
while True:
    best = None
    for border_index, (index, ratio_index, increment) in enumerate(borders):
        curr_ratio = ratios[ratio_index][index]
        if best is None or curr_ratio > best[1]:
            best = border_index, curr_ratio

    if best[1] < 0.1:
        break

    borders[best[0]][0] += borders[best[0]][2]


# while black_ratio_cols[xmin] == 1:
#     xmin += 1

# while black_ratio_cols[xmax] == 1:
#     xmax -= 1

# while black_ratio_rows[ymin] == 1:
#     ymin += 1

# while black_ratio_rows[ymax] == 1:
#     ymax -= 1

# cropped = image[:, xmin : xmax + 1, :]

xmin, xmax, ymin, ymax = [b[0] for b in borders]
cropped = image.copy()
cropped[:, :xmin] = [0, 0, 127]
cropped[:, xmax + 1 :] = [0, 0, 127]
cropped[:ymin, :] = [0, 0, 127]
cropped[ymax + 1 :, :] = [0, 0, 127]
print(f"Cropped image from {(0, width - 1)} to {(xmin, xmax)} and {(0, height - 1)} to {(ymin, ymax)}")

print(time.perf_counter() - start)

cv2.imshow("f", cropped.astype(np.uint8))
cv2.waitKey(0)
