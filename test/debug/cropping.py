import cv2
import numpy as np
import time

image = cv2.imread("data/out/debug/result.jpg")

start = time.perf_counter()
height, width = image.shape[:2]


is_black = np.all(image[:, :] == [0, 0, 0], axis=2).astype(np.float64)
edges = [[0, 1, 1], [width - 1, 1, -1], [0, 0, 1], [height - 1, 0, -1]]

while True:
    best = None
    for edge_index, (index, axis, increment) in enumerate(edges):
        other_dims = [e[0] for e in edges if e[1] != axis]
        line_ratio = np.average(np.take(is_black, index, axis)[other_dims[0] : other_dims[1] + 1])
        if best is None or line_ratio > best[1]:
            best = edge_index, line_ratio

    if best[1] < 0.02:
        break

    edges[best[0]][0] += edges[best[0]][2]


xmin, xmax, ymin, ymax = [b[0] for b in edges]
cropped = image.copy()
cropped[:, :xmin] = [0, 0, 127]
cropped[:, xmax + 1 :] = [0, 0, 127]
cropped[:ymin, :] = [0, 0, 127]
cropped[ymax + 1 :, :] = [0, 0, 127]
print(f"Cropped image from {(0, width - 1)} to {(xmin, xmax)} and {(0, height - 1)} to {(ymin, ymax)}")

print(time.perf_counter() - start)

cv2.imshow("f", cropped.astype(np.uint8))
cv2.waitKey(0)
