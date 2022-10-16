import cv2
import numpy as np

img1 = cv2.imread("data/out/debug/img0_bowstern.png")
img2 = cv2.imread("data/out/debug/img1_bowstern.png")
homography_matrix = np.array(
    [
        [-1.0746634072336902, 0.21470403381513473, 1908.9075699417156],
        [-0.2803016477992872, -0.9018752325755722, 787.3071382502388],
        [-0.00010127147329093602, 8.487885432886828e-05, 1.0],
    ]
)
target_width, target_height = (1946, 1024)
tx, ty = (0, 236)
crop = [(236, 613), (955, 1279)]

img1_height, img1_width = img1.shape[:2]
img1_translated = np.zeros(shape=(target_height, target_width, 3), dtype=np.uint8)
img1_translated[ty : img1_height + ty, tx : img1_width + tx] = img1

translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
img2_warped = cv2.warpPerspective(img2, translation_matrix @ homography_matrix, (target_width, target_height))


faded_mask = np.full_like(img1_translated, 0.5, dtype=np.float64)
result = img1_translated * faded_mask + img2_warped * faded_mask

cv2.circle(result, crop[0][::-1], 4, (255, 0, 0), -1)
cv2.circle(result, crop[1][::-1], 4, (255, 0, 0), -1)

cv2.imshow("DEBUG1", result.astype(np.uint8))
cv2.waitKey(0)
