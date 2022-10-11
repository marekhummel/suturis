import numpy as np

homography = np.load("data/out/homography.npy")
print(homography, homography.shape)

mask = np.load("data/out/mask.npy")
print(mask.shape)
