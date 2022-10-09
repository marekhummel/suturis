import numpy as np
import cv2

GAUSS_SIZE = 17


def _create_mask_from_seam(seammat, im1, im2):
    """
    Creates a mask without needing to floodfill because in each row there is one true value.
    """
    mask_mat = np.ones_like(im1)

    # Much smarter is to just follow the seam
    currPos = 0
    for row in range(seammat.shape[0]):
        if row == 0:
            for col in range(seammat.shape[1]):
                if seammat[row][col]:
                    currPos = col
                    mask_mat[
                        row : row + 1,
                        col : seammat.shape[1] + 1,
                    ] = [0, 0, 0]
                    break
        else:
            if currPos > 1 and seammat[row][currPos - 2]:
                currPos = currPos - 2
            elif currPos > 0 and seammat[row][currPos - 1]:
                currPos = currPos - 1
            elif currPos < seammat.shape[1] - 3 and seammat[row][currPos + 2]:
                currPos = currPos + 2
            elif currPos < seammat.shape[1] - 2 and seammat[row][currPos + 1]:
                currPos = currPos + 1
            mask_mat[
                row : row + 1,
                currPos : seammat.shape[1] + 1,
            ] = [0, 0, 0]

    # stitch.show_image('Carvmask: After', mask_mat)
    return cv2.GaussianBlur(
        mask_mat,
        (GAUSS_SIZE, GAUSS_SIZE),
        0,
        sigmaY=0,
        borderType=cv2.BORDER_REPLICATE,
    )


seammat = np.loadtxt("seammat2.txt", dtype=int)
img1 = np.zeros((*seammat.shape, 3))
img2 = np.zeros((*seammat.shape, 3))
mask = _create_mask_from_seam(seammat, img1, img2)
cv2.imwrite("data/out/mask2.jpg", mask * 255)
