import logging as log

import cv2
import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import CvSize, Image, Mask, CvPoint, NpPoint, SeamMatrix, TranslationVector

END = 0
LEFT_LEFT = 1
LEFT = 2
BOTTOM = 3
RIGHT = 4
RIGHT_RIGHT = 5

LOW_LAB_IN_BGR = [195, 59, 0]
HIGH_LAB_IN_BGR = [0, 62, 255]

GAUSS_SIZE = 17


class SeamCarving(BaseMaskingHandler):
    blocked_area_one: tuple[CvPoint, CvPoint] | None
    blocked_area_two: tuple[CvPoint, CvPoint] | None

    def __init__(
        self,
        continous_recomputation: bool,
        blocked_area_one: tuple[CvPoint, CvPoint] | None = None,
        blocked_area_two: tuple[CvPoint, CvPoint] | None = None,
    ):
        log.debug("Init Seam Carving Masking Handler")
        super().__init__(continous_recomputation)
        self.blocked_area_one = blocked_area_one
        self.blocked_area_two = blocked_area_two

    def _compute_mask(self, img1: Image, img2: Image, target_size: CvSize, translation: TranslationVector) -> Mask:
        img1_modified, img2_modified = self._insert_blocked_areas(img1, img2)
        seam_matrix = self._find_seam(img1_modified, img2_modified)
        return self._create_mask_from_seam(seam_matrix, img1, img2, translation, target_size)

    def _insert_blocked_areas(self, img1: Image, img2: Image) -> tuple[Image, Image]:
        img1_modified = img1.copy()
        img2_modified = img2.copy()

        start1, end1 = self.blocked_area_one
        img1_modified[start1[1] : end1[1] + 1, start1[0] : end1[0] + 1] = LOW_LAB_IN_BGR
        img2_modified[start1[1] : end1[1] + 1, start1[0] : end1[0] + 1] = HIGH_LAB_IN_BGR

        start2, end2 = self.blocked_area_two
        img1_modified[start2[1] : end2[1] + 1, start2[0] : end2[0] + 1] = HIGH_LAB_IN_BGR
        img2_modified[start2[1] : end2[1] + 1, start2[0] : end2[0] + 1] = LOW_LAB_IN_BGR

        return img1_modified, img2_modified

    def _find_seam(self, im1: Image, im2: Image) -> SeamMatrix:
        ystart, xstart = 0, 0
        yend, xend = im1.shape[:2]
        dif = self._get_energy(im1, im2, xstart, ystart, xend, yend)
        previous = np.zeros((yend - ystart, xend - xstart))
        paths = np.zeros_like(previous)
        for row in range(yend - ystart):
            previous, paths = self._fill_row(paths, yend - ystart - row - 1, dif, previous)
        return self._find_bool_matrix(previous, paths)

    def _get_energy(self, img1, img2, xstart, ystart, xend, yend):
        width = xend - xstart
        height = yend - ystart
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

    def _fill_row(self, paths, row, dif, previous):
        """
        Fill one row based on the values from the previous row.
        """
        assert 0 <= row < dif.shape[0]
        assert dif.shape == paths.shape

        # First row -> lowest
        if row == dif.shape[0] - 1:
            # This row only consists of the differences between the two images
            paths[row, :] = dif[row, :]
            # No previous field to come from.
            previous[row, :] = 0
        else:
            for col in range(dif.shape[1]):
                if 0 < col < dif.shape[1] - 1:
                    # Just somewhere in the middle
                    leftval = paths[row + 1][col - 1]
                    downval = paths[row + 1][col]
                    rightval = paths[row + 1][col + 1]
                    m = min(leftval, downval, rightval)
                    # Get a new value
                    paths[row][col] = dif[row][col] + m
                    # Determine which value we used
                    if m == leftval:
                        previous[row][col] = LEFT
                    elif m == downval:
                        previous[row][col] = BOTTOM
                    else:
                        previous[row][col] = RIGHT
                elif col == 0:
                    # Left side of the image.
                    downval = paths[row + 1][col]
                    rightval = paths[row + 1][col + 1]
                    m = min(downval, rightval)
                    paths[row][col] = dif[row][col] + m
                    previous[row][col] = BOTTOM if m == downval else RIGHT
                elif col < dif.shape[1]:
                    # Right end of image
                    # left = paths[row + 1][col - 1]
                    # down = paths[row + 1][col]
                    m = min(leftval, downval)
                    paths[row][col] = dif[row][col] + m
                    previous[row][col] = LEFT if m == leftval else BOTTOM
                else:
                    # Big trouble
                    assert False, f"col is: {col}"

        return previous, paths

    def _find_bool_matrix(self, previous, paths):
        """
        Finds a bool matrix for the seam for not so sharp angles.
        """
        index = 0
        current_min = 0
        bools = np.full(previous.shape, False)
        current_row = 0
        for col in range(paths.shape[1]):
            # Find minimum in the top row
            if col == 0 or paths[current_row][col] < current_min:
                index = col
                current_min = paths[current_row][col]

        finished = False
        # Starting from that minimum in the top row we try to find the path back using the values in previous
        while not finished:
            # Setting the current position to true
            bools[current_row][index] = True

            # Get direction for next position
            val = previous[current_row][index]
            if val == END:
                finished = True
            elif val == LEFT:
                # Making a step to the left
                index -= 1
            elif val == RIGHT:
                # Making a step to the right
                index += 1

            # Moving a step down
            current_row += 1
        return bools

    def _create_mask_from_seam(self, seammat, im1, im2, translation, target_size):
        """
        Creates a mask for the merged images with values between 0 and 1. 1 is for fully filling the left image,
        0 is for fully filling the right image. Everything else is in between.
        """
        assert im1.shape == im2.shape
        mask_mat = np.zeros(im1.shape)
        result_width, result_height = target_size
        x_trans, y_trans = translation

        mask_mat[y_trans : result_height + y_trans, x_trans : result_width + x_trans] = [
            1.0,
            1.0,
            1.0,
        ]

        # Use the flood fill algorithm to fill the seam matrix
        if not seammat[seammat.shape[0] - 1][0]:
            self._flood_fill(
                seammat,
                seammat.shape[0],
                seammat.shape[1],
                seammat.shape[0] - 1,
                0,
                False,
                True,
            )
        else:
            for row in range(seammat.shape[0]):
                if not seammat[row][0]:
                    self._flood_fill(seammat, seammat.shape[0], seammat.shape[1], 0, row, False, True)

            for col in range(seammat.shape[1]):
                if not seammat[seammat.shape[0] - 1][col]:
                    self._flood_fill(
                        seammat,
                        seammat.shape[0],
                        seammat.shape[1],
                        col,
                        seammat.shape[0] - 1,
                        False,
                        True,
                    )

        # Go through seammat area and fill in zeros at the right manually
        for row in range(seammat.shape[0]):
            for col in range(seammat.shape[1]):
                if not seammat[row][col]:
                    mask_mat[row][col + 1] = [0.0, 0.0, 0.0]
        # stitch.show_image('Mask', mask_mat)

        # Add a bit of gauss and return it
        return cv2.GaussianBlur(
            mask_mat,
            (GAUSS_SIZE, GAUSS_SIZE),
            0,
            sigmaY=0,
            borderType=cv2.BORDER_REPLICATE,
        )

    def _flood_fill(self, seammat, xmax, ymax, x, y, previous_val, new_val):
        """
        Implementation of floodfill with a queue (Q) instead of a recursion because the stack is
        usually limited. We have to respect that. Still very sad, that the recursion didn't work.
        """

        # Append the position of starting
        # pixel of the component
        queue = [[x, y]]

        # Change first pixel to new_val
        seammat[x][y] = new_val

        # While the queue is not empty i.e. the
        # whole component having previous_val
        # is not changed to new_val
        while queue:

            # Dequeue the front node
            currPixel = queue.pop()

            posX = currPixel[0]
            posY = currPixel[1]

            # Check if the adjacent
            # pixels are valid
            if self._is_valid(seammat, xmax, ymax, posX + 1, posY, previous_val, new_val):
                # Right
                seammat[posX + 1][posY] = new_val
                queue.append([posX + 1, posY])

            if self._is_valid(seammat, xmax, ymax, posX - 1, posY, previous_val, new_val):
                # Left
                seammat[posX - 1][posY] = new_val
                queue.append([posX - 1, posY])

            if self._is_valid(seammat, xmax, ymax, posX, posY + 1, previous_val, new_val):
                # Bottom
                seammat[posX][posY + 1] = new_val
                queue.append([posX, posY + 1])

            if self._is_valid(seammat, xmax, ymax, posX, posY - 1, previous_val, new_val):
                # Top
                seammat[posX][posY - 1] = new_val
                queue.append([posX, posY - 1])

    def _is_valid(self, seammat, xmax, ymax, x, y, previous_val, new_val):
        """
        Helping function for the floodfill that checks, whether a given pixel can be overridden.
        """
        return (
            x >= 0 and x < xmax and y >= 0 and y < ymax and seammat[x][y] == previous_val and seammat[x][y] != new_val
        )
