import logging as log
from typing import Any

import cv2
import numpy as np
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import CvRect, Image, ImagePair, Mask, Matrix, SeamMatrix

END = 0
LEFT = 2
BOTTOM = 3
RIGHT = 4

LOW_LAB_IN_BGR = [195, 59, 0]
HIGH_LAB_IN_BGR = [0, 62, 255]

GAUSS_SIZE = 17


class SeamCarving(BaseMaskingHandler):
    """Masking handler which finds an low energy (least difference in color) seam from top to bottom"""

    blocked_area_one: CvRect | None
    blocked_area_two: CvRect | None

    def __init__(self, blocked_area_one: CvRect | None = None, blocked_area_two: CvRect | None = None, **kwargs: Any):
        """_summary_

        Parameters
        ----------
        blocked_area_one : CvRect | None, optional
            Rectangle for first image, in which an artifical high energy will be created, by default None
        blocked_area_two : CvRect | None, optional
            Rectangle for second image, in which an artifical high energy will be created, by default None
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init Seam Carving Masking Handler")
        super().__init__(**kwargs)
        self.blocked_area_one = blocked_area_one
        self.blocked_area_two = blocked_area_two

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        """Computation of the mask with the seam carving algorithm (implementation could be better).

        Parameters
        ----------
        img1 : Image
            Transformed and cropped first image
        img2 : Image
            Transformed and cropped second image

        Returns
        -------
        Mask
            The mask matrix used to combine the images.
        """
        log.debug("Compute new mask from images")

        log.debug("Create high contrast areas to guide seam")
        img1_modified, img2_modified = self._insert_blocked_areas(img1, img2)

        log.debug("Find seam through images")
        seam_matrix = self._find_seam(img1_modified, img2_modified)

        log.debug("Convert seam matrix into image mask")
        return self._create_mask_from_seam(seam_matrix, img1, img2)

    def _insert_blocked_areas(self, img1: Image, img2: Image) -> ImagePair:
        """Sets rectangles in both image to certain colors to artifically create high energy areas,
        which will be avoided by the seam.

        Parameters
        ----------
        img1 : Image
            Transformed and cropped first image
        img2 : Image
            Transformed and cropped second image

        Returns
        -------
        ImagePair
            Modified images.
        """
        img1_modified = img1.copy()
        img2_modified = img2.copy()

        if self.blocked_area_one:
            start1, end1 = self.blocked_area_one
            img1_modified[start1[1] : end1[1] + 1, start1[0] : end1[0] + 1] = LOW_LAB_IN_BGR
            img2_modified[start1[1] : end1[1] + 1, start1[0] : end1[0] + 1] = HIGH_LAB_IN_BGR

        if self.blocked_area_two:
            start2, end2 = self.blocked_area_two
            img1_modified[start2[1] : end2[1] + 1, start2[0] : end2[0] + 1] = HIGH_LAB_IN_BGR
            img2_modified[start2[1] : end2[1] + 1, start2[0] : end2[0] + 1] = LOW_LAB_IN_BGR

        return img1_modified, img2_modified

    def _find_seam(self, im1: Image, im2: Image) -> SeamMatrix:
        """Finds bool matrix to describe the seam for the mask.

        Parameters
        ----------
        img1 : Image
            Transformed and cropped first image
        img2 : Image
            Transformed and cropped second image
        Returns
        -------
        SeamMatrix
            Bool matrix to describe seam.
        """
        ystart, xstart = 0, 0
        yend, xend = im1.shape[:2]
        dif = self._get_energy(im1, im2, xstart, ystart, xend, yend)
        previous = np.zeros((yend - ystart, xend - xstart))
        paths = np.zeros_like(previous)
        for row in range(yend - ystart):
            previous, paths = self._fill_row(paths, yend - ystart - row - 1, dif, previous)

        return self._find_bool_matrix(previous, paths)

    def _get_energy(self, img1: Image, img2: Image, xstart: int, ystart: int, xend: int, yend: int) -> Matrix:
        """Creates energy map between both images, energy meaning color difference (higher energy = less similar colors)

        Parameters
        ----------
        img1 : Image
            Transformed and cropped first image
        img2 : Image
            Transformed and cropped second image
        xstart : int
            Starting x-value.
        ystart : int
            Starting y-value.
        xend : int
            Ending x-value.
        yend : int
            Ending y-value.

        Returns
        -------
        Matrix
            Energy map. Values inside the blocked areas will have max energy.
        """
        log.debug("Compute energy map")
        width = xend - xstart
        height = yend - ystart
        dif = np.zeros((height, width))
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
        img1_lab = np.where(img1 == [0, 0, 0], LOW_LAB_IN_BGR, img1_lab)
        img2_lab = np.where(img2 == [0, 0, 0], HIGH_LAB_IN_BGR, img2_lab)
        for row in range(height):
            for col in range(width):
                dif[row][col] = np.sqrt(
                    sum(
                        abs(int(x) - int(y))
                        for x, y in zip(
                            img1_lab[row + ystart][col + xstart],
                            img2_lab[row + ystart][col + xstart],
                        )
                    )
                )
        return dif

    def _fill_row(self, paths: Matrix, row: int, dif: Matrix, previous: Matrix) -> tuple[Matrix, Matrix]:
        """Fill one row based on the values from the previous row.

        Parameters
        ----------
        paths : Matrix
            Current accumulated energies
        row : int
            Current row to update
        dif : Matrix
            Energy map
        previous : Matrix
            Current path array

        Returns
        -------
        tuple[Matrix, Matrix]
            Updated previous and paths arrays.
        """
        assert 0 <= row < dif.shape[0]
        assert dif.shape == paths.shape

        # First row -> lowest
        if row == dif.shape[0] - 1:
            # This row only consists of the differences between the two images
            paths[row, :] = dif[row, :]
            # No previous field to come from.
            previous[row, :] = END
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
                    leftval = paths[row + 1][col - 1]
                    downval = paths[row + 1][col]
                    m = min(leftval, downval)
                    paths[row][col] = dif[row][col] + m
                    previous[row][col] = LEFT if m == leftval else BOTTOM
                else:
                    # Big trouble
                    assert False, f"col is: {col}"

        return previous, paths

    def _find_bool_matrix(self, previous: Matrix, paths: Matrix) -> SeamMatrix:
        """Finds a bool matrix for the seam for not so sharp angles.

        Parameters
        ----------
        previous : Matrix
            Path array
        paths : Matrix
            Accumulated energy values

        Returns
        -------
        SeamMatrix
            Bool matrix with seam.
        """
        index = 0
        current_min = 0
        bools = SeamMatrix(np.full(previous.shape, False))
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

    def _create_mask_from_seam(self, seammat: SeamMatrix, im1: Image, im2: Image) -> Mask:
        """Creates a mask without needing to floodfill because in each row there is one true value.

        Parameters
        ----------
        seammat : SeamMatrix
            Bool matrix with the seam
        im1 : Image
            First image
        im2 : Image
            Second image

        Returns
        -------
        Mask
            Mask applicable to images based on the seam matrix.
        """
        mask_mat = np.ones((*seammat.shape, 3))

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
        return Mask(
            cv2.GaussianBlur(
                mask_mat,
                (GAUSS_SIZE, GAUSS_SIZE),
                0,
                sigmaY=0,
                borderType=cv2.BORDER_REPLICATE,
            )
        )
