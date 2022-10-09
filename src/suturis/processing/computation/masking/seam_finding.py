import logging as log

import cv2
import numpy as np
import numpy.typing as npt
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import CvPoint, NpSize, Image, Mask, SeamMatrix, TranslationVector

END = 0
F_BOT_LEFT = 1
F_LEFT = 2
F_TOP_LEFT = 3
F_TOP = 4
F_TOP_RIGHT = 5

GAUSS_SIZE = 17


class SeamFinding(BaseMaskingHandler):
    preferred_seam: tuple[CvPoint, CvPoint] | None

    def __init__(self, continous_recomputation: bool, preferred_seam: tuple[CvPoint, CvPoint] | None = None):
        log.debug("Init Seam Finding Masking Handler")
        super().__init__(continous_recomputation)
        self.preferred_seam = preferred_seam

    def _compute_mask(self, img1: Image, img2: Image, output_size: NpSize) -> Mask:
        img1_modified = self._insert_preferred_seam(img1, img2)
        seam_matrix = self._find_seam(img1_modified, img2)
        return self._create_mask_from_seam(seam_matrix, img1, img2)

    def _insert_preferred_seam(self, img1: Image, img2: Image) -> Image:
        img1_modified = img1.copy()
        if self.preferred_seam is None:
            return img1_modified

        start, end = self.preferred_seam
        slope = (end[1] - start[1]) / (end[0] - start[0])
        y = float(start[1])
        for x in range(start[0], end[0] + 1):
            iy = int(y)
            img1_modified[iy][x] = img2[iy][x]
            y += slope

        return img1_modified

    def _find_seam(self, img1: Image, img2: Image) -> SeamMatrix:
        # Convert color into Lab space, because we would like to follow EN ISO 11664-4
        img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2Lab)
        img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2Lab)

        # Doing some initializations
        height, width = img1.shape[:2]
        errors = np.zeros((height, width))
        previous = np.zeros((height, width))

        # Calculate distances
        for idx in range(min(errors.shape[0], errors.shape[1])):
            dist = np.sqrt(sum(abs(int(x) - int(y)) for x, y in zip(img1[idx][idx], img2[idx][idx])))

            is_last_col, is_last_row = False, False
            # Check whether we are at an edge
            if idx == 0:
                # Top left corner. Nothing to do here but put the distance in and indicate the end
                errors[idx][idx] = dist
                previous[idx][idx] = END
            else:
                # Somewhere in the middle
                if idx < errors.shape[0] - 1 and idx < errors.shape[1] - 1:
                    # Really somewhere in the middle
                    errors[idx][idx] = dist + min(
                        errors[idx - 1][idx - 1],
                        errors[idx][idx - 1],
                        errors[idx - 1][idx],
                        errors[idx + 1][idx - 1],
                        errors[idx - 1][idx + 1],
                    )
                elif idx < errors.shape[0] - 1:
                    # At the bottom of the area
                    errors[idx][idx] = dist + min(
                        errors[idx - 1][idx - 1],
                        errors[idx][idx - 1],
                        errors[idx - 1][idx],
                        errors[idx + 1][idx - 1],
                    )
                    is_last_col = True
                elif idx < errors.shape[1] - 1:
                    # Reached the right side of the area
                    errors[idx][idx] = dist + min(
                        errors[idx - 1][idx - 1],
                        errors[idx][idx - 1],
                        errors[idx - 1][idx],
                        errors[idx - 1][idx + 1],
                    )
                    is_last_row = True
                else:
                    # Reached bottom right of the area
                    errors[idx][idx] = dist + min(errors[idx - 1][idx - 1], errors[idx][idx - 1], errors[idx - 1][idx])
                    is_last_row = True
                    is_last_col = True

                # Where is the value from you may ask? Let's find that out!
                if errors[idx][idx] == dist + errors[idx - 1][idx - 1]:
                    # Top left
                    previous[idx][idx] = F_TOP_LEFT
                elif errors[idx][idx] == dist + errors[idx][idx - 1]:
                    # Left
                    previous[idx][idx] = F_LEFT
                elif errors[idx][idx] == dist + errors[idx - 1][idx]:
                    # Top
                    previous[idx][idx] = F_TOP
                elif not is_last_row and errors[idx][idx] == dist + errors[idx + 1][idx - 1]:
                    # Bottom left
                    previous[idx][idx] = F_BOT_LEFT
                elif not is_last_col and errors[idx][idx] == dist + errors[idx - 1][idx + 1]:
                    # Top right
                    previous[idx][idx] = F_TOP_RIGHT

            if is_last_col:
                # Now we reached the end of the columns so we only need to fill that column, then we're done
                errors, previous = self._fill_column(idx, img1, img2, errors, previous)
            elif is_last_row:
                # Now we reached the end of the rows, so we only need to fill that row before we're done
                errors, previous = self._fill_row(idx, img1, img2, errors, previous)
            else:
                # We have to fill both rows and columns
                errors, previous = self._fill_row(idx, img1, img2, errors, previous)
                errors, previous = self._fill_column(idx, img1, img2, errors, previous)

        return self._find_bool_matrix(previous)

    def _fill_row(
        self, row: int, im1: Image, im2: Image, errors: npt.NDArray, previous: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Iterate through one row and fill in values
        """
        assert row < errors.shape[0], f"idx is: {row}, errors.shape[0] is: {errors.shape[0]}"
        # Go through columns
        for col in range(row + 1, errors.shape[1]):
            dist = np.sqrt(sum(abs(int(x) - int(y)) for x, y in zip(im1[row][col], im2[row][col])))

            if row > 0:
                is_last_one = False
                if col < errors.shape[1] - 1:
                    errors[row][col] = dist + min(
                        errors[row - 1][col - 1],
                        errors[row][col - 1],
                        errors[row - 1][col],
                        errors[row - 1][col + 1],
                    )
                else:
                    is_last_one = True
                    errors[row][col] = dist + min(errors[row - 1][col - 1], errors[row][col - 1], errors[row - 1][col])
                if errors[row][col] == dist + errors[row - 1][col - 1]:
                    # Top left
                    previous[row][col] = F_TOP_LEFT
                elif errors[row][col] == dist + errors[row][col - 1]:
                    # Left
                    previous[row][col] = F_LEFT
                elif errors[row][col] == dist + errors[row - 1][col]:
                    # Top
                    previous[row][col] = F_TOP
                elif not is_last_one and errors[row][col] == dist + errors[row - 1][col + 1]:
                    # Top right
                    previous[row][col] = F_TOP_RIGHT
            else:
                errors[row][col] = dist + errors[row][col - 1]
                # Left
                previous[row][col] = F_LEFT
        return errors, previous

    def _fill_column(
        self, col: int, im1: Image, im2: Image, errors: npt.NDArray, previous: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Iterate through one column and fill in values
        """
        assert col < errors.shape[1]
        # Go through rows
        for row in range(col + 1, errors.shape[0]):
            dist = np.sqrt(sum(abs(int(x) - int(y)) for x, y in zip(im1[row][col], im2[row][col])))

            if col > 0:
                # Check to not run out of bounds
                is_last_one = False
                if row < errors.shape[0] - 1:
                    errors[row][col] = dist + min(
                        errors[row - 1][col],
                        errors[row - 1][col - 1],
                        errors[row][col - 1],
                        errors[row + 1][col - 1],
                    )
                else:
                    errors[row][col] = dist + min(errors[row - 1][col], errors[row - 1][col - 1], errors[row][col - 1])
                    is_last_one = True
                if errors[row][col] == dist + errors[row - 1][col]:
                    # Top
                    previous[row][col] = F_TOP
                elif errors[row][col] == dist + errors[row - 1][col - 1]:
                    # Top left
                    previous[row][col] = F_TOP_LEFT
                elif errors[row][col] == dist + errors[row][col - 1]:
                    # Left
                    previous[row][col] = F_LEFT
                elif not is_last_one and errors[row][col] == dist + errors[row + 1][col - 1]:
                    # Bottom left
                    previous[row][col] = F_BOT_LEFT
            else:
                errors[row][col] = dist + errors[row - 1][col]
                # Top
                previous[row][col] = F_TOP
        return errors, previous

    def _find_bool_matrix(self, previous: npt.NDArray) -> SeamMatrix:
        """
        Does the last seam finding step by calculating the perfect seam from the found path.
        This is to be considered going back the best path.
        """
        finished = False
        current_x = previous.shape[1] - 1
        current_y = previous.shape[0] - 1
        bool_matrix = SeamMatrix(np.full(previous.shape, False))
        while not finished:
            bool_matrix[current_y][current_x] = True
            if previous[current_y][current_x] == F_BOT_LEFT:
                current_x -= 1
                current_y += 1
            elif previous[current_y][current_x] == F_LEFT:
                current_x -= 1
            elif previous[current_y][current_x] == F_TOP_LEFT:
                current_x -= 1
                current_y -= 1
            elif previous[current_y][current_x] == F_TOP:
                current_y -= 1
            elif previous[current_y][current_x] == F_TOP_RIGHT:
                current_x += 1
                current_y -= 1
            elif previous[current_y][current_x] == END:
                finished = True
        return bool_matrix

    def _create_mask_from_seam(self, seammat: SeamMatrix, im1: Image, im2: Image) -> Mask:
        """
        Creates a mask for the merged images with values between 0 and 1. 1 is for fully filling the left image,
        0 is for fully filling the right image. Everything else is in between.
        """
        assert im1.shape == im2.shape
        mask_mat = np.ones_like(im1)

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
        yoff, xoff = 0, 0
        for row in range(seammat.shape[0]):
            for col in range(seammat.shape[1]):
                if not seammat[row][col]:
                    mask_mat[row + yoff][col + xoff] = [0.0, 0.0, 0.0]
        # stitch.show_image('Mask', mask_mat)

        # Add a bit of gauss and return it
        return cv2.GaussianBlur(
            mask_mat,
            (GAUSS_SIZE, GAUSS_SIZE),
            0,
            sigmaY=0,
            borderType=cv2.BORDER_REPLICATE,
        )

    def _flood_fill(
        self, seammat: SeamMatrix, xmax: int, ymax: int, x: int, y: int, previous_val: bool, new_val: bool
    ) -> None:
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

    def _is_valid(
        self, seammat: SeamMatrix, xmax: int, ymax: int, x: int, y: int, previous_val: bool, new_val: bool
    ) -> bool:
        """
        Helping function for the floodfill that checks, whether a given pixel can be overridden.
        """
        return (
            x >= 0 and x < xmax and y >= 0 and y < ymax and seammat[x][y] == previous_val and seammat[x][y] != new_val
        )
