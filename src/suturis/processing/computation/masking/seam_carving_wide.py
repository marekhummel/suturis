import numpy as np
from suturis.processing.computation.masking import SeamCarving
from suturis.typing import Matrix, SeamMatrix

END = 0
LEFT_LEFT = 1
LEFT = 2
BOTTOM = 3
RIGHT = 4
RIGHT_RIGHT = 5

LOW_LAB_IN_BGR = [195, 59, 0]
HIGH_LAB_IN_BGR = [0, 62, 255]

GAUSS_SIZE = 17


class SeamCarvingWide(SeamCarving):
    """Masking handler which finds an low energy (least difference in color) seam from top to bottom,
    improved by using wider angles."""

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
                if 1 < col < dif.shape[1] - 2:
                    # Just somewhere in the middle
                    left_left = paths[row + 1][col - 2]
                    left = paths[row + 1][col - 1]
                    down = paths[row + 1][col]
                    right = paths[row + 1][col + 1]
                    right_right = paths[row + 1][col + 2]

                    m = min(left_left, left, down, right, right_right)
                    # Get a new value
                    paths[row][col] = dif[row][col] + m
                    # Determine which value we used
                    if m == left_left:
                        previous[row][col] = LEFT_LEFT
                    elif m == left:
                        previous[row][col] = LEFT
                    elif m == down:
                        previous[row][col] = BOTTOM
                    elif m == right:
                        previous[row][col] = RIGHT
                    else:
                        previous[row][col] = RIGHT_RIGHT
                elif col == 1:
                    # 1 off left side of the image.
                    left = paths[row + 1][col - 1]
                    down = paths[row + 1][col]
                    right = paths[row + 1][col + 1]
                    right_right = paths[row + 1][col + 2]
                    m = min(left, down, right, right_right)
                    paths[row][col] = dif[row][col] + m
                    if m == left:
                        previous[row][col] = LEFT
                    elif m == down:
                        previous[row][col] = BOTTOM
                    elif m == right:
                        previous[row][col] = RIGHT
                    else:
                        previous[row][col] = RIGHT_RIGHT
                elif col == 0:
                    # Very left side of image
                    down = paths[row + 1][col]
                    right = paths[row + 1][col + 1]
                    right_right = paths[row + 1][col + 2]
                    m = min(down, right, right_right)
                    paths[row][col] = dif[row][col] + m
                    if m == down:
                        previous[row][col] = BOTTOM
                    elif m == right:
                        previous[row][col] = RIGHT
                    else:
                        previous[row][col] = RIGHT_RIGHT
                elif col < dif.shape[1] - 1:
                    # Nearly Right end of image
                    left_left = paths[row + 1][col - 2]
                    left = paths[row + 1][col - 1]
                    down = paths[row + 1][col]
                    right = paths[row + 1][col + 1]

                    m = min(left_left, left, down, right)
                    # Get a new value
                    paths[row][col] = dif[row][col] + m
                    # Determine which value we used
                    if m == left_left:
                        previous[row][col] = LEFT_LEFT
                    elif m == left:
                        previous[row][col] = LEFT
                    elif m == down:
                        previous[row][col] = BOTTOM
                    else:
                        previous[row][col] = RIGHT
                elif col < dif.shape[1]:
                    # Nearly Right end of image
                    left_left = paths[row + 1][col - 2]
                    left = paths[row + 1][col - 1]
                    down = paths[row + 1][col]

                    m = min(left_left, left, down)
                    # Get a new value
                    paths[row][col] = dif[row][col] + m
                    # Determine which value we used
                    if m == left_left:
                        previous[row][col] = LEFT_LEFT
                    elif m == left:
                        previous[row][col] = LEFT
                    else:
                        previous[row][col] = BOTTOM
                else:
                    # Big logical trouble
                    assert False, f"col is: {col}"

        return previous, paths

    def _find_bool_matrix(self, previous: Matrix, paths: Matrix) -> SeamMatrix:
        """Finds a bool matrix for the seam for better angles.

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
            elif val == LEFT_LEFT:
                # Making two steps to the left
                index -= 2
            elif val == LEFT:
                # Making a step to the left
                index -= 1
            elif val == RIGHT:
                index += 1
            elif val == RIGHT_RIGHT:
                index += 2
            # Moving a step down
            current_row += 1
        return bools
