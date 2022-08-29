import numpy as np
import cv2


def create_binary_mask(
    seammat, im1, im2, xoff, yoff, x_trans, y_trans, result_width, result_height
):
    """
    Creates a mask for the merged images with values between 0 and 1. 1 is for fully filling the left image,
    0 is for fully filling the right image. Everything else is in between.
    """
    assert im1.shape == im2.shape

    mask_mat = np.zeros(im1.shape)

    mask_mat[y_trans : result_height + y_trans, x_trans : result_width + x_trans] = [
        1.0,
        1.0,
        1.0,
    ]

    # Use the flood fill algorithm to fill the seam matrix
    if not seammat[seammat.shape[0] - 1][0]:
        _flood_fill(
            seammat,
            seammat.shape[0],
            seammat.shape[1],
            seammat.shape[0] - 1,
            0,
            False,
            True,
        )
    else:
        for row in range(0, seammat.shape[0]):
            if not seammat[row][0]:
                _flood_fill(
                    seammat, seammat.shape[0], seammat.shape[1], 0, row, False, True
                )

        for col in range(0, seammat.shape[1]):
            if not seammat[seammat.shape[0] - 1][col]:
                _flood_fill(
                    seammat,
                    seammat.shape[0],
                    seammat.shape[1],
                    col,
                    seammat.shape[0] - 1,
                    False,
                    True,
                )

    # Go through seammat area and fill in zeros at the right manually
    for row in range(0, seammat.shape[0]):
        for col in range(0, seammat.shape[1]):
            if not seammat[row][col]:
                mask_mat[row + yoff][col + xoff + 1] = [0.0, 0.0, 0.0]
    # Add a bit of gauss and return it
    # stitch.show_image('Mask', mask_mat)
    return cv2.GaussianBlur(
        mask_mat, (17, 17), 0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )


def _is_valid(seammat, xmax, ymax, x, y, previous_val, new_val):
    """
    Helping function for the floodfill that checks, whether a given pixel can be overridden.
    """
    if (
        x < 0
        or x >= xmax
        or y < 0
        or y >= ymax
        or seammat[x][y] != previous_val
        or seammat[x][y] == new_val
    ):
        return False
    return True


def _flood_fill(seammat, xmax, ymax, x, y, previous_val, new_val):
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
        if _is_valid(seammat, xmax, ymax, posX + 1, posY, previous_val, new_val):
            # Right
            seammat[posX + 1][posY] = new_val
            queue.append([posX + 1, posY])

        if _is_valid(seammat, xmax, ymax, posX - 1, posY, previous_val, new_val):
            # Left
            seammat[posX - 1][posY] = new_val
            queue.append([posX - 1, posY])

        if _is_valid(seammat, xmax, ymax, posX, posY + 1, previous_val, new_val):
            # Bottom
            seammat[posX][posY + 1] = new_val
            queue.append([posX, posY + 1])

        if _is_valid(seammat, xmax, ymax, posX, posY - 1, previous_val, new_val):
            # Top
            seammat[posX][posY - 1] = new_val
            queue.append([posX, posY - 1])


# Methods that aren't used but might be later, if find a better way of doing stuff.


def _is_top(im):
    for col in range(0, im.shape[1]):
        if sum(im[0][col]) != 0:
            return True
    return False


def _is_bottom(im):
    for col in range(0, im.shape[1]):
        if sum(im[im.shape[0] - 1][col]) != 0:
            return True
    return False


def _is_left(im):
    for row in range(0, im.shape[0]):
        if sum(im[row][0]) != 0:
            return True
    return False


def _is_right(im):
    for row in range(0, im.shape[0]):
        if sum(im[row][im.shape[1] - 1]):
            return True
    return False
