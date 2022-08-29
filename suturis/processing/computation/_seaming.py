import numpy as np
import cv2


def find_important_pixels(orig_img, hom, trans):
    """
    Finds start and end pixel using euclidean measure. Start is pixel with lowest euclidean distance to the origin,
    end is pixel with highest euclidean distance to origin. Big performance issues, but works correctly so far.
    """
    #        Top left       Top right          Bottom left            Bottom right
    points = np.array(
        [
            [0, 0, 1],
            [orig_img.shape[1] - 1, 0, 1],
            [0, orig_img.shape[0] - 1, 1],
            [orig_img.shape[1] - 1, orig_img.shape[0] - 1, 1],
        ]
    )
    hom_points, trans_points = [], []
    # Apply transformations to all of those corner points
    for pts in points:
        # Warp the points
        tmp = cv2.perspectiveTransform(
            np.array([[[pts[0], pts[1]]]], dtype=np.float32), hom
        )
        # Add the translation
        tmp = np.matmul(trans, np.array([tmp[0][0][0], tmp[0][0][1], 1]))
        hom_points = np.concatenate((hom_points, tmp))
        trans_points = np.concatenate((trans_points, np.matmul(trans, pts)))

    # Calculating the perfect corner points
    start = (
        int(round(max(min(hom_points[1::3]), min(trans_points[1::3])))),
        int(round(max(min(hom_points[0::3]), min(trans_points[0::3])))),
    )
    end = (
        int(round(min(max(hom_points[1::3]), max(trans_points[1::3])))),
        int(round(min(max(hom_points[0::3]), max(trans_points[0::3])))),
    )
    return start, end


def find_seam_dynamically(im1, im2, start, end):
    """
    Finds an optimal seam using the dynamic approach. Error computes with the squared differences of each of the L*a*b*
    values
    """
    assert im1.shape == im2.shape

    # Convert color into Lab space, because we would like to follow EN ISO 11664-4
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2Lab)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2Lab)

    # Doing some initializations
    x_off = start[1]
    y_off = start[0]
    errors = np.zeros((end[0] - start[0], end[1] - start[1]))
    previous = np.zeros((end[0] - start[0], end[1] - start[1]))

    # Calculate distances
    for idx in range(0, min(errors.shape[0], errors.shape[1])):
        dist = np.sqrt(
            sum(
                [
                    abs(int(x) - int(y))
                    for x, y in zip(
                        im1[idx + y_off][idx + x_off], im2[idx + y_off][idx + x_off]
                    )
                ]
            )
        )

        is_last_col, is_last_row = False, False
        # Check whether we are at an edge
        if idx == 0:
            # Top left corner. Nothing to do here but put the distance in and indicate the end
            errors[idx][idx] = dist
            previous[idx][idx] = 0
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
                errors[idx][idx] = dist + min(
                    errors[idx - 1][idx - 1], errors[idx][idx - 1], errors[idx - 1][idx]
                )
                is_last_row = True
                is_last_col = True

            # Where is the value from you ask? Let's find that out!
            if errors[idx][idx] == dist + errors[idx - 1][idx - 1]:
                # Top left
                previous[idx][idx] = 3
            elif errors[idx][idx] == dist + errors[idx][idx - 1]:
                # Left
                previous[idx][idx] = 2
            elif errors[idx][idx] == dist + errors[idx - 1][idx]:
                # Top
                previous[idx][idx] = 4
            elif (
                not is_last_row and errors[idx][idx] == dist + errors[idx + 1][idx - 1]
            ):
                # Bottom left
                previous[idx][idx] = 1
            elif (
                not is_last_col and errors[idx][idx] == dist + errors[idx - 1][idx + 1]
            ):
                # Top right
                previous[idx][idx] = 5

        if is_last_col:
            # Now we reached the end of the columns so we only need to fill that column, then we're done
            errors, previous = _fill_column(
                idx, im1, im2, errors, y_off, x_off, previous
            )
        elif is_last_row:
            # Now we reached the end of the rows, so we only need to fill that row before we're done
            errors, previous = _fill_row(idx, im1, im2, errors, y_off, x_off, previous)
        else:
            # We have to fill both rows and columns
            errors, previous = _fill_row(idx, im1, im2, errors, y_off, x_off, previous)
            errors, previous = _fill_column(
                idx, im1, im2, errors, y_off, x_off, previous
            )
    return _find_bool_matrix(previous)


def prepare_img_for_seam_finding(base, query, base_seam, offset):
    modified_img = base.copy()
    y_off, x_off = offset
    for (x, y) in base_seam:
        modified_img[y + y_off][x + x_off] = query[y + y_off][x + x_off]

    return modified_img


def _fill_row(row, im1, im2, errors, y_off, x_off, previous):
    """
    Iterate through one row and fill in values
    """
    assert (
        row < errors.shape[0]
    ), f"idx is: {row}, errors.shape[0] is: {errors.shape[0]}"
    # Go through columns
    for col in range(row + 1, errors.shape[1]):
        dist = np.sqrt(
            sum(
                [
                    abs(int(x) - int(y))
                    for x, y in zip(
                        im1[row + y_off][col + x_off], im2[row + y_off][col + x_off]
                    )
                ]
            )
        )
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
                errors[row][col] = dist + min(
                    errors[row - 1][col - 1], errors[row][col - 1], errors[row - 1][col]
                )
            if errors[row][col] == dist + errors[row - 1][col - 1]:
                # Top left
                previous[row][col] = 3
            elif errors[row][col] == dist + errors[row][col - 1]:
                # Left
                previous[row][col] = 2
            elif errors[row][col] == dist + errors[row - 1][col]:
                # Top
                previous[row][col] = 4
            elif (
                not is_last_one and errors[row][col] == dist + errors[row - 1][col + 1]
            ):
                # Top right
                previous[row][col] = 5
        else:
            errors[row][col] = dist + errors[row][col - 1]
            # Left
            previous[row][col] = 2
    return errors, previous


def _fill_column(col, im1, im2, errors, yoff, xoff, previous):
    """
    Iterate through one column and fill in values
    """
    assert col < errors.shape[1]
    # Go through rows
    for row in range(col + 1, errors.shape[0]):
        dist = np.sqrt(
            sum(
                [
                    abs(int(x) - int(y))
                    for x, y in zip(
                        im1[row + yoff][col + xoff], im2[row + yoff][col + xoff]
                    )
                ]
            )
        )
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
                errors[row][col] = dist + min(
                    errors[row - 1][col], errors[row - 1][col - 1], errors[row][col - 1]
                )
                is_last_one = True

            if errors[row][col] == dist + errors[row - 1][col]:
                # Top
                previous[row][col] = 4
            elif errors[row][col] == dist + errors[row - 1][col - 1]:
                # Top left
                previous[row][col] = 3
            elif errors[row][col] == dist + errors[row][col - 1]:
                # Left
                previous[row][col] = 2
            elif (
                not is_last_one and errors[row][col] == dist + errors[row + 1][col - 1]
            ):
                # Bottom left
                previous[row][col] = 1
        else:
            errors[row][col] = dist + errors[row - 1][col]
            # Top
            previous[row][col] = 4
    return errors, previous


def _find_bool_matrix(previous):
    """
    Does the last seam finding step by calculating the perfect seam from the found path
    """
    finished = False
    current_x = previous.shape[1] - 1
    current_y = previous.shape[0] - 1
    bool_matrix = np.full(previous.shape, False)
    while not finished:
        bool_matrix[current_y][current_x] = True
        if previous[current_y][current_x] == 1:
            current_x -= 1
            current_y += 1
        elif previous[current_y][current_x] == 2:
            current_x -= 1
        elif previous[current_y][current_x] == 3:
            current_x -= 1
            current_y -= 1
        elif previous[current_y][current_x] == 4:
            current_y -= 1
        elif previous[current_y][current_x] == 5:
            current_x += 1
            current_y -= 1
        elif previous[current_y][current_x] == 0:
            finished = True
    return bool_matrix


def _create_seam_image(xoff, yoff, im1, im2, seammat, x_max, y_max):
    """
    Creates an image with a hard seam along a previously calculated seam.
    """
    seam_image = np.zeros_like(im2)
    for i in range(0, im2.shape[0]):
        is_overlapping_row = y_max >= i >= yoff
        first_image = True
        for j in range(0, im2.shape[1]):
            if (
                is_overlapping_row
                and i >= yoff
                and j >= xoff
                and i < y_max
                and j < x_max
                and not seammat[i - yoff][j - xoff]
            ):
                if j > xoff and seammat[i - yoff][j - xoff - 1]:
                    first_image = not first_image

            if is_overlapping_row and first_image:
                seam_image[i][j] = im1[i][j]
            elif is_overlapping_row and not first_image:
                seam_image[i][j] = im2[i][j]
            else:
                seam_image[i][j] = im1[i][j] + im2[i][j]
    return seam_image
