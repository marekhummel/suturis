from time import perf_counter
import numpy as np

WINDOW_SIZE = 1
END = 0
LEFT_LEFT = 1
LEFT = 2
BOTTOM = 3
RIGHT = 4
RIGHT_RIGHT = 5

LOW_LAB_IN_BGR = [195, 59, 0]
HIGH_LAB_IN_BGR = [0, 62, 255]

GAUSS_SIZE = 17


def _compute_costs(energy):
    height, width = energy.shape
    energy = energy.astype(np.float64)
    energy[:3] = 5000
    energy[-3:] = 5000
    costs = np.empty(shape=energy.shape)
    paths = np.empty(shape=(energy.shape[0], energy.shape[1] - 1), dtype=np.int32)
    costs[:, -1] = energy[:, -1]  # start of with given energy in last col

    for col in range(width - 2, -1, -1):
        prev_column = costs[:, col + 1]

        # Create list of windows to analyze by shifting the column up and down
        window_list = []
        for offset in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            sliced_column = prev_column[max(0, offset) : min(height + offset, height)]
            column = np.pad(sliced_column, (max(0, -offset), max(0, offset)), constant_values=np.nan)
            window_list.append(column)
        window_matrix = np.column_stack(window_list)

        # Compute min value along all those shifted columns for each row
        argmin_values = np.nanargmin(window_matrix, axis=1)
        min_values = window_matrix[np.array(range(height)), argmin_values]
        costs[:, col] = energy[:, col] + min_values
        paths[:, col] = argmin_values - WINDOW_SIZE  # column now stores the offset to use for next column

    return costs, paths


def _trace_back(costs, paths):
    curr = np.argmin(costs[:, 0])
    bool_matrix = np.zeros_like(costs, dtype=bool)
    bool_matrix[curr, 0] = True
    # path = [(curr, 0)]
    for col in range(paths.shape[1]):
        offset = paths[curr, col]
        curr = curr + offset
        # path.append((curr, col + 1))
        bool_matrix[curr, col + 1] = True

    return bool_matrix


def _find_seam(dif):
    xstart, ystart = (0, 0)
    yend, xend = dif.shape
    previous = np.zeros((yend - ystart, xend - xstart))
    paths = np.zeros_like(previous)
    for row in range(yend - ystart):
        previous, paths = _fill_row(paths, yend - ystart - row - 1, dif, previous)
    return _find_bool_matrix(previous, paths)


def _fill_row(paths, row, dif, previous):
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


def _find_bool_matrix(previous, paths):
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


# energy = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [5, 3, 1, 5, 6], [6, 1, 2, 4, 5], [12, 4, 3, 1, 2]])
energy = np.random.rand(8, 8)
# energy = np.array(
#     [
#         [0.62577254, 0.70949754, 0.30096836, 0.58417793, 0.64873411],
#         [0.96956316, 0.30620459, 0.79445889, 0.31212922, 0.85419942],
#         [0.92722623, 0.30542461, 0.17316804, 0.26564466, 0.14052509],
#         [0.79230725, 0.43473213, 0.19283791, 0.59372231, 0.7563433],
#         [0.55234212, 0.22798053, 0.55005411, 0.17497752, 0.84080138],
#     ]
# )

print(energy)
start = perf_counter()
costs, paths = _compute_costs(energy)
path = _trace_back(costs, paths)

print(costs)
print(paths)
print(path)
print(perf_counter() - start)


# print()
# print()
# print()

# energy_t = np.transpose(energy)
# print(energy_t)
# start = perf_counter()
# seam = _find_seam(energy_t)
# print(np.transpose(seam))
# print(perf_counter() - start)

# [6, 6, 5, 2, nan]
# [5, 6, 6, 5, 2]
# [nan, 5, 6, 6, 5]
