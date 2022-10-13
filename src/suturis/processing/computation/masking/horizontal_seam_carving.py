import logging as log

import cv2
import numpy as np
import numpy.typing as npt
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask, NpSize, SeamMatrix

# Max value in energy map. Lab in float32 has ranges in L from 0 to 100, and a/b from -127 to 127.
# L2 distance between min and max gives around 373
MAX_ENERGY_VALUE = np.sqrt(np.sum(np.square(np.array([0, -127, -127]) - np.array([100, 127, 127]))))


class HorizontalSeamCarving(BaseMaskingHandler):
    half_window_size: int
    gauss_size: int
    yrange: tuple[float, float] | None

    def __init__(
        self,
        continous_recomputation: bool,
        save_to_file: bool = False,
        invert: bool = False,
        half_window_size: int = 3,
        gauss_size: int = 17,
        yrange: tuple[float, float] | None = None,
    ):
        log.debug("Init Seam Carving Masking Handler")
        super().__init__(continous_recomputation, save_to_file, invert)
        self.half_window_size = half_window_size
        self.gauss_size = gauss_size
        self.yrange = yrange

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        log.debug("Compute new mask")
        output_size = NpSize((img1.shape[0], img1.shape[1]))

        log.debug("Get energy map from transformed images")
        energy = self._get_energy(img1, img2)

        log.debug("Deriving cost map from energy map")
        cost_matrix, path_matrix = self._compute_costs(output_size, energy)

        log.debug("Use cost matrix to compute path")
        bool_mask = self._trace_back(cost_matrix, path_matrix)

        log.debug("Compute image mask from path")
        img_mask = self._bool_to_img_mask(bool_mask)

        log.debug("Add blur to mask")
        blurred_mask = cv2.GaussianBlur(
            img_mask, (self.gauss_size, self.gauss_size), 0, borderType=cv2.BORDER_REPLICATE
        )
        return Mask(blurred_mask)

    def _get_energy(self, img1: Image, img2: Image) -> npt.NDArray[np.float64]:
        # Convert to Lab color space (and to float32 for better accuracy)
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab).astype(np.float32)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab).astype(np.float32)

        # Compute L2 distance of both pixels in each image
        diff = np.sqrt(np.sum(np.square(img1_lab - img2_lab), axis=2))

        # Block off some values to restrict the path by setting them to the max possible value
        if self.yrange is not None:
            log.debug(f"Set rows outside of {self.yrange} to max value")
            ystart, yend = np.array(self.yrange) * diff.shape[0]
            diff[: int(ystart), :] = MAX_ENERGY_VALUE
            diff[int(yend + 1) :, :] = MAX_ENERGY_VALUE

        # Return
        return diff

    def _compute_costs(self, output_size: NpSize, energy: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        # Prepare arrays
        height, width = output_size
        costs = np.empty(shape=output_size)
        paths = np.empty(shape=(height, width - 1), dtype=np.int32)
        costs[:, -1] = energy[:, -1]  # start of with given energy in last col

        # Iterate from second to last col up to first col for dyn prog
        for col in range(width - 2, -1, -1):
            prev_column = costs[:, col + 1]

            # Create list of windows to analyze by shifting the column up and down
            window_list = []
            for offset in range(-self.half_window_size, self.half_window_size + 1):
                sliced_column = prev_column[max(0, offset) : min(height + offset, height)]
                column = np.pad(sliced_column, (max(0, -offset), max(0, offset)), constant_values=np.nan)
                window_list.append(column)
            window_matrix = np.column_stack(window_list)

            # Compute min value along all those shifted columns for each row
            argmin_values = np.nanargmin(window_matrix, axis=1)
            min_values = window_matrix[np.array(range(height)), argmin_values]
            costs[:, col] = energy[:, col] + min_values
            paths[:, col] = argmin_values - self.half_window_size  # column now stores the offset to use for next column

        return costs, paths

    def _trace_back(self, cost_matrix: npt.NDArray, path_matrix: npt.NDArray) -> SeamMatrix:
        curr = int(np.argmin(cost_matrix[:, 0]))
        bool_matrix = SeamMatrix(np.zeros_like(cost_matrix, dtype=bool))
        bool_matrix[curr:, 0] = True
        path = [curr]
        for col in range(path_matrix.shape[1]):
            offset = path_matrix[curr, col]
            curr = curr + offset
            path.append(curr)
            bool_matrix[curr:, col + 1] = True
        return bool_matrix

    def _bool_to_img_mask(self, bool_mask: SeamMatrix) -> Mask:
        # Given a bool matrix for each pixel, turns into mask (adding third dimension for 3 color channels)
        stacked = np.stack([bool_mask for _ in range(3)], axis=-1)
        return Mask(stacked.astype(np.float64))
