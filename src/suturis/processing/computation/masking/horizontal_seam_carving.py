import logging as log

import cv2
import numpy as np
import numpy.typing as npt
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.typing import Image, Mask, NpSize, TranslationVector


class HorizontalSeamCarving(BaseMaskingHandler):
    half_window_size: int
    gauss_size: int

    def __init__(
        self,
        continous_recomputation: bool,
        half_window_size: int = 3,
        gauss_size: int = 17,
    ):
        log.debug("Init Seam Carving Masking Handler")
        super().__init__(continous_recomputation)
        self.half_window_size = half_window_size
        self.gauss_size = gauss_size

    def _compute_mask(self, img1: Image, img2: Image, output_size: NpSize) -> Mask:
        energy = self._get_energy(img1, img2)
        cost_matrix, path_matrix = self._compute_costs(output_size, energy)
        bool_mask = self._trace_back(cost_matrix, path_matrix)
        img_mask = self._bool_to_img_mask(bool_mask)
        blurred_mask = cv2.GaussianBlur(
            img_mask, (self.gauss_size, self.gauss_size), 0, borderType=cv2.BORDER_REPLICATE
        )
        return Mask(blurred_mask)

    def _get_energy(self, img1: Image, img2: Image) -> npt.NDArray:
        # Convert to Lab color space (and to int32 to avoid over/underflow)
        img1_lab = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.int32)
        img2_lab = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.int32)

        # Compute L2 distance of both pixel in each image
        diff = np.sqrt(np.sum(np.abs(img1_lab - img2_lab), axis=2))
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

    def _trace_back(self, cost_matrix: npt.NDArray, path_matrix: npt.NDArray) -> npt.NDArray:
        curr = int(np.argmin(cost_matrix[:, 0]))
        bool_matrix = np.zeros_like(cost_matrix, dtype=bool)
        bool_matrix[curr:, 0] = True
        for col in range(path_matrix.shape[1]):
            offset = path_matrix[curr, col]
            curr = curr + offset
            bool_matrix[curr:, col + 1] = True

        return bool_matrix

    def _bool_to_img_mask(self, bool_mask: npt.NDArray) -> npt.NDArray:
        # Given a bool matrix for each pixel, turns into mask (adding third dimension for 3 color channels)
        stacked = np.stack([bool_mask for _ in range(3)], axis=-1)
        return stacked.astype(np.float64)
