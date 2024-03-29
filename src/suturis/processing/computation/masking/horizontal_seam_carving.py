import logging as log
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.timer import track_timings
from suturis.typing import Image, Mask, NpSize, SeamMatrix

# Max value in energy map. Lab in float32 has ranges in L from 0 to 100, and a/b from -127 to 127.
# L2 distance between min and max gives around 373
MAX_ENERGY_VALUE = np.sqrt(np.sum(np.square(np.array([0, -127, -127]) - np.array([100, 127, 127]))))

EnergyMatrix = npt.NDArray[np.float64]
CostMatrix = npt.NDArray[np.float64]
PathMatrix = npt.NDArray[np.int8]


class HorizontalSeamCarving(BaseMaskingHandler):
    """Masking handler which finds an low energy (least difference in color) seam from left to right."""

    half_window_size: int
    gauss_size: int
    yrange: tuple[float, float] | None

    def __init__(
        self, half_window_size: int = 3, gauss_size: int = 17, yrange: tuple[float, float] | None = None, **kwargs: Any
    ):
        """Creates new horizontal seam carving handler.

        Parameters
        ----------
        half_window_size : int, optional
            When using dynamic programming to find the optimal seam,
            the algorithm checks half_window_size rows above and below, by default 3
        gauss_size : int, optional
            Kernel size of gaussian blur after mask finding to blur the seam, by default 17
        yrange : tuple[float, float] | None, optional
            Range of y coordiantes ranging from 0 to 1 to fixate the seam between two rows, by default None
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init Seam Carving Masking Handler")
        super().__init__(**kwargs)
        self.half_window_size = half_window_size
        self.gauss_size = gauss_size
        self.yrange = yrange

    def _compute_mask(self, img1: Image, img2: Image) -> Mask:
        """Computation of the mask with the seam carving algorithm.

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

        if self._debugging_enabled:
            log.debug("Write hsc debug images")
            cv2.imwrite(f"{self._debug_path}hsc_energy_map.jpg", (energy / MAX_ENERGY_VALUE * 255).astype(np.uint8))

        log.debug("Add blur to mask")
        blurred_mask = cv2.GaussianBlur(
            img_mask, (self.gauss_size, self.gauss_size), 0, borderType=cv2.BORDER_REPLICATE
        )
        return Mask(blurred_mask)

    def _get_energy(self, img1: Image, img2: Image) -> EnergyMatrix:
        """Creates energy map between both images, energy meaning color difference (higher energy = less similar colors)

        Parameters
        ----------
        img1 : Image
            Transformed and cropped first image
        img2 : Image
            Transformed and cropped second image

        Returns
        -------
        EnergyMatrix
            Energy map. Values outside the given y-range will be set to a max value.
        """
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

    def _compute_costs(self, output_size: NpSize, energy: EnergyMatrix) -> tuple[CostMatrix, PathMatrix]:
        """Uses dynamic programming to find route from left to right with minimal cumulative energy.

        Parameters
        ----------
        output_size : NpSize
            Size of the images and the energy map
        energy : EnergyMatrix
            Energy map computed for images

        Returns
        -------
        tuple[CostMatrix, PathMatrix]
            First matrix contains accumulated energies (increasing going left), meaning a cell value describes the
            total min energy needed to reach the right border. Second matrix contains offset in rows to achieve the
            minimum energy path, meaning a cell value describes the offset in y to take for next column.
        """
        # Prepare arrays
        height, width = output_size
        costs = np.empty(shape=output_size, dtype=np.float64)
        paths = np.empty(shape=(height, width - 1), dtype=np.int8)
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

    def _trace_back(self, cost_matrix: CostMatrix, path_matrix: PathMatrix) -> SeamMatrix:
        """Given the cost and path matrix, we can back track the optimal path to find the seam.

        Parameters
        ----------
        cost_matrix : CostMatrix
            Cost matrix containing accumulated energy values
        path_matrix : PathMatrix
            Path matrix containing row offsets to find minimum path

        Returns
        -------
        SeamMatrix
            Boolean matrix describing the seam.
        """
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
        """Converts seam matrix (with seam as boolean values) to float mask which can be applied to images

        Parameters
        ----------
        bool_mask : SeamMatrix
            The seam matrix with booleans.

        Returns
        -------
        Mask
            Mask usable for images.
        """
        # Given a bool matrix for each pixel, turns into mask (adding third dimension for 3 color channels)
        stacked = np.stack([bool_mask for _ in range(3)], axis=-1)
        return Mask(stacked.astype(np.float32))

    @track_timings(name="Mask Application")
    def apply_mask(self, img1: Image, img2: Image, mask: Mask) -> Image:
        """Applies mask to transformed images to create stitched result.

        Parameters
        ----------
        img1 : Image
            First input image, transformed and cropped
        img2 : Image
            Second input image, transformed and cropped
        mask : Mask
            The mask to use. Values correspond to the percentage to be used of first image.

        Returns
        -------
        Image
            Stitched image created by the mask.
        """
        # Custom method, because big areas can be copied instead of explicit calculation
        log.debug("Apply horizontal seam carving mask to images")

        # If there's no restriction, use base method
        if not self.yrange:
            return super().apply_mask(img1, img2, mask)

        # Copy areas
        height = mask.shape[0]
        top, bot = int(self.yrange[0] * height), int(self.yrange[1] * height)

        final = np.zeros_like(mask)
        top_img = img1 if self.invert else img2
        bot_img = img2 if self.invert else img1
        final[:top, :] = top_img[:top, :]
        final[bot:, :] = bot_img[bot:, :]

        # Only compute around seam
        img1_section = img1[top:bot, :]
        img2_section = img2[top:bot, :]
        mask_section = mask[top:bot, :]
        final[top:bot, :] = img1_section * mask_section + img2_section * (1 - mask_section)

        return Image(final.astype(np.uint8))
