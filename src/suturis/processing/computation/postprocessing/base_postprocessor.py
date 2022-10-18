import logging as log

from suturis.processing.computation.base_computation_handler import BaseComputationHandler
from suturis.typing import Image


class BasePostprocessor(BaseComputationHandler):
    """Base class for postprocessors."""

    index: int
    needed_for_computation: bool

    def __init__(self, index: int, /) -> None:
        """Create new base postprocessor instance.

        Parameters
        ----------
        index : int
            0-based index of this preprocessor. Given implicitly by list index in config
        """
        log.debug(f"Init postprocessing handler #{index}")
        super().__init__()
        self.index = index

    def process(self, image: Image) -> Image:
        """Abstract method to process image and return modified image.

        Parameters
        ----------
        image: Image
            Stitched image, may be modified by previous postprocessors

        Returns
        -------
        Image
            Modified image.

        Raises
        ------
        NotImplementedError
            Unless overriden, this method will raise an error.
        """
        raise NotImplementedError("Abstract method needs to be overriden")
