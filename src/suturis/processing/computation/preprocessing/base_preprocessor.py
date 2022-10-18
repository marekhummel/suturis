import logging as log

from suturis.processing.computation.base_computation_handler import BaseComputationHandler
from suturis.typing import Image, ImagePair


class BasePreprocessor(BaseComputationHandler[ImagePair]):
    """Base class for preprocessors."""

    index: int
    needed_for_computation: bool

    def __init__(self, index: int, /, needed_for_computation: bool = False) -> None:
        """Create new base preprocessor instance.

        Parameters
        ----------
        index : int
            0-based index of this preprocessor. Given implicitly by list index in config
        needed_for_computation : bool, optional
            Flag to indicate of this preprocessor should be used for computation,
            if not they're only used when applying the params, by default False
        """
        log.debug(f"Init preprocessing handler #{index}, with needed_for_computation set to {needed_for_computation}")
        super().__init__()
        self.index = index
        self.needed_for_computation = needed_for_computation

    def process(self, img1: Image, img2: Image) -> ImagePair:
        """Abstract method to process images and return modified images.

        Parameters
        ----------
        img1 : Image
            First input image, may be modified by previous preprocessors
        img2 : Image
            Second input image, may be modified by previous preprocessors

        Returns
        -------
        ImagePair
            Modified images.

        Raises
        ------
        NotImplementedError
            Unless overriden, this method will raise an error.
        """
        raise NotImplementedError("Abstract method needs to be overriden")
