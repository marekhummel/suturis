import logging as log
from typing import Literal, Union

from suturis.typing import Image

_ReadImageType = Union[tuple[Literal[True], Image], tuple[Literal[False], None]]


class BaseReader:
    """Abstract base class for any type of image input (aka a reader).

    Raises
    ------
    NotImplementedError
        Methods raise this becuase this class is abstract.
    """

    index: int

    def __init__(self, index: int, /) -> None:
        """Creates new reader instance, should not be called explicitly only from subclasses.

        Parameters
        ----------
        index : int
            0-based index of this reader. Given implicitly by list index in config
        """
        log.debug(f"Init reader #{index}")
        self.index = index

    def read_image(self) -> _ReadImageType:
        """Abstract method for reading the next image.

        Raises
        ------
        NotImplementedError
            Unless overriden, this method will raise an error.
        """
        raise NotImplementedError("Abstract method needs to be overriden")
