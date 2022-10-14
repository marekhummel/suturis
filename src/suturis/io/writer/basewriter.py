import enum

from suturis.typing import Image


class SourceImage(enum.IntEnum):
    """Enum to describe image source"""

    OUTPUT = 0
    INPUT_ONE = 1
    INPUT_TWO = 2


class BaseWriter:
    """Abstract base class for any type of image output (aka a writer).

    Raises
    ------
    NotImplementedError
        Methods raise this beause this class is abstract.
    """

    index: int
    source: SourceImage

    def __init__(self, index: int, /, source: str) -> None:
        """Creates new writer instance, should not be called explicitly only from subclasses.

        Parameters
        ----------
        index : int
            0-based index of this writer. Given implicitly by list index in config
        source : str
            Describes what image to write, has to be member of SourceImage
        """
        self.index = index
        self.source = SourceImage[source]

    def write_image(self, img: Image) -> None:
        """Abstract method for writing the given image.

        Parameters
        ----------
        img : Image
            The image to write.

        Raises
        ------
        NotImplementedError
            Unless overriden, this method will raise an error.
        """
        raise NotImplementedError("Abstract method needs to be overriden")
