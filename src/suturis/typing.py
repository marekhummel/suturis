import numpy.typing as npt
from typing import NewType

NpShape = tuple[int, ...]
NpPoint = NewType("NpPoint", tuple[int, int])
NpSize = NewType("NpSize", tuple[int, int])
CvPoint = NewType("CvPoint", tuple[int, int])
CvSize = NewType("CvSize", tuple[int, int])

Image = NewType("Image", npt.NDArray)
Homography = NewType("Homography", npt.NDArray)
TranslationVector = NewType("TranslationVector", tuple[int, int])
Mask = NewType("Mask", npt.NDArray)
SeamMatrix = NewType("SeamMatrix", npt.NDArray)

WarpingInfo = tuple[TranslationVector, CvSize, Homography]
CropArea = tuple[NpPoint, NpPoint]
ComputationParams = tuple[WarpingInfo, CropArea, Mask]
