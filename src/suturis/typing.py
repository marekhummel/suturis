import numpy.typing as npt

Image = npt.NDArray
NpPoint = tuple[int, int]
CvPoint = tuple[int, int]
NpSize = tuple[int, int]
CvSize = tuple[int, int]

Homography = npt.NDArray
TranslationVector = tuple[int, int]
Mask = npt.NDArray
SeamMatrix = npt.NDArray

WarpingInfo = tuple[TranslationVector, CvSize, Homography]
CropArea = tuple[NpPoint, NpPoint]
ComputationParams = tuple[WarpingInfo, CropArea, Mask]
