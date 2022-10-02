import numpy.typing as npt

Point = tuple[int, int]
Image = npt.NDArray
NpSize = tuple[int, int]
CvSize = tuple[int, int]

Homography = npt.NDArray
TranslationVector = tuple[int, int]
Mask = npt.NDArray
SeamMatrix = npt.NDArray
WarpingInfo = tuple[TranslationVector, CvSize, Homography]
ComputationParams = tuple[WarpingInfo, Mask]
