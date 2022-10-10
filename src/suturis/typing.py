from typing import NewType

import numpy as np
import numpy.typing as npt

NpShape = tuple[int, ...]
NpPoint = NewType("NpPoint", tuple[int, int])
NpSize = NewType("NpSize", tuple[int, int])
CvPoint = NewType("CvPoint", tuple[int, int])
CvSize = NewType("CvSize", tuple[int, int])
NpRect = tuple[NpPoint, NpPoint]
CvRect = tuple[CvPoint, CvPoint]

Image = NewType("Image", npt.NDArray[np.uint8])
Homography = NewType("Homography", npt.NDArray[np.float64])
TranslationVector = NewType("TranslationVector", tuple[int, int])
Mask = NewType("Mask", npt.NDArray[np.float64])
SeamMatrix = NewType("SeamMatrix", npt.NDArray[np.bool_])


CropArea = NpRect
CanvasSize = CvSize
TransformationInfo = tuple[CanvasSize, TranslationVector, Homography]
CanvasInfo = tuple[CanvasSize, TranslationVector, CropArea]
ComputationParams = tuple[TransformationInfo, CropArea, Mask]
