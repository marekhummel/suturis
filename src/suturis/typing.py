from typing import NewType

import numpy as np
import numpy.typing as npt

# Basic tuples, NP means (y, x), CV means (X, Y)
NpShape = tuple[int, ...]
NpPoint = NewType("NpPoint", tuple[int, int])
NpSize = NewType("NpSize", tuple[int, int])
NpRect = tuple[NpPoint, NpPoint]
CvPoint = NewType("CvPoint", tuple[int, int])
CvSize = NewType("CvSize", tuple[int, int])
CvRect = tuple[CvPoint, CvPoint]

# New types for various arrays to avoid confusion in passing
Image = NewType("Image", npt.NDArray[np.uint8])
Homography = NewType("Homography", npt.NDArray[np.float64])
TranslationVector = NewType("TranslationVector", tuple[int, int])
Mask = NewType("Mask", npt.NDArray[np.float64])
SeamMatrix = NewType("SeamMatrix", npt.NDArray[np.bool_])

# Aliases for readability
ImagePair = tuple[Image, Image]
CanvasSize = CvSize
TransformationInfo = tuple[CanvasSize, TranslationVector, Homography]
CanvasInfo = tuple[CanvasSize, TranslationVector]
ComputationParams = tuple[TransformationInfo, Mask]
