"""Custom type definition."""

from typing import Any, Callable, Dict, NewType, Tuple, Union

import kornia.augmentation as K
import torch

SizePx = NewType("SizePx", Tuple[int, int])
SizeMM = NewType("SizeMM", Tuple[float, float])
SizePatch = NewType("SizePatch", Tuple[int, float, float])

# ImageTensor has shape [3, height, width]
ImageTensor = NewType("ImageTensor", torch.FloatTensor)
# MaskTensor has shape [1, height, width]
MaskTensor = Union[torch.FloatTensor, torch.BoolTensor]
# ImageTensorRGBA has shape [4, height, width]
ImageTensorRGBA = NewType("ImageTensorRGBA", torch.FloatTensor)
# Image tensor of any number of channels
ImageTensorGeneric = Union[ImageTensor, MaskTensor, ImageTensorRGBA]
# Detectron2 input image format (can be BGR + [0, 255])
ImageTensorDet = NewType("ImageTensorDet", torch.FloatTensor)

# BatchImageTensor has shape [batch, 3, height, width]
BatchImageTensor = NewType("BatchImageTensor", torch.FloatTensor)
# BatchMaskTensor has shape [batch, 1, height, width]
BatchMaskTensor = Union[torch.FloatTensor, torch.BoolTensor]
# BatchImageTensorRGBA has shape [batch, 4, height, width]
BatchImageTensorRGBA = NewType("BatchImageTensorRGBA", torch.FloatTensor)
# Image tensor of any number of channels
BatchImageTensorGeneric = Union[
    BatchImageTensor, BatchMaskTensor, BatchImageTensorRGBA
]

# Transform function (both geometric and lighting)
TransformFn = Union[
    Callable[[BatchImageTensorGeneric], BatchImageTensorGeneric],
    K.GeometricAugmentationBase2D,
    K.IntensityAugmentationBase2D,
]
# Transform function that also returns params
TransformParamFn = Callable[
    [BatchImageTensorGeneric], Tuple[BatchImageTensorGeneric, torch.Tensor]
]

# TODO(YOLO): Unify target type for detectron and YOLO
Target = Union[Dict[str, Any], Any]

DetectronSample = NewType("DetectronSample", Dict[str, Any])
