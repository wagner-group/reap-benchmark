from turtle import forward
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean, std, *args, **kwargs):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None])
        self.register_buffer('std', torch.tensor(std)[None, :, None, None])

    def forward(self, x):
        return (x - self.mean) / self.std


# class ColorDetectorWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.color_dict = {
#             'circle-750.0': ['white', 'blue', 'red'],   # (1) white+red, (2) blue+white
#             'triangle-900.0': ['white', 'yellow'],  # (1) white, (2) yellow
#             'triangle_inverted-1220.0': [],   # (1) white+red
#             'diamond-600.0': [],    # (1) white+yellow
#             'diamond-915.0': [],    # (1) yellow
#             'square-600.0': [],     # (1) blue
#             'rect-458.0-610.0': ['white', 'other'],  # (1) chevron (also multi-color), (2) white
#             'rect-762.0-915.0': [],  # (1) white
#             'rect-915.0-1220.0': [],    # (1) white
#             'pentagon-915.0': [],   # (1) yellow
#             'octagon-915.0': [],    # (1) red
#             'other': [],
#         }
#         self.class_list = list(self.color_dict.keys())

#         self.class_idx = {
#             'circle-750.0': 0,   # (1) white+red, (2) blue+white
#             'triangle-900.0': 3,  # (1) white, (2) yellow
#             'triangle_inverted-1220.0': 5,   # (1) white+red
#             'diamond-600.0': 6,    # (1) white+yellow
#             'diamond-915.0': 7,    # (1) yellow
#             'square-600.0': 8,     # (1) blue
#             'rect-458.0-610.0': 9,  # (1) chevron (also multi-color), (2) white
#             'rect-762.0-915.0': 11,  # (1) white
#             'rect-915.0-1220.0': 12,    # (1) white
#             'pentagon-915.0': 13,   # (1) yellow
#             'octagon-915.0': 14,    # (1) red
#             'other': 15,
#         }

#         # Define HSV range of the desired colors (H, S, L)
#         WHITE = [[0, 0, 95], [360, 360, 100]]


#     def forward(self, x):
#         logits = self.model(x)
#         y = logits.argmax(-1)

#         # Change image to HSL color space

#         # Count pixels that satisfy the color range
