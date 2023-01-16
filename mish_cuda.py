"""Dummy mish_cuda to silence YOLOv7 warning."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


def mish(inputs):
    """Mish activation function."""
    return inputs.mul(F.softplus(inputs).tanh())


class MishCuda(nn.Module):
    """Dummy mish_cuda."""

    def forward(self, inputs):
        """Forward pass."""
        return mish(inputs)
