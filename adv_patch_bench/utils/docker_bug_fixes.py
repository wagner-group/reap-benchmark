"""Script to ensure backward compatibility with Nvidia PyTorch docker."""

import os

AM_I_IN_A_DOCKER_CONTAINER = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False)

# Calling subprocess.check_output() with python version 3.8.10 or lower will
# raise NotADirectoryError. When torch calls this to call hipconfig, it does
# not catch this exception but only FileNotFoundError or PermissionError.
# This hack makes sure that correct exception is raised.
if AM_I_IN_A_DOCKER_CONTAINER:
    import subprocess

    def _hacky_subprocess_fix(*args, **kwargs):
        _ = args, kwargs
        raise FileNotFoundError(
            "Hacky exception. If this interferes with your workflow, consider "
            "using python >= 3.8.10 or simply try to comment this out."
        )

    subprocess.check_output = _hacky_subprocess_fix

    # This floating point (core dumped) error is caused by the function
    # `erfinv_(x)` when initializing SwinTransformer If you don't face this
    # problem, you can safely comment this out.
    import torch
    from scipy import special

    def _hacky_erfinv(self):
        # pylint: disable=no-member
        erfinv = torch.from_numpy(special.erfinv(self.numpy()))
        self.zero_()
        self.add_(erfinv)

    torch.Tensor.erfinv_ = _hacky_erfinv
