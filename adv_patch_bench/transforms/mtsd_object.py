"""Implement REAP patch rendering for MTSD data."""

from __future__ import annotations

from adv_patch_bench.transforms import reap_object


class MtsdObject(reap_object.ReapObject):
    """Object wrapper for MTSD samples."""

    def __init__(self, **kwargs) -> None:
        """Initialize MtsdObject."""
        super().__init__(
            dataset="mtsd_no_color",
            pad_to_square=True,
            use_box_mode=True,
            **kwargs,
        )
