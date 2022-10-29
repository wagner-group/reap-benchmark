"""Load data for classification task."""

from adv_patch_bench.dataloaders.mtsd_mapillary import MTSD_MAPILLARY

DATASET_DICT = {
    "mtsd": MTSD_MAPILLARY,
    # TODO: Clean up. This is not a bug. Only used to get normalization.
    "mapillary_no_color": MTSD_MAPILLARY,
    "mapillary_color": MTSD_MAPILLARY,
}


def load_dataset(args):
    """Load classification dataset."""
    loader = DATASET_DICT[args.dataset]["loader"]
    return loader(args)
