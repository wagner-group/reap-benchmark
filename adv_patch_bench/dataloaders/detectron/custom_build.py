"""Custom build_detection_test_loader for Detectron2 models."""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch.utils.data as torchdata
from detectron2.config import configurable
from detectron2.data import build
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler

logger = logging.getLogger(__name__)


# pylint: disable=protected-access
@configurable(from_config=build._test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 8,
    pin_memory: bool = True,
    split_file_names: Optional[Set[str]] = None,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    """Custom test dataloader to incorporate split_file_names.

    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all
    workers to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts, or a pytorch dataset (either map-style
            or iterable). They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset and returns
            the format to be consumed by the model. When using cfg, the default
            choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces indices to be applied on ``dataset``.
            Default to :class:`InferenceSampler`, which splits the dataset
            across all workers. Sampler must be None if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created. Default to
            1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection dataset,
        with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    # Filter data by filenames
    if split_file_names:
        new_dataset = [
            d
            for d in dataset
            if d["file_name"].split("/")[-1] in split_file_names
        ]
        if len(new_dataset) != len(split_file_names):
            logger.warning(
                "Not all files listed in filter_file_names are found "
                "(dataset length: %d vs split files: %d)!",
                len(new_dataset),
                len(split_file_names),
            )
        dataset = new_dataset

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert (
            sampler is None
        ), "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=build.trivial_batch_collator
        if collate_fn is None
        else collate_fn,
    )
