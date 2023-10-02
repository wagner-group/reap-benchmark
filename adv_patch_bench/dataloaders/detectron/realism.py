"""Register and load Relism Test dataset."""

from __future__ import annotations

import logging
import pathlib

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from tqdm.auto import tqdm

from adv_patch_bench.utils.tqdm_logger import TqdmLoggingHandler
from adv_patch_bench.utils.types import DetectronSample
from hparams import Metadata

_NUM_KEYPOINTS = 4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())


def get_data_dict(modifier: str) -> list[DetectronSample]:
    """Get Realism dataset as list of samples in Detectron2 format.

    Args:
        modifier: Dataset modifier.

    Returns:
        List of Realism samples in Detectron2 format.
    """
    metadata = Metadata.get("realism")
    data_path = pathlib.Path(metadata.data_path) / modifier
    dataset_dicts: list[DetectronSample] = []

    for idx, img_path in enumerate(
        tqdm(data_path.glob("images/*.jpg"), mininterval=10)
    ):

        label_name = img_path.stem + ".txt"
        label_path: pathlib.Path = data_path / "labels" / label_name

        # Read label file
        with label_path.open("r", encoding="utf-8") as file:
            anno = file.readlines()[0].strip().split(",")

        obj_class, _, _, _, _, width, height, _ = anno
        record: DetectronSample = {
            "file_name": str(img_path),
            "image_id": idx,
            "width": int(width),
            "height": int(height),
        }

        # Populate record or sample with its objects
        record["annotations"] = [
            {
                "bbox": [float(xy) for xy in anno[1:5]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(obj_class),
                "has_reap": True,
                "keypoints": [0] * _NUM_KEYPOINTS * 3,
                "ct_coeffs": None,
                "poly_coeffs": None,
            }
        ]
        dataset_dicts.append(record)

    return dataset_dicts


def register_realism(dataset_name: str = "realism") -> None:
    """Register Realism dataset on Detectron2.

    Args:
        dataset_name: Dataset name along with modifiers. Defaults to "realism".
    """
    metadata = Metadata.get("realism")
    bg_class: int = len(metadata.class_names) - 1
    thing_classes = metadata.class_names
    if Metadata.parse_dataset_name(dataset_name).ignore_bg_class:
        thing_classes = thing_classes[:-1]

    d2_metadata = {
        "thing_classes": thing_classes,
        "keypoint_names": [f"p{i}" for i in range(_NUM_KEYPOINTS)],
        "keypoint_flip_map": [
            (f"p{i}", f"p{i}") for i in range(_NUM_KEYPOINTS)
        ],
        "obj_dim_dict": metadata,  # TODO(check)
        "bg_class": bg_class,
    }
    # Register dataset without split to keep metadata
    base_dataset, modifier = dataset_name.split("-")
    DatasetCatalog.register(base_dataset, lambda x: [])
    MetadataCatalog.get(base_dataset).set(**d2_metadata)
    DatasetCatalog.register(dataset_name, lambda m=modifier: get_data_dict(m))
    MetadataCatalog.get(dataset_name).set(**d2_metadata)
