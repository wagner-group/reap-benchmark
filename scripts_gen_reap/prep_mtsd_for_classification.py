"""Create classification dataset from MTSD."""

import json
import os
import random
from os.path import join

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def _load_annotation(label_path, image_key):
    with open(
        join(label_path, "{:s}.json".format(image_key)), "r", encoding="utf-8"
    ) as fid:
        anno = json.load(fid)
    return anno


def main(data_dir, new_label_dict, dataset_name, pad=0.0):
    """Create classification dataset from MTSD."""
    bg_idx = max(list(new_label_dict.values())) + 1
    images, labels, names = [], [], []

    for split in ["train", "val"]:

        print(f"Running on {split} split...")
        img_path = join(data_dir, split)
        label_path = join(data_dir, "annotations")

        image_keys, exts = [], []
        for entry in os.scandir(img_path):
            if (
                entry.path.endswith(".jpg") or entry.path.endswith(".png")
            ) and entry.is_file():
                image_keys.append(entry.name.split(".")[0])
                exts.append(entry.name.split(".")[1])
        print(f"Found {len(image_keys)} images.")

        for image_key, ext in tqdm(zip(image_keys, exts)):
            anno = _load_annotation(label_path, image_key)

            with Image.open(
                os.path.join(img_path, f"{image_key}.{ext}")
            ) as img:
                img = img.convert("RGB")

            for index, obj in enumerate(anno["objects"]):
                if obj["properties"]["ambiguous"]:
                    continue
                class_name = obj["label"]
                shape_index = new_label_dict.get(class_name, bg_idx)
                x1 = obj["bbox"]["xmin"]
                y1 = obj["bbox"]["ymin"]
                x2 = obj["bbox"]["xmax"]
                y2 = obj["bbox"]["ymax"]

                # Square crop traffic sign with padding
                box_length = (1 + pad) * max((x2 - x1, y2 - y1))
                width_change = box_length - (x2 - x1)
                height_change = box_length - (y2 - y1)
                x1 = x1 - width_change / 2
                x2 = x2 + width_change / 2
                y1 = y1 - height_change / 2
                y2 = y2 + height_change / 2
                img_cropped = img.crop((x1, y1, x2, y2))
                images.append(
                    img_cropped.resize(
                        (CROPPED_SIZE, CROPPED_SIZE),
                        resample=Image.Resampling.BICUBIC,
                    )
                )
                labels.append(shape_index)
                names.append(f"{image_key}_{index}.png")

            # DEBUG
            # if len(images) > 100:
            #     break

    # label_counts is a tuple of (labels, counts)
    label_counts = np.unique(labels, return_counts=True)
    print("Label distribution: ", label_counts)

    # Sort labels by count and keep only top MAX_NUM_CLASSES
    sorted_idx = np.argsort(label_counts[1])[::-1]
    kept_classes = label_counts[0][sorted_idx][:MAX_NUM_CLASSES]
    # Set all labels not in kept_classes to background
    for i, label in enumerate(labels):
        if label not in kept_classes:
            labels[i] = bg_idx
    print(
        f"Label distribution after keeping only top {MAX_NUM_CLASSES} classes:",
        np.unique(labels, return_counts=True),
    )

    # Train and val split
    num_samples = len(images)
    num_train = int(0.9 * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    indices = {"train": indices[:num_train], "val": indices[num_train:]}

    for split in ["train", "val"]:
        save_dir = join(data_dir, dataset_name, split)
        for i in range(bg_idx + 1):
            os.makedirs(join(save_dir, f"{i:03d}"), exist_ok=True)
        for i in tqdm(indices[split]):
            images[i].save(join(save_dir, f"{labels[i]:03d}", names[i]))


if __name__ == "__main__":
    # Set data paths
    DATA_DIR = "~/data/mtsd_v2_fully_annotated/"
    CSV_PATH = "~/data/mtsd_v2_fully_annotated/traffic_sign_dimension_v6.csv"
    DATA_DIR = os.path.expanduser(DATA_DIR)
    CSV_PATH = os.path.expanduser(CSV_PATH)

    # Set parameters for dataset
    SEED = 0
    CROPPED_SIZE = 224
    PAD_SIZE = 0.0
    MAX_NUM_CLASSES = 100
    DATASET_NAME = f"cropped_signs_mtsd-{MAX_NUM_CLASSES}"

    np.random.seed(SEED)
    random.seed(SEED)
    data = pd.read_csv(CSV_PATH)

    print(np.unique(list(data["target"])))

    if DATASET_NAME == "cropped_signs_no_colors":
        # Shape classification only
        selected_labels = [
            "circle-750.0",
            "triangle-900.0",
            "triangle_inverted-1220.0",
            "diamond-600.0",
            "diamond-915.0",
            "square-600.0",
            "rect-458.0-610.0",
            "rect-762.0-915.0",
            "rect-915.0-1220.0",
            "pentagon-915.0",
            "octagon-915.0",
        ]
        NEW_LABEL_DICT = {}
        for _, row in data.iterrows():
            if row["target"] in selected_labels:
                NEW_LABEL_DICT[row["sign"]] = selected_labels.index(
                    row["target"]
                )

    elif DATASET_NAME == "cropped_signs_with_colors":
        # Shape and some color classification
        # There is one yellow circle. It is set to white.
        color_dict = {
            "circle-750.0": [
                "white",
                "blue",
                "red",
            ],  # (1) white+red, (2) blue+white
            "triangle-900.0": ["white", "yellow"],  # (1) white, (2) yellow
            "triangle_inverted-1220.0": [],  # (1) white+red
            "diamond-600.0": [],  # (1) white+yellow
            "diamond-915.0": [],  # (1) yellow
            "square-600.0": [],  # (1) blue
            "rect-458.0-610.0": [
                "white",
                "other",
            ],  # (1) chevron (also multi-color), (2) white
            "rect-762.0-915.0": [],  # (1) white
            "rect-915.0-1220.0": [],  # (1) white
            "pentagon-915.0": [],  # (1) yellow
            "octagon-915.0": [],  # (1) red
        }
        class_idx = {
            "circle-750.0": 0,  # (1) white+red, (2) blue+white
            "triangle-900.0": 3,  # (1) white, (2) yellow
            "triangle_inverted-1220.0": 5,  # (1) white+red
            "diamond-600.0": 6,  # (1) white+yellow
            "diamond-915.0": 7,  # (1) yellow
            "square-600.0": 8,  # (1) blue
            "rect-458.0-610.0": 9,  # (1) chevron (also multi-color), (2) white
            "rect-762.0-915.0": 11,  # (1) white
            "rect-915.0-1220.0": 12,  # (1) white
            "pentagon-915.0": 13,  # (1) yellow
            "octagon-915.0": 14,  # (1) red
        }
        selected_labels = list(class_idx.keys())
        NEW_LABEL_DICT = {}
        for _, row in data.iterrows():
            if row["target"] in class_idx:
                idx = class_idx[row["target"]]
                color_list = color_dict[row["target"]]
                # print(row['sign'], row['target'])
                if len(color_list) > 0:
                    idx += color_list.index(row["color"])
                NEW_LABEL_DICT[row["sign"]] = idx

    elif "cropped_signs_mtsd" in DATASET_NAME:
        # Use MTSD labels that fall into the set of known shapes/sizes
        selected_shapes = {
            "circle-750.0",
            "triangle-900.0",
            "triangle_inverted-1220.0",
            "diamond-600.0",
            "diamond-915.0",
            "square-600.0",
            "rect-458.0-610.0",
            "rect-762.0-915.0",
            "rect-915.0-1220.0",
            "pentagon-915.0",
            "octagon-915.0",
        }
        NEW_LABEL_DICT = {}
        counter = 0
        for _, row in data.iterrows():
            if row["target"] not in selected_shapes:
                continue
            NEW_LABEL_DICT[row["sign"]] = counter
            counter += 1
    else:
        raise NotImplementedError("Dataset name not implemented.")

    print("Number of classes:", len(NEW_LABEL_DICT))
    main(DATA_DIR, NEW_LABEL_DICT, DATASET_NAME, pad=PAD_SIZE)
