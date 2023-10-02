"""Script for preparing MTSD dataset for YOLO training."""

import json
import os
import shutil
from os.path import expanduser, join
from typing import Literal

import pandas as pd
from tqdm import tqdm

from hparams import (
    PATH_DUPLICATE_FILES,
    TS_COLOR_DICT,
    TS_COLOR_OFFSET_DICT,
    Metadata,
)

MIN_OBJ_AREA = 100
PATH_APB_ANNO = "./mtsd_label_metadata.csv"
PATH_MTSD_BASE = "~/data/mtsd_v2_fully_annotated/"


def readlines(path):
    """Read lines from a file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def main():
    """Main function."""
    mode: Literal["orig", "100", "shape"] = "shape"  # "100", "shape", "orig"
    use_mtsd_original_labels = mode == "orig"
    # # DEPRECATED: This is default options; Don't change!
    ignore_other = False
    use_color = False

    if mode == "100":
        # MTSD-100 dataset (100 most common classes)
        dataset = "mtsd-100"
        label_path = "labels_100"
    elif mode == "orig":
        # Original MTSD labels
        dataset = "mtsd-orig"
        label_path = "labels_orig"
    else:
        # Default shape-based MTSD dataset
        dataset = "mtsd"
        label_path = "labels_no_color"

    path = os.path.expanduser(PATH_MTSD_BASE)
    csv_path = os.path.expanduser(PATH_APB_ANNO)
    similarity_df_csv_path = PATH_DUPLICATE_FILES
    anno_path = join(path, "annotations")

    bg_idx = None
    for i, class_name in Metadata.get(dataset).class_names.items():
        if class_name == "other":
            bg_idx = i
            break
    assert bg_idx is not None, "other class is not found!"

    label_path = join(path, label_path)
    data = pd.read_csv(csv_path)
    similar_files_df = pd.read_csv(similarity_df_csv_path)

    mtsd_label_to_class_index = {}
    if use_color:
        selected_labels = list(TS_COLOR_OFFSET_DICT.keys())
        for idx, row in data.iterrows():
            if (
                row["target"] in TS_COLOR_OFFSET_DICT
                and not use_mtsd_original_labels
            ):
                idx = TS_COLOR_OFFSET_DICT[row["target"]]
                color_list = TS_COLOR_DICT[row["target"]]
                if len(color_list) > 0:
                    idx += color_list.index(row["color"])
                mtsd_label_to_class_index[row["sign"]] = idx
            elif use_mtsd_original_labels:
                mtsd_label_to_class_index[row["sign"]] = idx
    else:
        selected_labels = list(TS_COLOR_DICT.keys())
        for idx, row in data.iterrows():
            if row["target"] in TS_COLOR_DICT and not use_mtsd_original_labels:
                idx = selected_labels.index(row["target"])
                mtsd_label_to_class_index[row["sign"]] = idx
            elif use_mtsd_original_labels:
                mtsd_label_to_class_index[row["sign"]] = idx

    # Save filenames and the data partition they belong to
    splits = ["train", "test", "val"]
    split_dict = {}
    for split in splits:
        os.makedirs(join(label_path, split), exist_ok=True)
        filenames = readlines(expanduser(join(path, "splits", split + ".txt")))
        for name in filenames:
            split_dict[name] = split

    # Get all JSON files
    json_files = [
        join(anno_path, f)
        for f in os.listdir(anno_path)
        if os.path.isfile(join(anno_path, f)) and f.endswith(".json")
    ]
    print(f"Found {len(json_files)} files")

    similar_files_count = 0
    num_too_small = 0
    num_other = 0
    num_cross_boundary = 0

    # plotting parameters
    # num_images_per_row = 5
    # num_images_per_col = 10
    # num_images_plotted = 0

    for json_file in tqdm(json_files):
        filename = json_file.split(".")[-2].split("/")[-1]

        jpg_filename = f"{filename}.jpg"
        if jpg_filename in similar_files_df["filename"].values:
            similar_files_count += 1
            continue

        split = split_dict[filename]

        # Read JSON files
        with open(json_file, "r", encoding="utf-8") as file:
            anno = json.load(file)

        text = ""
        width, height = anno["width"], anno["height"]
        for obj in anno["objects"]:
            x_center = (obj["bbox"]["xmin"] + obj["bbox"]["xmax"]) / 2 / width
            y_center = (obj["bbox"]["ymin"] + obj["bbox"]["ymax"]) / 2 / height
            obj_width = (obj["bbox"]["xmax"] - obj["bbox"]["xmin"]) / width
            obj_height = (obj["bbox"]["ymax"] - obj["bbox"]["ymin"]) / height

            class_index = mtsd_label_to_class_index.get(obj["label"], bg_idx)
            # Compute object area if the image were to be resized to have width
            # of 1280 pixels
            obj_area = (obj_width * 1280) * (obj_height * height / width * 1280)
            # Remove labels for small or "other" objects
            if "cross_boundary" in obj["bbox"]:
                num_cross_boundary += 1
                continue
            if obj_area < MIN_OBJ_AREA:
                num_too_small += 1
                continue
            if ignore_other and class_index == bg_idx:
                num_other += 1
                continue
            text += f"{class_index:d} {x_center} {y_center} {obj_width} {obj_height} 0\n"

        if text:
            with open(
                join(label_path, split, filename + ".txt"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(text)

    print(
        f"There are {similar_files_count} similar files in Mapillary and MTSD"
    )
    print("These duplicates will be removed from MTSD")
    print("The following objects are being excluded from the label files:")
    print(f"{num_cross_boundary} signs cross boundary.")
    print(
        f"{num_too_small} of the remaining ones are too small (< {MIN_OBJ_AREA}"
        " pixels)."
    )
    print(f'{num_other} of the remaining ones are in "other" class.')

    # Moving duplicated files to a separate directory
    data_path = join(path, "images/")
    new_data_path = join(path, "images_mtsd_duplicates/")
    for split in splits:
        os.makedirs(os.path.join(new_data_path, split), exist_ok=True)

    for json_file in tqdm(json_files):
        filename = json_file.split(".")[-2].split("/")[-1]
        jpg_filename = f"{filename}.jpg"
        split = split_dict[filename]
        if jpg_filename not in similar_files_df["filename"].values:
            continue
        image_path = os.path.join(data_path, split, jpg_filename)
        image_new_path = os.path.join(new_data_path, split, jpg_filename)
        if os.path.isfile(image_path):
            shutil.move(image_path, image_new_path)

    print("Finished.")


if __name__ == "__main__":
    main()
