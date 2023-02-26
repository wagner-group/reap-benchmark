"""Split evaluation dataset into attack and test sets.

Attack set is used to generate adversarial patch, and the teset set is used to
evaluate the patch as well as the original image.

The generated splits will be saved as .txt files at
./splits/<DATASET>/bg_<NUM_ATTACK_IMGS>/ under the names <CLASS_NAME>_attack.txt
and <CLASS_NAME>_test.txt for attack and test splits respectively.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

from hparams import DATASET_METADATA, MTSD100_TO_SHAPE


def main():
    """Main function."""
    anno = pd.read_csv("reap_annotations.csv")

    # Class to filenames
    label_to_files = {}
    class_names = DATASET_METADATA[DATASET]["class_name"].values()
    label_col = "100_label" if DATASET == "reap-100" else "final_shape"

    for _, obj_df in anno.iterrows():
        filename = obj_df["filename"]
        label = obj_df[label_col]
        assert label in class_names
        if label not in label_to_files:
            label_to_files[label] = set()
        # Same file does not appear twice in the same class
        label_to_files[label].add(filename)

    num_files = [len(files) for files in label_to_files.values()]
    print("Sorted number of images per class:", sorted(num_files, reverse=True))

    label_to_files = {
        label: files
        for label, files in label_to_files.items()
        if len(files) >= MIN_NUM_IMGS
    }
    if "-100" in DATASET:
        print(
            "Shapes of remaning classes:",
            [MTSD100_TO_SHAPE[label] for label in label_to_files.keys()],
        )
    print(
        f"Number of classes after filtering out ones with fewer than "
        f"{MIN_NUM_IMGS} images: {len(label_to_files)}"
    )

    for label, files in label_to_files.items():
        print(f"Processing class {label}...")
        if NUM_ATTACK_IMGS == "auto":
            # TODO(feature): Not sure what this should be yet
            num_attack_imgs = len(files) // 2
        else:
            num_attack_imgs = NUM_ATTACK_IMGS
        attack_imgs = random.sample(list(files), num_attack_imgs)
        assert len(attack_imgs) == len(set(attack_imgs)), "Duplicate images!"
        test_imgs = list(set(files) - set(attack_imgs))
        assert len(attack_imgs) + len(test_imgs) == len(files)

        # Save splits
        with open(
            os.path.join(SPLIT_PATH, f"{label}_attack.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            for img in attack_imgs:
                file.write(f"{img}\n")
        with open(
            os.path.join(SPLIT_PATH, f"{label}_test.txt"), "w", encoding="utf-8"
        ) as file:
            for img in test_imgs:
                file.write(f"{img}\n")

    print("Finished!")


if __name__ == "__main__":
    DATASET = "reap-100"
    # DATASET = "reap"
    # Number of images to allocate for attack split ("auto" or int). We use 50
    # for normal reap and 5 for reap-100.
    NUM_ATTACK_IMGS = 5
    # Minimum number of images per class to include in the dataset
    MIN_NUM_IMGS = 10
    SPLIT_PATH = os.path.expanduser(f"./splits/{DATASET}/bg_{NUM_ATTACK_IMGS}/")
    if not os.path.exists(SPLIT_PATH):
        os.makedirs(SPLIT_PATH)

    SEED = 0  # Random seed
    np.random.seed(SEED)
    random.seed(SEED)

    main()
