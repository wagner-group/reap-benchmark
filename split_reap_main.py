import os

import pandas as pd
import numpy as np

from hparams import DATASET_METADATA, MTSD100_TO_SHAPE


def main(dataset: str):
    anno = pd.read_csv("reap_annotations.csv")

    # Class to filenames
    label_to_files = {}
    class_names = DATASET_METADATA[dataset]["class_name"].values()
    label_col = "100_label" if dataset == "reap-100" else "final_shape"
    
    for _, obj_df in anno.iterrows():
        filename = obj_df["filename"]
        label = obj_df[label_col]
        assert label in class_names
        if label not in label_to_files:
            label_to_files[label] = []
        label_to_files[label].append(filename)

    num_files = [len(files) for files in label_to_files.values()]
    # print([MTSD100_TO_SHAPE[l] for l, files in label_to_files.items() if len(files) >= 9])
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    # DATASET = "reap-100"
    DATASET = "reap"
    main(DATASET)