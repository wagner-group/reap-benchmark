from __future__ import annotations
import os

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

IMAGE_PATH = "/data/data/reap-benchmark/reap_realism_test/images_jpg/"
ANNO_PATH = "./realism_test_anno.csv"
SAVE_PATH = "/data/data/reap-benchmark/reap_realism_test/test/"
IMAGE_PATH = Path(IMAGE_PATH).expanduser()
ANNO_PATH = Path(ANNO_PATH).expanduser()
SAVE_PATH = Path(SAVE_PATH).expanduser()


def main():
    wpatch_path = SAVE_PATH / "with_patch"
    wopatch_path = SAVE_PATH / "without_patch"
    image_files = [str(p) for p in IMAGE_PATH.glob("*.png")]
    image_files.sort()
    # filenames = [p.split(".", maxsplit=1)[0] for p in image_files]
    wpatch_path.mkdir(parents=True, exist_ok=True)
    wopatch_path.mkdir(parents=True, exist_ok=True)

    annotation_df = pd.read_csv("realism_test_anno.csv")
    import pdb
    pdb.set_trace()
    for (index, row), image_file in tqdm(zip(annotation_df.iterrows(), image_files)):
        print(index)
        name = image_file.split(".", maxsplit=1)[0]
        is_with_patch = index % 2 == 1
        (
            sign_x1,
            sign_y1,
            sign_x2,
            sign_y2,
            sign_x3,
            sign_y3,
            sign_x4,
            sign_y4,
        ) = (
            float(row["sign_x1"]),
            float(row["sign_y1"]),
            float(row["sign_x2"]),
            float(row["sign_y2"]),
            float(row["sign_x3"]),
            float(row["sign_y3"]),
            float(row["sign_x4"]),
            float(row["sign_y4"]),
        )
        xmin = min(sign_x1, sign_x2, sign_x3, sign_x4)
        xmax = max(sign_x1, sign_x2, sign_x3, sign_x4)
        ymin = min(sign_y1, sign_y2, sign_y3, sign_y4)
        ymax = max(sign_y1, sign_y2, sign_y3, sign_y4)
        class_label = int((index % 22) // 2)
        # Annotation to save to text file
        line = f"{class_label},{xmin},{ymin},{xmax},{ymax},6036,4020,0"

        base_path = wpatch_path if is_with_patch else wopatch_path
        label_file = str(base_path / "labels" / f"{name}.txt")
        image_link_file = str(base_path / "images" / f"{name}.png")
        # Save annotation to text file
        with open(label_file, 'w', encoding="utf-8") as file:
            file.write(line)
        os.symlink(str(IMAGE_PATH / image_file), image_link_file)


if __name__ == "__main__":
    main()
