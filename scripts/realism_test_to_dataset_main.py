"""This small script turns the realism test samples into dataset."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import torch
from tqdm.auto import tqdm

from adv_patch_bench.transforms import util
from adv_patch_bench.utils.realism import compute_relight_params
from hparams import DATASET_METADATA
import adv_patch_bench.utils.image as img_util

IMAGE_PATH = "~/data/reap-benchmark/reap_realism_test/images_jpg/"
ANNO_PATH = "./realism_test_anno.csv"
SAVE_PATH = "~/data/reap-benchmark/reap_realism_test/"
PATCH_PATH = "~/data/reap-benchmark/reap_realism_test/synthetic-load-64-15-1-0.4-0-0-pd64-bg50-augimg1-rp2_1e-05_0_1_1000_adam_0.01_False"
IMAGE_PATH = Path(IMAGE_PATH).expanduser()
ANNO_PATH = Path(ANNO_PATH).expanduser()
SAVE_PATH = Path(SAVE_PATH).expanduser()
PATCH_PATH = Path(PATCH_PATH).expanduser()


def main(relight_method, relight_params):
    """Main function."""
    wpatch_path = SAVE_PATH / "real"
    wopatch_path = SAVE_PATH / "percentile0.2"
    image_files = [str(p) for p in IMAGE_PATH.glob("*.jpg")]
    image_files.sort()
    # filenames = [p.split(".", maxsplit=1)[0] for p in image_files]
    (wpatch_path / "labels").mkdir(parents=True, exist_ok=True)
    (wpatch_path / "images").mkdir(parents=True, exist_ok=True)
    (wopatch_path / "labels").mkdir(parents=True, exist_ok=True)
    (wopatch_path / "images").mkdir(parents=True, exist_ok=True)

    annotation_df = pd.read_csv("realism_test_anno.csv")
    for (index, row), image_file in tqdm(
        zip(annotation_df.iterrows(), image_files)
    ):
        name = Path(image_file).stem
        obj_class = int((index % 22) // 2)
        is_with_patch = index % 2 == 1
        sign_keypoints = [
            row["sign_x1"],
            row["sign_y1"],
            row["sign_x2"],
            row["sign_y2"],
            row["sign_x3"],
            row["sign_y3"],
            row["sign_x4"],
            row["sign_y4"],
        ]
        sign_keypoints = [float(k) for k in sign_keypoints]
        sign_keypoints = [k * 2048 / 6036 for k in sign_keypoints]

        # image = skimage.io.imread(image_file)
        # torch_image = torch.from_numpy(image).float().permute(2, 0, 1)
        # torch_image.unsqueeze_(0)
        # torch_image /= 255.0

        # # get image dimensions
        # # img_height, img_width = torch_image.shape[-2:]

        # # read in patch and mask from file
        # obj_name = DATASET_METADATA["mapillary-no_color"]["class_name"][obj_class]
        # with open(str(PATCH_PATH / obj_name / "adv_patch.pkl"), "rb") as file:
        #     patch, _ = pickle.load(file)
        # patch_size_in_pixel = patch.shape[-1]

        # hw_ratio_dict = DATASET_METADATA["mapillary-no_color"]["hw_ratio"]
        # # get aspect ratio for current object class
        # hw_ratio = hw_ratio_dict[obj_class]
        # obj_shape = DATASET_METADATA["mapillary-no_color"]["shape"][obj_class]

        # # Generate mask for object in image
        # sign_mask, src = util.gen_sign_mask(
        #     obj_shape,
        #     hw_ratio=hw_ratio,
        #     obj_width_px=round(patch_size_in_pixel * hw_ratio)
        #     if "rect" in obj_shape
        #     else patch_size_in_pixel,
        #     pad_to_square=False,
        # )
        # src = np.array(src).astype(np.float32)
        # tgt_list = [[sign_x1, sign_y1], [sign_x2, sign_y2], [sign_x3, sign_y3]]
        # if len(src) == 4:
        #     tgt_list.append([sign_x4, sign_y4])
        # tgt = np.array(tgt_list).astype(np.float32)
        # tgt *= 1024 / 6036

        # relight_coeffs, _ = compute_relight_params(
        #     torch_image,
        #     sign_mask,
        #     relight_method,
        #     relight_params,
        #     obj_name,
        #     src,
        #     tgt,
        # )

        xmin = min(sign_keypoints[::2])
        xmax = max(sign_keypoints[::2])
        ymin = min(sign_keypoints[1::2])
        ymax = max(sign_keypoints[1::2])

        # Scale to new image size
        _, scales, padding = img_util.resize_and_pad(
            orig_size=(1364, 2048),
            resize_size=(1536, 2048),
            pad_size=(1536, 2048),
            keep_aspect_ratio=True,
            return_params=True,
        )
        ymin = ymin * scales[0] + padding[1]
        ymax = ymax * scales[0] + padding[1]
        xmin = xmin * scales[1] + padding[0]
        xmax = xmax * scales[1] + padding[0]

        # Annotation to save to text file
        line = f"{obj_class},{xmin},{ymin},{xmax},{ymax},2048,1536,0\n"

        base_path = wpatch_path if is_with_patch else wopatch_path
        # label_file = str(base_path / "labels" / f"{name}.txt")
        label_file = str(base_path / "labels" / f"{index + int(not is_with_patch):03d}.txt")
        # image_link_file = str(base_path / "images" / f"{name}.png")
        # Save annotation to text file
        with open(label_file, "w", encoding="utf-8") as file:
            file.write(line)
        # if not Path(image_link_file).exists():
        #     os.symlink(str(IMAGE_PATH / image_file), image_link_file)


if __name__ == "__main__":
    relight_method = "percentile"
    percentile = 0.2
    relight_params = {"percentile": percentile}
    # results[f"{RELIGHT_METHOD}_{percentile}"] = main(
    #     GEO_METHOD, RELIGHT_METHOD, params, use_jpeg=True
    # )

    # with open("tmp/realism_test_results.pkl", "wb") as f:
    #     pickle.dump(results, f)
    main(relight_method, relight_params)
