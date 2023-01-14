"""Run realism test for REAP benchmark."""

import math
import os
import pathlib
import pickle

import cv2 as cv
import kornia.geometry.transform as kornia_tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import torch
from kornia.geometry.transform import get_perspective_transform
from torchvision.utils import save_image
from tqdm import tqdm

from adv_patch_bench.transforms import lighting_tf, util
from hparams import OBJ_DIM_DICT, TS_COLOR_DICT

# list of point colors for visualizing image points
POINT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def _save_images(
    image,
    src,
    tgt,
    patch_src,
    patch_src_transformed,
    patch_tgt,
    img_width,
    img_height,
    is_clean,
):
    if patch_tgt is None:
        patch_tgt = [None for _ in patch_src]

    for i, (src_i, tgt_i) in enumerate(zip(src, tgt)):
        cv.circle(image, (int(src_i[0]), int(src_i[1])), 5, POINT_COLORS[i], -1)
        cv.circle(image, (int(tgt_i[0]), int(tgt_i[1])), 5, POINT_COLORS[i], -1)

    for i, (src_i, srct_i, tgt_i) in enumerate(
        zip(patch_src, patch_src_transformed, patch_tgt)
    ):
        cv.circle(image, (int(src_i[0]), int(src_i[1])), 5, POINT_COLORS[i], -1)
        cv.circle(
            image, (int(srct_i[0]), int(srct_i[1])), 5, POINT_COLORS[i], -1
        )
        if not is_clean:
            cv.circle(
                image, (int(tgt_i[0]), int(tgt_i[1])), 5, POINT_COLORS[i], -1
            )
    # resize image and scale down by 8
    image_resized = cv.resize(image, (img_width // 8, img_height // 8))
    image_resized = torch.from_numpy(image_resized).permute(2, 0, 1)
    return image_resized.float() / 255


def main():
    """Main function for running realism test."""
    # file directory where images are stored
    file_dir = "~/data/reap-benchmark/reap_realism_test/images/"
    file_dir = os.path.expanduser(file_dir)

    # path to directory where patch files are stored
    patch_path = pathlib.Path(
        "~/data/reap-benchmark/reap_realism_test/"
        "synthetic-load-64-15-1-0.4-0-0-pd64-bg50-augimg1-rp2_1e-05_0_1_1000_adam_0.01_False"
    )
    patch_path = patch_path.expanduser()

    # read in annotation data from csv file
    annotation_df = pd.read_csv("realism_test_anno.csv")

    obj_class_to_shape = {
        "circle": "circle",
        "up-triangle": "triangle_inverted",
        "triangle": "triangle",
        "rect-s": "rect",
        "rect-m": "rect",
        "rect-l": "rect",
        "diamond-s": "diamond",
        "diamond-l": "diamond",
        "pentagon": "pentagon",
        "octagon": "octagon",
        "square": "square",
    }

    # list of traffic sign classes
    traffic_sign_classes = list(TS_COLOR_DICT.keys())
    # remove 'other' class from list
    traffic_sign_classes.remove("other")

    # lists to store geometric and lighting errors for each image
    geometric_errors = []
    lighting_errors = []

    for index, row in tqdm(annotation_df.iterrows()):
        is_clean = index % 2 == 0
        if is_clean:
            alpha = None
            beta = None
        else:
            assert (
                alpha is not None
            ), "alpha must be specified for adversarial images"
            assert (
                beta is not None
            ), "beta must be specified for adversarial images"

        # obj_class = traffic_sign_classes[(index // 4) % len(traffic_sign_classes)]
        obj_class = traffic_sign_classes[
            (index // 2) % len(traffic_sign_classes)
        ]

        # get file path for image
        filename = row["file_name"]
        filepath = os.path.join(file_dir, filename)
        # check if file exists
        if not os.path.exists(filepath):
            print("File not found: ", filepath)
            continue

        # read image
        image = skimage.io.imread(filepath)
        torch_image = torch.from_numpy(image).float().permute(2, 0, 1)
        torch_image.unsqueeze_(0)
        torch_image /= 255.0

        # get image dimensions
        img_width, img_height = row["width"], row["height"]

        # get labeled coordinates for object in image
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

        # get labeled coordinates for patch in image
        (
            patch_x1,
            patch_y1,
            patch_x2,
            patch_y2,
            patch_x3,
            patch_y3,
            patch_x4,
            patch_y4,
        ) = (
            float(row["patch_x1"]),
            float(row["patch_y1"]),
            float(row["patch_x2"]),
            float(row["patch_y2"]),
            float(row["patch_x3"]),
            float(row["patch_y3"]),
            float(row["patch_x4"]),
            float(row["patch_y4"]),
        )

        # read in patch and mask from file
        with open(str(patch_path / obj_class / "adv_patch.pkl"), "rb") as file:
            patch, patch_mask = pickle.load(file)

        patch_size_in_pixel = patch.shape[-1]
        hw_ratio_dict = OBJ_DIM_DICT["mapillary_no_color"]["hw_ratio"]
        # get aspect ratio for current object class
        hw_ratio = hw_ratio_dict[(index // 2) % len(traffic_sign_classes)]
        obj_shape = obj_class_to_shape[obj_class]

        # generate mask for object in image
        sign_mask, src = util.gen_sign_mask(
            obj_shape,
            hw_ratio=hw_ratio,
            obj_width_px=patch_size_in_pixel,
            pad_to_square=False,
        )
        src = np.array(src).astype(np.float32)

        # get location of patch in canonical sign
        _, hh, ww = np.where(patch_mask.numpy())
        h_min, h_max = hh.min(), hh.max() + 1
        w_min, w_max = ww.min(), ww.max() + 1
        if obj_class == "diamond-s":
            factor = 0.2
        elif obj_class == "diamond-l":
            factor = 0.15
        # elif obj_class in ("circle", "up-triangle"):
        #     factor = 0.1
        elif obj_class == "circle":
            factor = 0.1
        elif obj_class == "up-triangle":
            factor = 0.1 * 64 / 56
        else:
            factor = 0.0
        shift = math.ceil(h_max * factor)
        h_min -= shift
        h_max -= shift
        patch_src = np.array(
            [[w_min, h_min], [w_max, h_min], [w_max, h_max], [w_min, h_max]]
        ).astype(np.float32)

        # Shift patch and mask
        patch_mask = torch.zeros_like(patch_mask)
        patch_mask[:, h_min:h_max, w_min:w_max] = 1

        # Get target patch loc if exists
        patch_tgt = None
        if not is_clean:
            patch_tgt = np.array(
                [
                    [patch_x1, patch_y1],
                    [patch_x2, patch_y2],
                    [patch_x3, patch_y3],
                    [patch_x4, patch_y4],
                ]
            ).astype(np.float32)

        transform_func = kornia_tf.warp_perspective
        if len(src) == 3:
            tgt = np.array(
                [[sign_x1, sign_y1], [sign_x2, sign_y2], [sign_x3, sign_y3]]
            ).astype(np.float32)

            sign_tf_matrix = (
                torch.from_numpy(cv.getAffineTransform(src, tgt))
                .unsqueeze(0)
                .float()
            )
            # add [0, 0, 1] to M1
            sign_tf_matrix = torch.cat(
                (sign_tf_matrix, torch.tensor([0, 0, 1]).view(1, 1, 3).float()),
                dim=1,
            )
        else:
            tgt = np.array(
                [
                    [sign_x1, sign_y1],
                    [sign_x2, sign_y2],
                    [sign_x3, sign_y3],
                    [sign_x4, sign_y4],
                ]
            ).astype(np.float32)
            src = torch.from_numpy(src).unsqueeze(0)
            tgt = torch.from_numpy(tgt).unsqueeze(0)
            sign_tf_matrix = get_perspective_transform(src, tgt)
            src = src[0]  # unsqueeze(0) above
            tgt = tgt[0]  # unsqueeze(0) above

        # apply perspective transform to src patch coordinates
        patch_src_transformed = cv.perspectiveTransform(
            patch_src.reshape((1, -1, 2)), sign_tf_matrix[0].numpy()
        )[0]

        if SAVE_IMG_DEBUG:
            image_resized = _save_images(
                image,
                src,
                tgt,
                patch_src,
                patch_src_transformed,
                patch_tgt,
                img_width,
                img_height,
                is_clean,
            )
            save_image(image_resized, f"tmp/{index:02d}_test.png")

        if is_clean:
            # warp canonical sign mask (to calculate alpha and beta)
            warped_sign_mask = transform_func(
                sign_mask.float(),
                sign_tf_matrix,
                dsize=(img_height, img_width),
                mode="nearest",
                padding_mode="zeros",
            )
            warped_sign_mask.clamp_(0, 1)
            cropped_traffic_sign = torch.masked_select(
                torch_image, warped_sign_mask.bool()
            )
            # calculate relighting parameters
            alpha, beta = lighting_tf.compute_relight_params(
                cropped_traffic_sign.numpy().reshape(-1, 1)
            )
            print(f"alpha: {alpha:.4f}, beta: {beta:.4f}")
            if SAVE_IMG_DEBUG:
                save_image(
                    warped_sign_mask, f"tmp/{index:02d}_M1_warped_sign_mask.png"
                )
            continue

        # calculate euclidean distance between patch_src_transformed and patch_tgt
        transform_l2_error = np.linalg.norm(
            patch_src_transformed - patch_tgt, axis=1
        ).mean()

        # calculate transform matrix M2 between transformed patch points and labeled patch points
        patch_tf_matrix = get_perspective_transform(
            torch.from_numpy(patch_src).unsqueeze(0),
            torch.from_numpy(patch_tgt).unsqueeze(0),
        )
        transform_func = kornia_tf.warp_perspective

        # apply relighting to transformed synthetic patch
        tmp_patch = torch.zeros_like(patch)
        patch.mul_(alpha).add_(beta)
        patch.clamp_(0, 1)
        tmp_patch[:, h_min:h_max, w_min:w_max] = patch[
            :, h_min + shift : h_max + shift, w_min:w_max
        ]
        patch = tmp_patch
        warped_patch = transform_func(
            patch.unsqueeze(0),
            patch_tf_matrix,
            dsize=(img_height, img_width),
            mode="bilinear",
            padding_mode="zeros",
        )
        warped_mask = transform_func(
            patch_mask.unsqueeze(0),
            patch_tf_matrix,
            dsize=(img_height, img_width),
            mode="nearest",
            padding_mode="zeros",
        )
        warped_patch.clamp_(0, 1)

        if SAVE_IMG_DEBUG:
            save_image(warped_patch, f"tmp/{index:02d}_M2_warped_patch.png")
            save_image(warped_mask, f"tmp/{index:02d}_M2_warped_mask.png")

        # real_patch = warped_mask[0].permute(1, 2, 0) * image
        warped_mask = warped_mask.bool()
        real_patch = torch.masked_select(torch_image, warped_mask)
        reap_patch = torch.masked_select(warped_patch, warped_mask)
        if SAVE_IMG_DEBUG:
            save_image(
                torch_image * warped_mask, f"tmp/{index:02d}_real_patch.png"
            )
            save_image(warped_patch, f"tmp/{index:02d}_reap_patch.png")

        # calculate relighting error between transformed synthetic patch and real patch
        relighting_l2_error = ((real_patch - reap_patch) ** 2).mean().sqrt()

        print()
        print(f"transform_l2_error: {transform_l2_error:.4f}")
        print(f"relighting_l2_error: {relighting_l2_error.item():.4f}")

        geometric_errors.append(transform_l2_error)
        lighting_errors.append(relighting_l2_error.item())

    # plot histogram of geometric_errors and save plot
    plt.hist(geometric_errors, bins=100)
    plt.savefig("tmp/geometric_errors.png")
    plt.clf()

    plt.hist(lighting_errors, bins=100)
    plt.savefig("tmp/relighting_errors.png")
    plt.clf()

    # print statistics for errors
    print("geometric error:")
    print(f"mean: {np.mean(geometric_errors)}")
    print(f"std: {np.std(geometric_errors)}")
    print(f"max: {np.max(geometric_errors)}")
    print(f"min: {np.min(geometric_errors)}")
    print()
    print("lighting error:")
    print(f"mean: {np.mean(lighting_errors)}")
    print(f"std: {np.std(lighting_errors)}")
    print(f"max: {np.max(lighting_errors)}")
    print(f"min: {np.min(lighting_errors)}")


if __name__ == "__main__":
    # flag to control whether to save images for debugging
    SAVE_IMG_DEBUG = False
    main()
