import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skimage
from tqdm import tqdm
import torch
import torch.nn.functional as F
import cv2 as cv
import pathlib
import pickle

import torchvision.transforms.functional as T
import kornia.geometry.transform as kornia_tf
from kornia.geometry.transform import get_perspective_transform
from adv_patch_bench.transforms import util, lighting_tf

from hparams import OBJ_DIM_DICT, TS_COLOR_DICT

# file directory where images are stored
file_dir = "/data/shared/adv-patch-bench/reap_realism_test/images/"

# path to directory where patch files are stored
patch_path = pathlib.Path(
    "/data/shared/adv-patch-bench/reap_realism_test/synthetic-load-64-15-1-0.4-0-0-pd64-bg50-augimg1-rp2_1e-05_0_1_1000_adam_0.01_False"
)

# read in annotation data from csv file
annotation_df = pd.read_csv(
    "realism_test_anno.csv"
)

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

# import pdb; pdb.set_trace()
# list of traffic sign classes
traffic_sign_classes = list(TS_COLOR_DICT.keys())
# remove 'other' class from list
traffic_sign_classes.remove("other")

# list of point colors for visualizing image points
POINT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# flag to control whether to save images for debugging
save_images_for_debugging = False

# lists to store geometric and lighting errors for each image
geometric_errors = []
lighting_errors = []

for index, row in tqdm(annotation_df.iterrows()):
    is_clean = index % 2 == 0
    if is_clean:
        alpha = None
        beta = None
    else:
        assert alpha != None, "alpha must be specified for adversarial images"
        assert beta != None, "alpha must be specified for adversarial images"

    # obj_class = traffic_sign_classes[(index // 4) % len(traffic_sign_classes)]
    obj_class = traffic_sign_classes[(index // 2) % len(traffic_sign_classes)]

    # get file path for image
    filename = row["file_name"]
    filepath = os.path.join(file_dir, filename)
    if not os.path.exists(filepath): # check if file exists
        print("File not found: ", filepath)
        continue
    
    # read image 
    image = skimage.io.imread(filepath)

    # get image dimensions
    img_width, img_height = row["width"], row["height"]
    
    # get labeled coordinates for object in image
    sign_x1, sign_y1, sign_x2, sign_y2, sign_x3, sign_y3, sign_x4, sign_y4 = (
        row["sign_x1"],
        row["sign_y1"],
        row["sign_x2"],
        row["sign_y2"],
        row["sign_x3"],
        row["sign_y3"],
        row["sign_x4"],
        row["sign_y4"],
    )

    # get labeled coordinates for patch in image
    patch_x1, patch_y1, patch_x2, patch_y2, patch_x3, patch_y3, patch_x4, patch_y4 = (
        row["patch_x1"],
        row["patch_y1"],
        row["patch_x2"],
        row["patch_y2"],
        row["patch_x3"],
        row["patch_y3"],
        row["patch_x4"],
        row["patch_y4"],
    )

    # read in patch and mask from file
    with open(str(patch_path / obj_class / "adv_patch.pkl"), "rb") as file:
        patch, patch_mask = pickle.load(file)

    patch_size_in_pixel = patch.shape[1]

    hw_ratio_dict = OBJ_DIM_DICT["mapillary_no_color"]["hw_ratio"]
    obj_size_dict = OBJ_DIM_DICT["mapillary_no_color"]["size_mm"]

    # get aspect ratio for current object class
    # hw_ratio = hw_ratio_dict[(index // 4) % len(traffic_sign_classes)]
    hw_ratio = hw_ratio_dict[(index // 2) % len(traffic_sign_classes)]

    obj_shape = obj_class_to_shape[obj_class]
    
    # generate mask for object in image
    sign_mask, src = util.gen_sign_mask(
        obj_shape, hw_ratio=hw_ratio, obj_width_px=patch_size_in_pixel
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
    elif obj_class in ("circle", "up-triangle"):
        factor = 0.1
    else:
        factor = 0.0
    shift = int(h_max * factor)
    h_min -= shift
    h_max -= shift
    
    patch_src = np.array(
        [[w_min, h_min], [w_max, h_min], [w_max, h_max], [w_min, h_max]]
    ).astype(np.float32)

    if not is_clean:
        patch_tgt = np.array(
            [
                [patch_x1, patch_y1],
                [patch_x2, patch_y2],
                [patch_x3, patch_y3],
                [patch_x4, patch_y4],
            ]
        ).astype(np.float32)

    if len(src) == 3:
        sign_x1, sign_y1, sign_x2, sign_y2, sign_x3, sign_y3 = (
            float(sign_x1),
            float(sign_y1),
            float(sign_x2),
            float(sign_y2),
            float(sign_x3),
            float(sign_y3),
        )
        tgt = np.array(
            [[sign_x1, sign_y1], [sign_x2, sign_y2], [sign_x3, sign_y3]]
        ).astype(np.float32)

        M1 = torch.from_numpy(cv.getAffineTransform(src, tgt)).unsqueeze(0).float()

        M1_homography = torch.cat(
            (M1, torch.tensor([0, 0, 1]).view(1, 1, 3).float()), dim=1
        )  # add [0, 0, 1] to M1
        
        # apply affube transform to src patch coordinates
        patch_src_transformed = cv.perspectiveTransform(
            patch_src.reshape(1, -1, 2), M1_homography[0].numpy()
        )

        transform_func = kornia_tf.warp_affine
    else:
        sign_x1, sign_y1, sign_x2, sign_y2, sign_x3, sign_y3, sign_x4, sign_y4 = (
            float(sign_x1),
            float(sign_y1),
            float(sign_x2),
            float(sign_y2),
            float(sign_x3),
            float(sign_y3),
            float(sign_x4),
            float(sign_y4),
        )
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
        M1 = get_perspective_transform(src, tgt)

        src = src[0]  # unsqueeze(0) above
        tgt = tgt[0]  # unsqueeze(0) above

        # apply perspective transform to src patch coordinates
        patch_src_transformed = cv.perspectiveTransform(
            patch_src.reshape(1, -1, 2), M1[0].numpy()
        )

        transform_func = kornia_tf.warp_perspective

    patch_src_transformed = patch_src_transformed[0]  # unsqueeze(0) above

    if save_images_for_debugging:
        for i in range(len(src)):
            cv.circle(image, (int(src[i][0]), int(src[i][1])), 5, POINT_COLORS[i], -1)
            cv.circle(image, (int(tgt[i][0]), int(tgt[i][1])), 5, POINT_COLORS[i], -1)

        for i in range(len(patch_src)):
            cv.circle(
                image,
                (int(patch_src[i][0]), int(patch_src[i][1])),
                5,
                POINT_COLORS[i],
                -1,
            )
            cv.circle(
                image,
                (int(patch_src_transformed[i][0]), int(patch_src_transformed[i][1])),
                5,
                POINT_COLORS[i],
                -1,
            )
            if not is_clean:
                cv.circle(
                    image,
                    (int(patch_tgt[i][0]), int(patch_tgt[i][1])),
                    5,
                    POINT_COLORS[i],
                    -1,
                )

        # resize image and scale down by 8
        image_resized = cv.resize(image, (img_width // 8, img_height // 8))
        print("saving image")
        skimage.io.imsave("test.png", image_resized)

    
    if not is_clean:
        # calculate euclidean distance between patch_src_transformed and patch_tgt
        transform_l2_error = np.linalg.norm(patch_src_transformed - patch_tgt, axis=1).mean()
    
    # pad patch_mask, patch and canonical sign for affine or perspective trasnform
    pad_size = (img_height, img_width)

    _, hh, ww = np.where(patch_mask.numpy())
    h_min, h_max = hh.min(), hh.max() + 1
    w_min, w_max = ww.min(), ww.max() + 1
    patch_width = w_max - w_min
    patch_height = h_max - h_min

    padding = (
        0,  # left
        0,  # top
        max(0, pad_size[1] - patch.shape[2]),  # right
        max(0, pad_size[0] - patch.shape[1]),  # bottom
    )

    patch_padded = T.pad(patch * patch_mask, padding)
    patch_mask_padded = T.pad(patch_mask, padding)

    padding = (
        0,  # left
        0,  # top
        max(0, pad_size[1] - sign_mask.shape[1]),  # right
        max(0, pad_size[0] - sign_mask.shape[0]),  # bottom
    )
    padding = (0, 0, max(0, pad_size[1] - sign_mask.shape[1]), max(0, pad_size[0] - sign_mask.shape[0]))
    sign_mask_padded = T.pad(torch.from_numpy(sign_mask * 1.0).float().unsqueeze(0), padding)
    
    if save_images_for_debugging:
        skimage.io.imsave("test_patch.png", patch_padded.permute(1, 2, 0))
        skimage.io.imsave("test_patch_mask.png", patch_mask_padded.permute(1, 2, 0))
        skimage.io.imsave("test_sign_mask.png", sign_mask_padded.permute(1, 2, 0))

    patch_padded = patch_padded.unsqueeze(0)
    patch_mask_padded = patch_mask_padded.unsqueeze(0)
    sign_mask_padded = sign_mask_padded.unsqueeze(0)

    # warp patch
    warped_patch = transform_func(
        patch_padded,
        M1,
        dsize=(img_height, img_width),
        mode="bilinear",
        padding_mode="zeros",
    )

    # warp patch mask
    warped_mask = transform_func(
        patch_mask_padded,
        M1,
        dsize=(img_height, img_width),
        mode="bilinear",
        padding_mode="zeros",
    )

    # warp canonical sign mask (to calculate alpha and beta)
    warped_sign_mask = transform_func(
        sign_mask_padded,
        M1,
        dsize=(img_height, img_width),
        mode="bilinear",
        padding_mode="zeros",
    )

    if is_clean:
        # convert numpy array to tensor
        torch_image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        cropped_traffic_sign = torch.masked_select(
            torch_image, warped_sign_mask.bool()
        )

        # calculate relighting parameters
        alpha, beta = lighting_tf.compute_relight_params(cropped_traffic_sign.numpy().reshape(-1, 1))
        # print("alpha", alpha)
        # print("beta", beta)
        continue
        
    warped_patch.clamp_(0, 1)
    warped_mask.clamp_(0, 1)
    warped_sign_mask.clamp_(0, 1)

    if save_images_for_debugging:
        skimage.io.imsave("test_M1_warped_patch.png", warped_patch[0].permute(1, 2, 0))
        skimage.io.imsave("test_M1_warped_patch_mask.png", warped_mask[0].permute(1, 2, 0))
        skimage.io.imsave("test_M1_warped_sign_mask.png", warped_sign_mask[0].permute(1, 2, 0))
    
    # calculate transform matrix M2 between transformed patch points and labeled patch points
    M2 = get_perspective_transform(
        torch.from_numpy(patch_src_transformed).unsqueeze(0),
        torch.from_numpy(patch_tgt).unsqueeze(0),
    )
    transform_func = kornia_tf.warp_perspective
    warped_patch = transform_func(
        warped_patch,
        M2,
        dsize=(img_height, img_width),
        mode="bilinear",
        padding_mode="zeros",
    )

    warped_mask = transform_func(
        warped_mask,
        M2,
        dsize=(img_height, img_width),
        mode="bilinear",
        padding_mode="zeros",
    )

    warped_patch.clamp_(0, 1)
    warped_mask.clamp_(0, 1)
    
    if save_images_for_debugging:
        skimage.io.imsave("test_M2_warped_patch.png", warped_patch[0].permute(1, 2, 0))
        skimage.io.imsave("test_M2_warped_mask.png", warped_mask[0].permute(1, 2, 0))

    real_patch = warped_mask[0].permute(1, 2, 0) * image
    if save_images_for_debugging:
        skimage.io.imsave("test_tgt_sign.png", warped_sign_mask[0].permute(1, 2, 0) * image)
        image_resized = cv.resize(real_patch.numpy(), (img_width // 8, img_height // 8))
        skimage.io.imsave("test_real_patch.png", image_resized)

    # apply relighting to transformed synthetic patch
    warped_patch = warped_patch.mul_(alpha).add_(beta)

    # calculate relighting error between transformed synthetic patch and real patch
    real_patch = real_patch / 255.0
    relighting_l2_error = (
        (real_patch - warped_patch[0].permute(1, 2, 0)) ** 2
    ).sum() ** 0.5

    print("transform_l2_error", transform_l2_error)
    print("relighting_l2_error", relighting_l2_error)

    geometric_errors.append(transform_l2_error)
    lighting_errors.append(relighting_l2_error.item())

    # import pdb; pdb.set_trace()


# plot histogram of geometric_errors and save plot
plt.hist(geometric_errors, bins=100)
plt.savefig("test_geometric_errors.png")
plt.clf()

plt.hist(lighting_errors, bins=100)
plt.savefig("test_relighting_errors.png")
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



