"""Main script for generating both geometric and relighting REAP transforms."""

import argparse
import json
import os
from os import listdir
from os.path import isfile, join

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from kornia.geometry.transform import (
    get_perspective_transform,
    resize,
    warp_affine,
    warp_perspective,
)
from PIL import Image
from tqdm.auto import tqdm

from adv_patch_bench.dataloaders.image_only_dataset import ImageOnlyFolder
from adv_patch_bench.models import build_classifier
from adv_patch_bench.transforms import geometric_tf, lighting_tf, util
from adv_patch_bench.utils import (
    draw_from_contours,
    img_numpy_to_torch,
    pad_image,
)

DATASET = "mapillary_vistas"
# DATASET = 'bdd100k'

if DATASET == "mapillary_vistas":
    TRAFFIC_SIGN_LABEL = 95
elif DATASET == "bdd100k":
    TRAFFIC_SIGN_LABEL = "traffic sign"

CLASS_LIST = [
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
    "other-0.0-0.0",
]

SHAPE_LIST = [
    "circle",
    "triangle",
    "triangle_inverted",
    "diamond",
    "square",
    "rect",
    "pentagon",
    "octagon",
    "other",
]

# PLOT_FOLDER = 'mapillaryvistas_plots_model_3_updated_optimized'
PLOT_FOLDER = "delete_mapillaryvistas_plots_model_3_updated_optimized"
NUM_IMGS_PER_PLOT = 100
COLUMN_NAMES = "abcdefghijklmnopqrstuvwxyz"
ROW_NAMES = list(range(NUM_IMGS_PER_PLOT))


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Part classification", add_help=False
    )
    parser.add_argument("--data", default="~/data/shared/", type=str)
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained model on ImageNet-1k",
    )
    parser.add_argument(
        "--output-dir", default="./", type=str, help="output dir"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers per process",
    )
    parser.add_argument(
        "--batch-size",
        default=256,
        type=int,
        help="mini-batch size per device.",
    )
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument("--distributed", default=False, type=bool)
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://localhost:10001",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB")
    # TODO
    parser.add_argument("--dataset", required=True, type=str, help="Dataset")
    parser.add_argument(
        "--num-classes", default=10, type=int, help="Number of classes"
    )
    parser.add_argument(
        "--experiment",
        required=False,
        type=str,
        help="Type of experiment to run",
    )
    # Unused
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--optim", default="sgd", type=str)
    parser.add_argument("--betas", default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument(
        "--resume", default="", type=str, help="path to latest checkpoint"
    )
    parser.add_argument("--split", default="training", type=str)
    return parser


def draw_vertices(traffic_sign, points, color=[0, 255, 0]):
    size = traffic_sign.size(-1)
    vert = draw_from_contours(np.zeros((size, size, 3)), points, color=color)
    vert = img_numpy_to_torch(cv.dilate(vert, None))
    vert_mask = (vert.sum(0, keepdim=True) > 0).float()
    return (1 - vert_mask) * traffic_sign + vert_mask * vert


def compute_example_transform(
    traffic_sign,
    mask,
    predicted_class,
    demo_patch,
    patch_size_in_mm=150,
    patch_size_in_pixel=32,
):

    # TODO: temporary only. delete afterwards
    alpha = None
    beta = None
    src = None

    height, width, _ = traffic_sign.shape
    assert width == height
    size = width
    predicted_shape = predicted_class.split("-")[0]
    bool_mask = (mask == 255).astype(np.uint8)
    traffic_sign = img_numpy_to_torch(traffic_sign)

    # Get vertices of mask
    vertices, hull = geometric_tf.get_corners(bool_mask)
    hull_mask = cv.drawContours(np.zeros_like(bool_mask), [hull], -1, (1,), 1)
    hull_draw_points = np.stack(np.where(hull_mask), axis=1)[:, ::-1]
    ellipse = cv.fitEllipse(hull_draw_points)
    ellipse_mask = cv.ellipse(
        np.zeros_like(bool_mask, dtype=np.float32), ellipse, (1,), thickness=-1
    )
    ellipse_error = (
        np.abs(ellipse_mask - bool_mask.astype(np.float32)).sum()
        / bool_mask.sum()
    )

    shape = geometric_tf.get_shape_from_vertices(vertices)
    tgt = geometric_tf.get_box_vertices(vertices, shape).astype(np.int64)
    traffic_sign = draw_vertices(traffic_sign, tgt, color=[0, 0, 255])

    # Determine polygon shape from vertices
    shape = geometric_tf.get_shape_from_vertices(vertices)

    if predicted_shape == "other":
        group = 3
    elif ellipse_error < 0.1:
        # Check circle based on ellipse fit error
        shape = "circle"
        vertices = ellipse
        group = 1 if predicted_shape == "circle" else 2
    else:
        if (shape != "other" and predicted_shape == shape) or (
            shape == "rect" and predicted_shape != "other"
        ):
            # Both classifier and verifier agree on some polygons or
            # the sign symbol is on a square sign (assume that dimension is
            # equal to the actual symbol)
            group = 1
        else:
            # Disagree but not other
            group = 2

    if shape != "other":
        tgt = geometric_tf.get_box_vertices(vertices, shape).astype(np.int64)
        # Filter some vertices that might be out of bound
        if (tgt < 0).any() or (tgt >= size).any():
            # TODO
            # raise ValueError('Out-of-bound vertices')
            pad_size = int(max((tgt - size + 1).max(), (-tgt).max()))
            # Pad traffic_sign, bool_mask, tgt, size
            traffic_sign = pad_image(traffic_sign, pad_size=pad_size)
            bool_mask = pad_image(bool_mask, pad_size=pad_size)
            tgt += pad_size
            size = traffic_sign.size(-1)
            # group = 2
            # results.append([traffic_sign, shape, predicted_shape, predicted_class, 4])

        # If shape is not other, draw vertices
        traffic_sign = draw_vertices(traffic_sign, tgt)

        # Group 1: draw both vertices and patch
        if group == 1:
            # FIXME: fix args and sign_canonical
            sign_mask, src = util.gen_sign_mask(
                predicted_class, patch_size_in_pixel, patch_size_in_mm
            )

            old_patch = torch.masked_select(
                traffic_sign, torch.from_numpy(bool_mask).bool()
            )
            alpha, beta = lighting_tf.compute_relight_params(old_patch.numpy().reshape(-1, 1))

            # TODO: run attack, optimize patch location, etc.
            new_demo_patch = demo_patch.clone()
            new_demo_patch.clamp_(0, 1).mul_(alpha).add_(beta).clamp_(0, 1)
            sign_size_in_pixel = sign_canonical.size(-1)
            begin = (sign_size_in_pixel - patch_size_in_pixel) // 2
            end = begin + patch_size_in_pixel
            sign_canonical[:-1, begin:end, begin:end] = new_demo_patch
            sign_canonical[-1, begin:end, begin:end] = 1
            # Crop patch that is not on the sign
            sign_canonical *= sign_mask
            patch_mask = torch.zeros(
                (1, sign_size_in_pixel, sign_size_in_pixel)
            )
            patch_mask[:, begin:end, begin:end] = 1

            # Compute perspective transform
            src = np.array(src).astype(np.float32)
            tgt = tgt.astype(np.float32)

            if len(src) == 3:
                print(len(src))
                print(len(tgt))
                print(predicted_class)
                M = (
                    torch.from_numpy(cv.getAffineTransform(src, tgt))
                    .unsqueeze(0)
                    .float()
                )
                transform_func = warp_affine
            else:
                src = torch.from_numpy(src).unsqueeze(0)
                tgt = torch.from_numpy(tgt).unsqueeze(0)
                M = get_perspective_transform(src, tgt)

                transform_func = warp_perspective

            warped_patch = transform_func(
                sign_canonical.unsqueeze(0),
                M,
                (size, size),
                mode="bicubic",
                padding_mode="zeros",
            )[0].clamp(0, 1)

            alpha_mask = warped_patch[-1].unsqueeze(0)
            traffic_sign = (
                1 - alpha_mask
            ) * traffic_sign + alpha_mask * warped_patch[:-1]

            # DEBUG
            # print(shape, predicted_class, group)
            # save_image(traffic_sign, 'test.png')
            # import pdb
            # pdb.set_trace()

        # DEBUG
        # if group in (1, 2):
        #     show_list.append(traffic_sign)
        # if len(show_list) > 100:
        #     save_image(show_list, 'test.png')
        #     import pdb
        #     pdb.set_trace()

    # DEBUG
    # if 'triangle' in shape:
    #     save_image(cropped_sign, 'test.png')
    #     import pdb
    #     pdb.set_trace()

    return traffic_sign, shape, group, tgt, alpha, beta


def plot_subgroup(
    adversarial_images,
    metadata,
    group,
    shape,
    plot_folder=PLOT_FOLDER,
    num_imgs_per_plot=NUM_IMGS_PER_PLOT,
):
    if len(adversarial_images) == 0:
        return
    if not os.path.exists("{}/{}/{}".format(plot_folder, shape, group)):
        os.makedirs(
            "{}/{}/{}".format(plot_folder, shape, group), exist_ok=False
        )

    fig = None
    num_images_plotted = 0
    batch_number = 0
    df_data = []

    for i, adv_img in enumerate(adversarial_images):
        filename, obj_id, predicted_class, tgt, alpha, beta = metadata[i]
        predicted_shape = predicted_class.split("-")[0]

        if fig is None:
            fig, ax = plt.subplots(int(num_imgs_per_plot / 5), 5)
            fig.set_figheight(int(num_imgs_per_plot / 5) * 5)
            fig.set_figwidth(5 * 5)

        col = num_images_plotted % 5
        row = num_images_plotted // 5
        col_name = COLUMN_NAMES[col]
        row_name = ROW_NAMES[row]

        if shape == "triangle_inverted":
            plot_shape_name = "t_inv"
        elif shape == "diamond":
            plot_shape_name = "dmnd"
        else:
            plot_shape_name = shape

        title = "id({}, {}) | ({}, {}) | cord({}{})".format(
            filename[:6],
            obj_id,
            plot_shape_name,
            predicted_class,
            row_name,
            col_name,
        )

        ax[row][col].set_title(title, fontsize=10)
        ax[row][col].imshow(adv_img)

        if num_images_plotted == num_imgs_per_plot - 1:
            fig.savefig(
                "{}/{}/{}/batch_{}.png".format(
                    plot_folder, shape, group, batch_number
                ),
                bbox_inches="tight",
                pad_inches=0,
            )
            batch_number += 1
            num_images_plotted = -1
            plt.close(fig)
            fig = None

        num_images_plotted += 1

        if tgt.ndim == 3:
            tgt = tgt[0]

        df_data.append(
            [
                filename,
                obj_id,
                shape,
                predicted_shape,
                predicted_class,
                group,
                batch_number,
                row_name,
                col_name,
                tgt.tolist(),
                alpha,
                beta,
            ]
        )

    if fig:
        fig.savefig(
            "{}/{}/{}/batch_{}.png".format(
                plot_folder, shape, group, batch_number
            ),
            bbox_inches="tight",
            pad_inches=0,
        )
    return df_data


def main(args):

    # Arguments
    # min_area = 1600
    # max_num_imgs = 200
    cudnn.benchmark = True

    if DATASET == "mapillary_vistas":
        if split == "training":
            data_dir = "/data/shared/mapillary_vistas/training/"
        elif split == "validation":
            data_dir = "/data/shared/mapillary_vistas/validation/"
    elif DATASET == "bdd100k":
        data_dir = "/data/shared/bdd100k/images/10k/train/"

    # data_dir = '/data/shared/mtsd_v2_fully_annotated/'
    # model_path = '/home/nab_126/adv-patch-bench/model_weights/resnet18_cropped_signs_good_resolution_and_not_edge_10_labels.pth'

    # Compute stats of best model
    model = build_classifier(args)[0]
    model.eval()

    if DATASET == "mapillary_vistas":
        img_path = join(data_dir, "traffic_signs")
        img_files = sorted(
            [
                join(img_path, f)
                for f in listdir(img_path)
                if isfile(join(img_path, f)) and f.endswith(".png")
            ]
        )
        mask_path = join(data_dir, "masks")
        mask_files = sorted(
            [
                join(mask_path, f)
                for f in listdir(mask_path)
                if isfile(join(mask_path, f)) and f.endswith(".png")
            ]
        )
    # TODO: Read in panoptic file
    elif DATASET == "bdd100k":
        panoptic_json_path = (
            "/data/shared/bdd100k/labels/pan_seg/polygons/pan_seg_train.json"
        )

        with open(panoptic_json_path) as panoptic_file:
            panoptic = json.load(panoptic_file)

        # creating same mapping for bdd100k
        panoptic_per_image_id = {}
        for image_annotation in tqdm(panoptic):
            filename = image_annotation["name"]
            image_id = filename.split(".jpg")[0]
            annotation = {}
            annotation["filename"] = filename
            annotation["image_id"] = image_id
            segments_info = []
            for label in image_annotation["labels"]:
                label_dict = {}

                # TODO: check if occluded and exclude if True
                if label["category"] == "traffic sign":
                    # if label['category'] == 'traffic sign' or label['category'] == 'traffic sign frame':
                    # label_dict['id'] = label['id']
                    label_dict["id"] = 26
                    label_dict["category_id"] = label["category"]
                    for sign in label["poly2d"]:
                        vertices = sign["vertices"]
                        vertices = np.array(vertices)

                        x_cords, y_cords = vertices[:, 0], vertices[:, 1]
                        xmin = min(x_cords)
                        xmax = max(x_cords)
                        ymin = min(y_cords)
                        ymax = max(y_cords)
                        width = xmax - xmin
                        height = ymax - ymin

                        label_dict["area"] = int(width) * int(height)
                        label_dict["bbox"] = [
                            int(xmin),
                            int(ymin),
                            int(width),
                            int(height),
                        ]
                        segments_info.append(label_dict)
            annotation["segments_info"] = segments_info
            panoptic_per_image_id[image_id] = annotation

        # data_dir = '/data/shared/bdd100k/images/10k/train/'
        # label_path = "/data/shared/bdd100k/labels/pan_seg/bitmasks/train/"
        filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        img_path = data_dir

        np.random.seed(1111)
        np.random.shuffle(filenames)

    demo_patch = torchvision.io.read_image("demo.png").float()[:3, :, :] / 255
    demo_patch = resize(demo_patch, (32, 32))

    print("[INFO] constructing a dataloader for cropped traffic signs...")
    bs = args.batch_size
    transform = transforms.Compose(
        [
            # transforms.RandomEqualize(p=1.0),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageOnlyFolder(img_files, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True
    )
    y_hat = torch.zeros(len(dataset), dtype=torch.long)
    print("[INFO] classifying traffic signs...")

    # model.eval()
    with torch.no_grad():
        for i, images in enumerate(dataloader):
            y_hat[i * bs : (i + 1) * bs] = model(images.cuda()).argmax(-1).cpu()

    print("==> Predicted label distribution")
    print(CLASS_LIST)
    print(torch.bincount(y_hat) / len(y_hat))
    print(len(dataset) * torch.bincount(y_hat) / len(y_hat))
    print()

    print("[INFO] running detection algorithm")

    missed_alpha_beta_df = pd.DataFrame()
    df_filenames = []
    df_alphas = []
    df_betas = []

    subgroup_to_images = {}
    for img_file, mask_file, y in tqdm(zip(img_files, mask_files, y_hat)):
        filename = img_file.split("/")[-1]

        assert filename == mask_file.split("/")[-1]
        image = np.asarray(Image.open(img_file))
        Image.open(mask_file).save("test_mask.png")
        mask = np.asarray(Image.open(mask_file))

        try:
            bool_mask = (mask == 255).astype(np.uint8)
            traffic_sign = img_numpy_to_torch(image)
            old_patch = torch.masked_select(
                traffic_sign, torch.from_numpy(bool_mask).bool()
            )
            alpha, beta = lighting_tf.compute_relight_params(old_patch.numpy().reshape(-1, 1))

            df_filenames.append(filename)
            df_alphas.append(alpha)
            df_betas.append(beta)
        except ValueError:
            continue

        output = compute_example_transform(
            image,
            mask,
            CLASS_LIST[y],
            demo_patch,
            patch_size_in_mm=150,
            patch_size_in_pixel=32,
        )

        predicted_class = CLASS_LIST[y]
        # predicted_shape = predicted_class.split("-")[0]
        obj_id = filename.split("_")[-1].split(".")[0]

        adv_image, shape, group, tgt, alpha, beta = output

        if (shape, group) not in subgroup_to_images:
            subgroup_to_images[(shape, group)] = []

        adv_image = adv_image.permute(1, 2, 0).numpy()
        subgroup_to_images[(shape, group)].append(
            [adv_image, filename, obj_id, predicted_class, tgt, alpha, beta]
        )

    missed_alpha_beta_df["filename"] = df_filenames
    missed_alpha_beta_df["alpha"] = df_alphas
    missed_alpha_beta_df["beta"] = df_betas
    missed_alpha_beta_df.to_csv(
        f"mapillary_vistas_{split}_alpha_beta.csv", index=False
    )

    column_names = [
        "filename",
        "object_id",
        "shape",
        "predicted_shape",
        "predicted_class",
        "group",
        "batch_number",
        "row",
        "column",
        "tgt",
        "alpha",
        "beta",
    ]
    csv_filename = f"{DATASET}_{split}_data.csv"

    df = pd.DataFrame(columns=column_names)
    for subgroup in tqdm(subgroup_to_images):
        shape, group = subgroup
        data = subgroup_to_images[(shape, group)]
        data = np.array(data, dtype=object)
        adversarial_images = data[:, 0]
        metadata = data[:, 1:]
        df_data_to_add = plot_subgroup(
            adversarial_images,
            metadata,
            group,
            shape,
            plot_folder=PLOT_FOLDER,
            num_imgs_per_plot=NUM_IMGS_PER_PLOT,
        )
        df_to_add = pd.DataFrame(df_data_to_add, columns=column_names)
        df = pd.concat([df, df_to_add], axis=0)
        df.to_csv(csv_filename, index=False)

    offset_df = pd.read_csv(f"offset_{split}.csv")
    df["filename_png"] = (
        df["filename"].str.split("_").str[:-1].str.join("_") + ".png"
    )
    df["filename"] = (
        df["filename"].str.split("_").str[:-1].str.join("_") + ".jpg"
    )

    df["object_id"] = df["object_id"].astype("int64")
    offset_df["obj_id"] = offset_df["obj_id"].astype("int64")
    df = df.merge(
        right=offset_df,
        left_on=["filename", "object_id"],
        right_on=["filename", "obj_id"],
    )
    df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Example Transform", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    split = args.split
    main(args)
