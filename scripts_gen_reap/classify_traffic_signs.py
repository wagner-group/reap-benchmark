import json
import os
import pprint
from os import listdir
from os.path import isfile, join

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms.functional as TF
from matplotlib import cm
from PIL import Image, ImageColor, ImageDraw, ImageFont
from skimage.exposure import match_histograms
from torch.serialization import save
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks, make_grid, save_image
from tqdm.auto import tqdm

from adv_patch_bench.models.common import Normalize
from adv_patch_bench.transforms import get_box_from_ellipse
from adv_patch_bench.utils import get_image_files, load_annotation, pad_image

plt.rcParams["savefig.bbox"] = "tight"

plt.style.use("seaborn-white")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"

POLYGON_ERROR = 0.04
SELECTED_SHAPES = [
    "octagon,36,36",
    "diamond,36,36",
    "pentagon,36,36",
    "rect,36,48",
    "rect,30,36",
]


def img_numpy_to_torch(img):
    if img.ndim == 3:
        return torch.from_numpy(img).permute(2, 0, 1) / 255.0
    return torch.from_numpy(img).permute(0, 3, 1, 2) / 255.0


def count_blobs(image):
    labeled, num_blobs = scipy.ndimage.label(image, structure=np.ones((3, 3)))
    # DEBUG
    # blob_sizes = []
    # for i in range(1, num_blobs + 1):
    #     blob_sizes.append((labeled == i).sum())
    # print(blob_sizes)
    return num_blobs


def detect_polygon(contour):
    eps = cv.arcLength(contour, True) * POLYGON_ERROR
    vertices = cv.approxPolyDP(contour, eps, True)
    return vertices


def draw_from_contours(img, contours, color=[0, 0, 255, 255]):
    if not isinstance(contours, list):
        contours = [contours]
    for contour in contours:
        if contour.ndim == 3:
            contour_coord = (contour[:, 0, 1], contour[:, 0, 0])
        elif contour.ndim == 2:
            contour_coord = (contour[:, 1], contour[:, 0])
        else:
            raise ValueError("Invalid contour shape.")
        img[contour_coord] = color
    return img


def show(imgs, num_cols=2, titles=None):
    num_imgs = len(imgs)
    num_rows = int(np.ceil(num_imgs / num_cols))
    if not isinstance(imgs, (list, np.ndarray)):
        imgs = [imgs]
    fix, axs = plt.subplots(
        ncols=num_cols, nrows=num_rows, figsize=(10, 6 / num_cols * num_rows)
    )
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = TF.to_pil_image(img)
        row, col = i // num_cols, i % num_cols
        if isinstance(axs, matplotlib.axes.Axes):
            axs.imshow(np.asarray(img))
            axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs.set_title(titles)
        elif axs.ndim == 1:
            axs[col].imshow(np.asarray(img))
            axs[col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if titles is not None:
                axs[col].set_title(titles[i])
        else:
            axs[row, col].imshow(np.asarray(img))
            axs[row, col].set(
                xticklabels=[], yticklabels=[], xticks=[], yticks=[]
            )
            if titles is not None:
                axs[row, col].set_title(titles[i])


def show_img_with_segment(label, panoptic_per_image_id, data_dir, min_area=0):
    img_path = join(data_dir, "images")
    label_path = join(data_dir, "v2.0/panoptic/")

    filenames = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    np.random.shuffle(filenames)

    num_imgs, max_num_imgs = 0, 10
    imgs = []
    for filename in filenames:
        if num_imgs >= max_num_imgs:
            break
        label_found = False
        img_id = filename.split(".")[0]
        segment = panoptic_per_image_id[img_id]["segments_info"]
        panoptic = np.array(Image.open(join(label_path, f"{img_id}.png")))

        # Find label id of each object present in the scene
        highlight_ids = []
        for obj in segment:
            if obj["category_id"] == label and obj["area"] >= min_area:
                label_found = True
                highlight_ids.append(obj["id"])

        if not label_found:
            continue
        num_imgs += 1

        # Get segmentation mask from panoptic
        img_pil = Image.open(join(img_path, filename))
        img = np.array(img_pil)
        mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        for i in highlight_ids:
            mask += panoptic[:, :, 0] == i
        bool_mask = (mask > 0).astype(np.float32)
        mask = np.stack(
            [
                bool_mask,
            ]
            * 4,
            axis=-1,
        )
        color_tuple = ImageColor.getrgb("green")
        mask[:, :, :3] *= color_tuple
        mask[:, :, 3] = bool_mask * 0.5
        mask = Image.fromarray(np.uint8(mask * 255))
        # mask = Image.fromarray(np.uint8(cm.gist_earth(mask) * 255))
        img = Image.alpha_composite(
            img_pil.convert("RGBA"), mask.convert("RGBA")
        )
        imgs.append(img)

    show(imgs, num_cols=2)
    plt.savefig("test.png", dpi=600)


def show_img_patch(
    model,
    label,
    panoptic_per_image_id,
    data_dir,
    max_num_imgs=1000,
    min_area=0,
    conf_thres=0.8,
    pad=0.05,
    num_classes=6,
    device="cuda",
):
    img_path = join(data_dir, "images")
    label_path = join(data_dir, "v2.0/panoptic/")

    filenames = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    np.random.shuffle(filenames)

    patches, resized_patches, masks, ids = [], [], [], []
    with tqdm(total=max_num_imgs) as pbar:
        for filename in filenames:
            img_id = filename.split(".")[0]
            segment = panoptic_per_image_id[img_id]["segments_info"]
            panoptic = np.array(Image.open(join(label_path, f"{img_id}.png")))
            img_pil = Image.open(join(img_path, filename))
            img = np.array(img_pil)
            img_height, img_width, _ = img.shape

            # Pad image to avoid cutting varying shapes due to boundary
            img_padded, pad_size = pad_image(
                img, pad_mode="edge", return_pad_size=True
            )

            # Crop the specified object
            for obj in segment:

                # Check if bounding box is cut off at the image boundary
                xmin, ymin, width, height = obj["bbox"]
                is_oob = (
                    (xmin == 0)
                    or (ymin == 0)
                    or ((xmin + width) >= img_width)
                    or ((ymin + height) >= img_height)
                )

                if (
                    obj["category_id"] != label
                    or obj["area"] < min_area
                    or is_oob
                ):
                    continue

                # Make sure that bounding box is square and add some padding to
                # avoid cutting into the sign
                size = max(width, height)
                xpad, ypad = int((size - width) / 2), int((size - height) / 2)
                extra_obj_pad = int(pad * size)
                size += 2 * extra_obj_pad
                xmin += pad_size - xpad - extra_obj_pad
                ymin += pad_size - ypad - extra_obj_pad
                xmax, ymax = xmin + size, ymin + size
                patch = img_padded[ymin:ymax, xmin:xmax]

                # Collect mask
                bool_mask = (panoptic[:, :, 0] == obj["id"]).astype(np.uint8)
                mask = np.stack(
                    [
                        bool_mask,
                    ]
                    * 4,
                    axis=-1,
                )
                mask *= np.array([0, 255, 0, 127], dtype=np.uint8)

                # # Run corner detection on mask
                # # blockSize: It is the size of neighbourhood considered for corner detection
                # block_size = int(size * 0.1)
                # # ksize: Aperture parameter of the Sobel derivative used
                # ksize = 3
                # corners = cv.cornerHarris(bool_mask, block_size, ksize, 0.04)
                # d_corners = cv.dilate(corners, None)
                # mask[corners > 0.5 * d_corners.max()] = [255, 0, 0, 255]

                # DEBUG
                # # print(corners.max())
                # print(corners.reshape(-1)[np.argsort(corners.reshape(-1))[::-1][:10]])
                # # print(np.sum(corners > 0.5 * corners.max()))
                # print(count_blobs(corners > 0.5 * corners.max()))

                contours, _ = cv.findContours(
                    bool_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
                )
                # mask = draw_from_contours(mask, contours, color=[0, 0, 255, 255])

                # Find convex hull to (1) combine multiple contours and/or
                # (2) fix some occlusion
                cat_contours = np.concatenate(contours, axis=0)
                hull = cv.convexHull(cat_contours, returnPoints=True)
                hull_color = (255, 255, 255, 255)
                mask = cv.drawContours(mask, [hull], -1, hull_color, 1)
                hull_mask = (mask == np.array(hull_color)).prod(-1)
                # mask = (1 - hull_mask) * mask + hull_mask * hull_draw

                # Fit polygon to remove some annotation errors and get vertices
                vertices = detect_polygon(hull)
                # mask = draw_from_contours(mask, vertices, color=[255, 255, 255, 255])
                print(vertices)

                # TODO: Check if matches with classifier prediction

                # TODO: if circle, fit ellipse instead
                hull_draw_points = np.stack(np.where(hull_mask), axis=1)[
                    :, ::-1
                ]
                ellipse = cv.fitEllipse(hull_draw_points)
                ellipse_color = (255, 0, 0, 255)
                mask = cv.ellipse(mask, ellipse, ellipse_color)

                # Get filled ellipse and compute ellipse error
                ellipse_mask = cv.ellipse(
                    np.zeros_like(bool_mask, dtype=np.float32),
                    ellipse,
                    (1,),
                    thickness=-1,
                )
                ellipse_error = (
                    np.abs(ellipse_mask - bool_mask.astype(np.float32)).sum()
                    / bool_mask.sum()
                )

                # DEBUG: Transform
                if len(vertices) == 4:
                    sign_height, sign_width = 100, 100
                    canonical = np.zeros((sign_height, sign_width, 3))
                    canonical_mask = np.zeros((sign_height, sign_width, 1))
                    adv_patch = np.random.rand(40, 40, 3)
                    # Histogram matching of the patch and the entire image
                    adv_patch_matched = match_histograms(adv_patch, img) / 255.0
                    canonical[20:60, 40:80, :] = adv_patch_matched
                    canonical_mask[20:60, 40:80, :] = 1
                    src = np.array(
                        [
                            [0, 0],
                            [0, sign_width - 1],
                            [sign_height - 1, sign_width - 1],
                            [sign_height - 1, 0],
                        ],
                        dtype=np.float32,
                    )
                    target = vertices[:, 0, :]
                    M = cv.getPerspectiveTransform(
                        src, target.astype(np.float32)
                    )
                    out = cv.warpPerspective(
                        canonical, M, (img_width, img_height)
                    )
                    out_mask = cv.warpPerspective(
                        canonical_mask, M, (img_width, img_height)
                    )
                    out_mask = (out_mask > 0.5).astype(np.float32)[:, :, None]
                    new_img = (1 - out_mask) * img / 255.0 + out_mask * out
                    # Mark vertices
                    vert = draw_from_contours(
                        np.zeros_like(new_img), vertices, color=[0, 255, 0]
                    )
                    vert = cv.dilate(vert, None) / 255.0
                    vert_mask = (vert.sum(-1) > 0).astype(np.float32)[
                        :, :, None
                    ]
                    new_img = (1 - vert_mask) * new_img + vert_mask * vert
                    new_img = pad_image(new_img, pad_mode="constant")[
                        ymin:ymax, xmin:xmax
                    ]
                    # save_image(torch.from_numpy(new_img).permute(2, 0, 1), 'test_warp.png')

                    # No histogram matching
                    canonical[20:60, 40:80, :] = adv_patch
                    out = cv.warpPerspective(
                        canonical, M, (img_width, img_height)
                    )
                    new_img2 = (1 - out_mask) * img / 255.0 + out_mask * out
                    new_img2 = (1 - vert_mask) * new_img2 + vert_mask * vert
                    new_img2 = pad_image(new_img2, pad_mode="constant")[
                        ymin:ymax, xmin:xmax
                    ]
                else:
                    new_img = np.zeros_like(patch)
                    new_img2 = np.zeros_like(patch)

                # emask = pad_image(ellipse_mask, pad_mode='constant')
                # save_image(torch.from_numpy(emask[ymin:ymax, xmin:xmax]), 'test.png')
                # save_image(torch.from_numpy(mask_patch[:, :, :3] / 255.).permute(2, 0, 1), 'test_mask.png')
                # save_image(torch.from_numpy(patch / 255.).permute(2, 0, 1), 'test_img.png')

                # DEBUG:
                if ellipse_error > 0.1:
                    pass
                    # vertices = get_corners(bool_mask)
                    # shape = get_shape_from_vertices(vertices)[0]
                    # print(shape)
                    # if shape != 'other':
                    #     get_box_vertices(vertices, shape)
                else:
                    print("found circle")
                    box = get_box_from_ellipse(ellipse).astype(np.int64)
                    mask = cv.drawContours(mask, [box], 0, ellipse_color, 1)

                    # emask = np.zeros_like(mask)
                    # # draw_from_contours(mask, box, color=[255, 255, 255, 255])
                    # emask[(box[:, 1], box[:, 0])] = [230, 230, 250, 255]
                    # # emask = cv.dilate(emask, None)
                    # mask = (emask == 0) * mask + (emask > 0) * emask
                    # mask_padded = pad_image(mask, pad_mode='constant')
                    # mask_patch = mask_padded[ymin:ymax, xmin:xmax]
                    # # mask_patch = mask_padded
                    # save_image(torch.from_numpy(mask_patch[:, :, :3] / 255.).permute(2, 0, 1), 'test_ellipse.png')

                    # TODO: Use edge detection instead of provided segmentation mask
                    # from skimage import feature, color, img_as_ubyte
                    # from skimage.transform import hough_ellipse
                    # from skimage.draw import ellipse_perimeter
                    # patch_gray = color.rgb2gray(patch)
                    # edges = feature.canny(patch_gray, sigma=1)
                    # tmp_edges = color.gray2rgb(img_as_ubyte(edges))
                    # save_image([img_numpy_to_torch(patch), img_numpy_to_torch(tmp_edges)], 'test_edge.png')
                    # result = hough_ellipse(
                    #     edges,
                    #     accuracy=20,
                    #     threshold=50,
                    #     min_size=int(max(width, height) * 0.8),     # Min of major axis
                    #     max_size=int(min(width, height) * 1.2),     # Max of minor axis
                    # )
                    # print(len(result))
                    # result.sort(order='accumulator')
                    # result = result[::-1]
                    # i = 0
                    # while i < len(result):
                    #     # Estimated parameters for the ellipse
                    #     best = list(result[i])
                    #     yc, xc, a, b = [int(round(x)) for x in best[1:5]]
                    #     orientation = best[5]
                    #     # Draw the ellipse on the original image
                    #     cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
                    #     try:
                    #         patch[cy, cx] = (0, 0, 255)
                    #     except IndexError:
                    #         i += 1
                    #         continue
                    #     # Draw the edge (white) and the resulting ellipse (red)
                    #     edges = color.gray2rgb(img_as_ubyte(edges))
                    #     edges[cy, cx] = (250, 0, 0)
                    #     save_image([img_numpy_to_torch(patch), img_numpy_to_torch(edges)], 'test_edge_new.png')

                # Mask should always be padded with zeros
                mask_padded = pad_image(mask, pad_mode="constant")
                mask_patch = mask_padded[ymin:ymax, xmin:xmax]

                final_img = [
                    patch / 255.0,
                    mask_patch[:, :, :3] / 255.0,
                    new_img2,
                    new_img,
                ]
                final_img = torch.from_numpy(
                    np.stack(final_img, axis=0)
                ).permute(0, 3, 1, 2)
                final_img = TF.resize(final_img, (128, 128))
                save_image(final_img, "test.png")

                import pdb

                pdb.set_trace()

                patches.append(patch)
                masks.append(mask_patch)
                ids.append(
                    {
                        "img_id": img_id,
                        "obj_id": obj["id"],
                        "num_vertices": len(vertices),
                        "ellipse_err": ellipse_error,
                    }
                )
                resized_patches.append(
                    TF.resize(
                        torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0),
                        (64, 64),
                    )
                )
                pbar.update(1)

            if len(patches) > max_num_imgs:
                break

    # Classify all patches
    resized_patches = torch.cat(resized_patches, dim=0)
    num_patches = len(patches)
    batch_size = 500
    num_batches = int(np.ceil(num_patches / batch_size))
    predicted_labels = torch.zeros(len(patches))
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            begin, end = i * batch_size, (i + 1) * batch_size
            logits = model(resized_patches[begin:end].to(device))
            outputs = logits.argmax(1)
            confidence = F.softmax(logits, dim=1)
            outputs[confidence.max(1)[0] < conf_thres] = num_classes - 1
            predicted_labels[begin:end] = outputs

    max_num_imgs_per_class = 100
    for i in range(num_classes):
        idx = np.where(i == predicted_labels)[0]
        if not idx.any():
            continue
        imgs = []
        for j in idx[:max_num_imgs_per_class]:
            mask = Image.fromarray(masks[j]).convert("RGBA")
            img = Image.fromarray(patches[j]).convert("RGBA")
            imgs.append(Image.alpha_composite(img, mask))
        show(imgs, num_cols=10)
        plt.savefig(f"predicted_corner_{i}.png", dpi=600)


def test_traffic_signs(
    model,
    data_dir,
    img_partition,
    max_num_imgs=None,
    min_area=0,
    conf_thres=0.8,
    pad_ratio=0.05,
    num_classes=6,
    device="cuda",
):

    img_path = join(data_dir, img_partition)
    label_path = join(data_dir, "annotations")
    shape_config_path = join(data_dir, "traffic_sign_dimension.csv")

    filenames = get_image_files(img_path)
    np.random.shuffle(filenames)
    max_num_imgs = len(filenames) if max_num_imgs is None else max_num_imgs

    grouped_labels = {}
    with open(shape_config_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            for group, selected_label in enumerate(SELECTED_SHAPES):
                if selected_label in line:
                    if group in grouped_labels:
                        grouped_labels[group].append(line.split(",")[0])
                    else:
                        grouped_labels[group] = [line.split(",")[0]]
    # data = pd.read_csv(shape_config_path)
    # for index, row in data.iterrows():
    #     if row['target'] in SELECTED_SHAPES:
    #         mtsd_label_to_shape_index[row['label']] = SELECTED_SHAPES.index(row['target'])

    mtsd_label_to_shape_index = {}
    for key in grouped_labels:
        for sign in grouped_labels[key]:
            mtsd_label_to_shape_index[sign] = key

    patches, resized_patches, masks, ids = [], [], [], []
    with tqdm(total=max_num_imgs) as pbar:
        for filename in filenames:
            img_id = filename.split(".")[0]
            anno = load_annotation(label_path, img_id)
            img = np.array(Image.open(join(img_path, filename)))
            img_height, img_width, _ = img.shape

            # Pad image to avoid cutting varying shapes due to boundary
            img_padded, pad_size = pad_image(
                img, pad_mode="edge", return_pad_size=True
            )

            for index, obj in enumerate(anno["objects"]):
                class_name = obj["label"]
                shape_index = mtsd_label_to_shape_index.get(
                    class_name, len(SELECTED_SHAPES)
                )
                xmin = obj["bbox"]["xmin"]
                ymin = obj["bbox"]["ymin"]
                xmax = obj["bbox"]["xmax"]
                ymax = obj["bbox"]["ymax"]
                width, height = xmax - xmin, ymax - ymin

                # Check if bounding box is cut off at the image boundary
                is_cut = (
                    (xmin == 0)
                    or (ymin == 0)
                    or ((xmin + width) >= img_width)
                    or ((ymin + height) >= img_height)
                )
                if obj["area"] < min_area or is_cut:
                    continue

                # Make sure that bounding box is square and add some padding to
                # avoid cutting into the sign
                size = max(width, height)
                xpad, ypad = int((size - width) / 2), int((size - height) / 2)
                extra_obj_pad = int(pad_ratio * size)
                size += 2 * extra_obj_pad
                xmin += pad_size - xpad
                ymin += pad_size - ypad
                xmax, ymax = xmin + size, ymin + size
                patch = img_padded[ymin:ymax, xmin:xmax]


def main():

    # Arguments
    min_area = 1000
    max_num_imgs = 200
    label_to_classify = 95  # Class id of traffic signs on Vistas
    conf_thres = 0.0
    use_ts_data = False
    data_dir = "/data/shared/mapillary_vistas/training/"
    # data_dir = '/data/shared/mtsd_v2_fully_annotated/'

    device = "cuda"
    # seed = 2021
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True

    # Create model
    # mean = [0.3891, 0.3978, 0.3728]
    # std = [0.1688, 0.1622, 0.1601]
    # For ImageNet scaling
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean, std)

    base = models.resnet18(pretrained=False)
    base.fc = nn.Linear(512, 6)

    model_path = "/home/nab_126/adv-patch-bench/model_weights/resnet18.pth"
    if os.path.exists(model_path):
        print("Loading model weights...")
        base.load_state_dict(torch.load(model_path))
    else:
        raise ValueError("Model weight not found!")

    model = nn.Sequential(normalize, base).to(device).eval()

    # with open('/data/shared/mapillary_vistas/config_v2.0.json') as config_file:
    #     config = json.load(config_file)

    if use_ts_data:
        test_traffic_signs(
            model,
            data_dir,
            "val",
            max_num_imgs=max_num_imgs,
            min_area=min_area,
            num_classes=6,
            device=device,
        )
    else:
        # Read in panoptic file
        panoptic_json_path = f"{data_dir}/v2.0/panoptic/panoptic_2020.json"
        with open(panoptic_json_path) as panoptic_file:
            panoptic = json.load(panoptic_file)

        # Convert annotation infos to image_id indexed dictionary
        panoptic_per_image_id = {}
        for annotation in panoptic["annotations"]:
            panoptic_per_image_id[annotation["image_id"]] = annotation

        # Convert category infos to category_id indexed dictionary
        panoptic_category_per_id = {}
        for category in panoptic["categories"]:
            panoptic_category_per_id[category["id"]] = category

        show_img_patch(
            model,
            label_to_classify,
            panoptic_per_image_id,
            data_dir,
            min_area=min_area,
            max_num_imgs=max_num_imgs,
            conf_thres=conf_thres,
            device=device,
        )


if __name__ == "__main__":
    main()
