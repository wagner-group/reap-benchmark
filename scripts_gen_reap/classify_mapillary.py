"""Original script for classifying Mapillary traffic signs by shapes."""

import argparse
import json
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm

from adv_patch_bench.dataloaders.reap_util import load_annotation_df
from adv_patch_bench.models import build_classifier
from adv_patch_bench.utils import pad_image


def classify(
    model,
    panoptic_per_image_id,
    device: str = "cuda",
):
    """Classify objects to get pseudo-labels."""
    img_path = join(DATA_DIR, "images")

    filenames = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    np.random.shuffle(filenames)

    objs, resized_patches, ids = [], [], []
    for filename in tqdm(filenames):
        # Read each image file and crop the traffic signs
        img_id = filename.split(".")[0]
        segment = panoptic_per_image_id[img_id]["segments_info"]
        img_pil = Image.open(join(img_path, filename))
        img = np.array(img_pil)
        img_height, img_width, _ = img.shape

        # Pad image to avoid cutting varying shapes due to boundary
        img_padded, pad_size = pad_image(
            img, pad_mode="edge", return_pad_size=True
        )

        # Crop the specified object
        for cropped_obj in segment:

            # Check if bounding box is cut off at the image boundary
            xmin, ymin, width, height = cropped_obj["bbox"]
            is_oob = (
                (xmin == 0)
                or (ymin == 0)
                or ((xmin + width) >= img_width)
                or ((ymin + height) >= img_height)
            )

            if (
                cropped_obj["category_id"] != LABEL_TO_CLF
                or cropped_obj["area"] < MIN_AREA
                or is_oob
            ):
                continue

            # Make sure that bounding box is square and add some padding to
            # avoid cutting into the sign
            size = max(width, height)
            xpad, ypad = int((size - width) / 2), int((size - height) / 2)
            xmin += pad_size - xpad
            ymin += pad_size - ypad
            xmax, ymax = xmin + size, ymin + size
            cropped_patch = img_padded[ymin:ymax, xmin:xmax]
            objs.append(cropped_patch)
            ids.append(
                {
                    "img_id": img_id,
                    "obj_id": cropped_obj["id"],
                }
            )
            cropped_patch = torch.from_numpy(cropped_patch).permute(2, 0, 1)
            cropped_patch.unsqueeze_(0)
            resized_patches.append(
                TF.resize(
                    cropped_patch,
                    (CLF_IMG_SIZE, CLF_IMG_SIZE),
                    interpolation=Image.BICUBIC,
                )
            )

        if len(objs) > MAX_NUM_IMGS:
            break

    # Classify all patches
    resized_patches = torch.cat(resized_patches, dim=0)
    num_patches = len(objs)
    num_batches = int(np.ceil(num_patches / CLF_BATCH_SIZE))
    predicted_labels = torch.zeros(len(objs))
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            begin, end = i * CLF_BATCH_SIZE, (i + 1) * CLF_BATCH_SIZE
            logits = model(resized_patches[begin:end].to(device))
            outputs = logits.argmax(1)
            confidence = F.softmax(logits, dim=1)
            # If confidene is below threshold, set label to background
            outputs[confidence.max(1)[0] < CONF_THRES] = CLF_NUM_CLASSES - 1
            predicted_labels[begin:end] = outputs

    assert len(predicted_labels) == len(ids)
    return predicted_labels, ids


def main():
    """Main function."""
    device = "cuda"
    seed = 2021
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Load trained model
    model, _, _ = build_classifier(args)

    # with open('/data/shared/mapillary_vistas/config_v2.0.json') as config_file:
    #     config = json.load(config_file)

    # Read in panoptic file
    panoptic_json_path = f"{DATA_DIR}/v2.0/panoptic/panoptic_2020.json"
    with open(panoptic_json_path, "r", encoding="utf-8") as panoptic_file:
        panoptic = json.load(panoptic_file)

    # Convert annotation infos to image_id indexed dictionary
    panoptic_per_image_id = {}
    for annotation in panoptic["annotations"]:
        panoptic_per_image_id[annotation["image_id"]] = annotation

    # Convert category infos to category_id indexed dictionary
    panoptic_category_per_id = {}
    for category in panoptic["categories"]:
        panoptic_category_per_id[category["id"]] = category

    # Get predicted labels from model
    predicted_labels, ids = classify(
        model, panoptic_per_image_id, device=device
    )

    # Merge predicted labels with current REAP annotations
    anno = load_annotation_df(BASE_REAP_ANNO_PATH, keep_others=True)
    new_col = f"{DATASET_NAME}_label"
    anno[new_col] = "other"

    for anno_id, label in zip(ids, predicted_labels):
        img_id, obj_id = anno_id["img_id"], anno_id["obj_id"]
        anno.loc[
            (anno["object_id"] == obj_id)
            & (anno["filename"] == img_id + ".jpg"),
            new_col,
        ] = LABEL_LIST[DATASET_NAME][int(label)]

    # Save new annotations
    anno.to_csv(BASE_REAP_ANNO_PATH, index=False)


if __name__ == "__main__":
    BASE_PATH = os.path.expanduser("~/reap-benchmark/")

    # Lazy arguments
    MODEL_PATH = f"{BASE_PATH}/results/classifier_mtsd-100/checkpoint_best.pt"
    ARCH = "convnext_small_in22k"
    DATASET_NAME = "reap_100"
    CLF_NUM_CLASSES = 100
    CLF_IMG_SIZE = 224
    CLF_BATCH_SIZE = 500
    BASE_REAP_ANNO_PATH = f"{BASE_PATH}/reap_annotations.csv"
    DATA_DIR = os.path.expanduser("~/data/mapillary_vistas/training/")
    MIN_AREA = 1000  # Minimum area of traffic signs to consider in pixels
    MAX_NUM_IMGS = 1e9  # Set to small number for debugging
    LABEL_TO_CLF = 95  # Class id of traffic signs on Vistas
    # If confidence score is below this threshold, set label to background
    CONF_THRES = 0.0

    # Hacky way of loading hyperparameters and metadata
    LABEL_LIST: dict[str, list[str]] = {}
    with open(f"{BASE_PATH}/hparams.py", "r", encoding="utf-8") as metadata:
        source = metadata.read()
    exec(source)  # pylint: disable=exec-used

    parser = argparse.ArgumentParser(
        description="Train/test traffic sign classifier.", add_help=False
    )
    args = parser.parse_args()
    args.arch = ARCH
    args.num_classes = CLF_NUM_CLASSES
    args.resume = MODEL_PATH

    # Dummy arguments
    args.dataset = "mtsd"
    args.distributed = False
    args.wd = 1e-4
    args.lr = 1e-4
    args.gpu = 0
    args.pretrained = False
    args.momentum = 1e-4
    args.betas = (0.99, 0.999)
    args.optim = "sgd"
    args.full_precision = True

    main()
