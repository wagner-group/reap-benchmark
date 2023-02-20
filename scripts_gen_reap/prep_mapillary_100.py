"""Original script for classifying Mapillary traffic signs by shapes."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.backends import cudnn
from torchvision.transforms import InterpolationMode
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
    img_path = os.path.join(DATA_DIR, "images")

    filenames = [
        filename
        for filename in os.listdir(img_path)
        if os.path.isfile(os.path.join(img_path, filename))
    ]
    np.random.shuffle(filenames)

    ids, bboxes, predicted_labels = [], [], []
    filename_to_idx = {}
    obj_idx = 0
    begin = 0

    for filename in tqdm(filenames):
        # Read each image file and crop the traffic signs
        img_id = filename.split(".")[0]
        segment = panoptic_per_image_id[img_id]["segments_info"]
        img_pil = Image.open(os.path.join(img_path, filename))
        img = np.array(img_pil)
        img_height, img_width, _ = img.shape

        # Pad image to avoid cutting varying shapes due to boundary
        img_padded, pad_size = pad_image(
            img, pad_mode="edge", return_pad_size=True
        )
        filename_to_idx[img_id] = []
        resized_patches = []

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
            xmin = max(xmin + pad_size - xpad, 0)
            ymin = max(ymin + pad_size - ypad, 0)
            xmax, ymax = xmin + size, ymin + size
            cropped_patch = img_padded[ymin:ymax, xmin:xmax]
            if any(s == 0 for s in cropped_patch.shape):
                print("WARNING: Cropped patch has zero dimension!")
                continue
            filename_to_idx[img_id].append(obj_idx)
            obj_idx += 1
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
                    interpolation=InterpolationMode.BICUBIC,
                )
            )
            bboxes.append(
                [xmin, ymin, xmin + width, ymin + height, img_width, img_height]
            )
            # DEBUG: Visualize cropped signs
            # from torchvision.utils import save_image
            # import pdb
            # pdb.set_trace()
            # save_image(resized_patches[-1].float() / 255, "tmp.png")

        if not resized_patches:
            continue

        # Classify reseized patches
        resized_patches = torch.cat(resized_patches, dim=0)
        with torch.no_grad():
            logits = model(resized_patches.to(device))
            outputs = logits.argmax(1)
            confidence = F.softmax(logits, dim=1)
            # If confidene is below threshold, set label to background
            outputs[confidence.max(1)[0] < CONF_THRES] = CLF_NUM_CLASSES - 1
            predicted_labels.append(outputs.cpu())
        begin += len(resized_patches)

        if begin > MAX_NUM_IMGS:
            break

    predicted_labels = torch.cat(predicted_labels, dim=0)
    assert len(predicted_labels) == len(ids)
    return predicted_labels, ids, filename_to_idx, bboxes


def main():
    """Main function."""
    device = "cuda"
    seed = 2021
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Load trained model
    model, _, _ = build_classifier(args)

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
    predicted_labels, ids, filename_to_idx, bboxes = classify(
        model, panoptic_per_image_id, device=device
    )

    # Merge predicted labels with current REAP annotations
    anno = load_annotation_df(BASE_REAP_ANNO_PATH, keep_others=True)
    new_col = f"{DATASET_MODIFIER}_label"
    anno[new_col] = "other"

    num_wrong = 0
    for anno_id, label in zip(ids, predicted_labels):
        img_id, obj_id = anno_id["img_id"], anno_id["obj_id"]
        new_label = LABEL_LIST[f"mapillary_{DATASET_MODIFIER}"][int(label)]
        if DATASET_MODIFIER == "100":
            new_shape = MTSD100_TO_SHAPE[new_label]
        else:
            raise NotImplementedError(
                f"Invalid DATASET_MODIFIER: {DATASET_MODIFIER}!"
            )

        cond = (anno["object_id"] == obj_id) & (
            anno["filename"] == img_id + ".jpg"
        )

        # If new predicted shape is different from the original shape which is
        # more trustworthy, then set the new label to background
        if anno.loc[cond, "final_shape"].empty:
            continue
        if new_shape != anno.loc[cond, "final_shape"].item():
            num_wrong += 1
            print(
                f"=> {num_wrong} total wrong predictions ({new_shape} "
                f"[{new_label}] vs {anno.loc[cond, 'final_shape'].item()})!"
            )
            new_label = "other"

        anno.loc[cond, new_col] = new_label

    # Save new annotations
    anno.to_csv(BASE_REAP_ANNO_PATH, index=False)

    # Create dir for new modified dataset
    label_path = os.path.join(DATA_DIR, f"mapillary_{DATASET_MODIFIER}")
    os.makedirs(label_path, exist_ok=True)

    for img_id, obj_idx in tqdm(filename_to_idx.items()):
        # Skip image with no valid objects
        if len(obj_idx) == 0:
            continue

        obj_target = ""
        for idx in obj_idx:
            # Write label in Detectron2 format
            class_label = int(predicted_labels[idx].item())
            obj_id = ids[idx]["obj_id"]
            assert img_id == ids[idx]["img_id"], (
                "Image ID mismatch! Sanity check failed!"
            )
            xmin, ymin, xmax, ymax, img_width, img_height = bboxes[idx]
            obj_target += (
                f"{class_label:d},{xmin},{ymin},{xmax},{ymax},{img_width},"
                f"{img_height},{obj_id}\n"
            )

        if obj_target:
            save_label_path = os.path.join(label_path, img_id + ".txt")
            with open(save_label_path, "w", encoding="utf-8") as file:
                file.write(obj_target)


if __name__ == "__main__":
    BASE_PATH = os.path.expanduser("~/reap-benchmark/")

    # Lazy arguments (classifier)
    MODEL_PATH = f"{BASE_PATH}/results/classifier_mtsd-100/checkpoint_best.pt"
    ARCH = "convnext_small_in22k"
    DATASET_MODIFIER = "100"
    CLF_NUM_CLASSES = 100
    CLF_IMG_SIZE = 224
    CLF_BATCH_SIZE = 32

    # Lazy arguments (data)
    BASE_REAP_ANNO_PATH = f"{BASE_PATH}/reap_annotations.csv"
    SPLIT = "training"  # "training" or "validation"
    DATA_DIR = os.path.expanduser(f"~/data/mapillary_vistas/{SPLIT}/")
    MIN_AREA = 1000  # Minimum area of traffic signs to consider in pixels
    MAX_NUM_IMGS = 1e9  # Set to small number for debugging
    LABEL_TO_CLF = 95  # Class id of traffic signs on Vistas
    # If confidence score is below this threshold, set label to background
    CONF_THRES = 0.0

    # Hacky way of loading hyperparameters and metadata
    LABEL_LIST: dict[str, list[str]] = {}
    MTSD100_TO_SHAPE: dict[str, str] = {}
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
