"""Script for generating realism test samples."""

import pathlib
import pickle

import numpy as np
import skimage
from torchvision import transforms

from adv_patch_bench.attacks.patch_mask_util import gen_patch_mask
from hparams import DATASET_METADATA, TS_COLOR_DICT


def _to_rgba(img):
    if img.shape[-1] == 3:
        return np.concatenate(
            [img, np.ones(img.shape[:2] + (1,)) * 255], axis=2
        )
    return img


# Desired patch size
patch_size = (1, 10, 10)

# Path to synthetic images and a sample of adversarial patch
img_path = pathlib.Path("attack_assets/synthetic/")
# patch = skimage.io.imread("attack_assets/example_patch.png")
# patch = _to_rgba(patch)
patch_path = pathlib.Path(
    "detectron_output/synthetic-load-64-15-1-0.4-0-0-pd64-bg50-augimg1-rp2_1e-05_0_1_1000_adam_0.01_False"
)
save_path = pathlib.Path("attack_assets/realism_test")
save_path.mkdir(exist_ok=True)
patch_size = (patch_size[0], patch_size[1] * 25.4, patch_size[2] * 25.4)
hw_ratio_dict = DATASET_METADATA["mapillary_no_color"]["hw_ratio"]
obj_size_dict = DATASET_METADATA["mapillary_no_color"]["size_mm"]
class_list = list(TS_COLOR_DICT.keys())

for path in img_path.glob("*.png"):
    print(path)
    image = skimage.io.imread(str(path))
    shape = str(path.name).split(".", maxsplit=1)[0]
    with open(str(patch_path / shape / "adv_patch.pkl"), "rb") as file:
        patch, mask = pickle.load(file)
    resize = transforms.Resize(
        image.shape[:2], interpolation=transforms.InterpolationMode.NEAREST
    )
    patch: np.ndarray = resize(patch).permute(1, 2, 0).numpy()
    mask = resize(mask).permute(1, 2, 0).numpy()
    # Lift patch so it's not cut-off
    hh, ww, _ = np.where(mask)
    h_min, h_max = hh.min(), hh.max() + 1
    w_min, w_max = ww.min(), ww.max() + 1
    if shape == "diamond-s":
        factor = 0.2
    elif shape == "diamond-l":
        factor = 0.15
    elif shape in ("circle", "up-triangle"):
        factor = 0.1
    else:
        factor = 0.0
    shift = int(h_max * factor)
    # patch[h_min] = 0
    # patch[h_max - 1] = 0
    # patch[:, w_min] = 0
    # patch[:, w_max - 1] = 0
    patch *= 255
    print(image.shape, patch.shape)
    patch = _to_rgba(patch)
    image = _to_rgba(image)
    # image = image * (1 - mask) + patch * mask * image[:, :, 3:] / 255
    # image[h_min - shift : h_max - shift, w_min:w_max, :3] = patch[
    #     h_min:h_max, w_min:w_max, :3
    # ]
    image = image.astype(np.long)

    # obj_size = obj_size_dict[class_list.index(shape)]
    # mask = gen_patch_mask(
    #     patch_size, image.shape[:2], obj_size, shift_height_mm=0.0
    # )
    # _, hh, ww = np.where(mask.numpy())
    # h_min, h_max = hh.min(), hh.max() + 1
    # w_min, w_max = ww.min(), ww.max() + 1
    # patch_resized = skimage.transform.resize(
    #     patch,
    #     (h_max - h_min, w_max - w_min),
    #     order=0,
    #     anti_aliasing=True,
    # )
    # image[h_min:h_max, w_min:w_max] = patch_resized
    skimage.io.imsave(str(save_path / f"{shape}_orig.png"), image)


# hh, ww = np.where(patch.mean(-1) < 255)
# h_min, h_max = hh.min(), hh.max() + 1
# w_min, w_max = ww.min(), ww.max() + 1
# patch = patch[h_min:h_max, w_min:w_max]
# patch[0] = 0
# patch[-1] = 0
# patch[:, 0] = 0
# patch[:, -1] = 0

# skimage.io.imsave("attack_assets/adversarial_patch_cropped.png", patch)

print("Finished.")
