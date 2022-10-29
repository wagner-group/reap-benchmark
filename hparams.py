"""Define common global variables.

The intention of this file is to centralize commonly used hyperparameters so it
is easy to modify. However, this is not an ideal way for passing around
metadata. We try to use these parameters minimally.

TODO(NewDataset): Use config file and dataset object to load and hold metadata.
"""

from typing import Any, Dict

# Set paths
PATH_MAPILLARY_ANNO = {
    "train": "./reap_annotations.csv",
    "val": "./reap_annotations.csv",
    "combined": "./reap_annotations.csv",
}

DEFAULT_DATA_PATHS = {
    "mtsd": "~/data/mtsd_v2_fully_annotated/",
    "mapillary": "~/data/mapillary_vistas/",
}
DEFAULT_DATA_PATHS["reap"] = DEFAULT_DATA_PATHS["mapillary"]
DEFAULT_DATA_PATHS["synthetic"] = DEFAULT_DATA_PATHS["mapillary"]
DEFAULT_SYN_OBJ_DIR = "./attack_assets/"

DEFAULT_PATH_MTSD_LABEL = "./mtsd_label_metadata.csv"
PATH_SIMILAR_FILES = "./similar_files_df.csv"
DEFAULT_PATH_BG_FILE_NAMES = "./bg_txt_files/"
DEFAULT_PATH_DEBUG_PATCH = f"{DEFAULT_SYN_OBJ_DIR}/debug.png"

# TODO: move to args in the future
SAVE_DIR_YOLO = "./runs/val/"

# Allowed interpolation methods
INTERPS = ("nearest", "bilinear", "bicubic")

# =========================================================================== #

# Available dataset and class labels
DATASETS = (
    "mtsd_orig",
    "mtsd_no_color",
    "mtsd_color",
    "mapillary_no_color",
    "mapillary_color",
    "reap",
    "synthetic",
)

# Traffic sign classes with colors
TS_COLOR_DICT = {
    "circle-750.0": ["white", "blue", "red"],  # (1) white+red, (2) blue+white
    "triangle-900.0": ["white", "yellow"],  # (1) white, (2) yellow
    "triangle_inverted-1220.0": [],  # (1) white+red
    "diamond-600.0": [],  # (1) white+yellow
    "diamond-915.0": [],  # (1) yellow
    "square-600.0": [],  # (1) blue
    "rect-458.0-610.0": [
        "white",
        "other",
    ],  # (1) chevron (also multi-color), (2) white
    "rect-762.0-915.0": [],  # (1) white
    "rect-915.0-1220.0": [],  # (1) white
    "pentagon-915.0": [],  # (1) yellow
    "octagon-915.0": [],  # (1) red
    "other-0.0-0.0": [],
}

# Generate dictionary of traffic sign class offset
TS_COLOR_OFFSET_DICT = {}
idx = 0
for k in TS_COLOR_DICT:
    TS_COLOR_OFFSET_DICT[k] = idx
    idx += max(1, len(TS_COLOR_DICT[k]))

# Generate dictionary of traffic sign class: name -> idx
TS_COLOR_LABEL_DICT = {}
idx = 0
for k in TS_COLOR_DICT:
    if len(TS_COLOR_DICT[k]) == 0:
        TS_COLOR_LABEL_DICT[f"{k}-none"] = idx
        idx += 1
    else:
        for color in TS_COLOR_DICT[k]:
            TS_COLOR_LABEL_DICT[f"{k}-{color}"] = idx
            idx += 1

# Make sure that ordering is correct
TS_COLOR_LABEL_LIST = list(TS_COLOR_LABEL_DICT.keys())
TS_NO_COLOR_LABEL_LIST = list(TS_COLOR_DICT.keys())
LABEL_LIST = {
    "mtsd_color": TS_COLOR_LABEL_LIST,
    "mapillary_color": TS_COLOR_LABEL_LIST,
    "mtsd_no_color": TS_NO_COLOR_LABEL_LIST,
    "mapillary_no_color": TS_NO_COLOR_LABEL_LIST,
}
LABEL_LIST["reap"] = LABEL_LIST["mapillary_no_color"]
LABEL_LIST["synthetic"] = LABEL_LIST["mapillary_no_color"]

# Get list of shape (no size, no color)
TS_SHAPE_LIST = list(
    set([shape.split("-")[0] for shape in TS_NO_COLOR_LABEL_LIST])
)

# =========================================================================== #

# Number of classes in each dataset
NUM_CLASSES = {
    "mtsd_orig": 401,
    "mtsd_no_color": len(TS_NO_COLOR_LABEL_LIST),
    "mtsd_color": len(TS_COLOR_LABEL_LIST),
    "mapillary_no_color": len(TS_NO_COLOR_LABEL_LIST),
    "mapillary_color": len(TS_COLOR_LABEL_LIST),
}
NUM_CLASSES["reap"] = NUM_CLASSES["mapillary_no_color"]
NUM_CLASSES["synthetic"] = NUM_CLASSES["mapillary_no_color"]

# =========================================================================== #

# Configure dimension
_MPL_NO_COLOR_SIZE_MM = {
    "circle-750.0": (750.0, 750.0),
    "triangle-900.0": (789.0, 900.0),
    "triangle_inverted-1220.0": (1072.3, 1220.0),
    "diamond-600.0": (600.0, 600.0),
    "diamond-915.0": (915.0, 915.0),
    "square-600.0": (600.0, 600.0),
    "rect-458.0-610.0": (610.0, 458.0),
    "rect-762.0-915.0": (915.0, 762.0),
    "rect-915.0-1220.0": (1220.0, 915.0),
    "pentagon-915.0": (915.0, 915.0),
    "octagon-915.0": (915.0, 915.0),
}
_MPL_NO_COLOR_SIZE_MM = list(_MPL_NO_COLOR_SIZE_MM.values())

# Geometric shape of objects
# This is straightforward for our traffic sign classes, but to extend to other
# dataset in general, we need a mapping from class names to shapes.
_MPL_NO_COLOR_SHAPE = {
    "circle-750.0": "circle",
    "triangle-900.0": "triangle",
    "triangle_inverted-1220.0": "triangle_inverted",
    "diamond-600.0": "diamond",
    "diamond-915.0": "diamond",
    "square-600.0": "square",
    "rect-458.0-610.0": "rect",
    "rect-762.0-915.0": "rect",
    "rect-915.0-1220.0": "rect",
    "pentagon-915.0": "pentagon",
    "octagon-915.0": "octagon",
}
_MPL_NO_COLOR_SHAPE = list(_MPL_NO_COLOR_SHAPE.values())

# Height-width ratio
_MPL_NO_COLOR_RATIO = [s[0] / s[1] for s in _MPL_NO_COLOR_SIZE_MM]

OBJ_DIM_DICT: Dict[str, Dict[str, Any]] = {
    "mapillary_no_color": {
        "size_mm": _MPL_NO_COLOR_SIZE_MM,
        "hw_ratio": _MPL_NO_COLOR_RATIO,
        "shape": _MPL_NO_COLOR_SHAPE,
    }
}
OBJ_DIM_DICT["reap"] = OBJ_DIM_DICT["mapillary_no_color"]
OBJ_DIM_DICT["synthetic"] = OBJ_DIM_DICT["mapillary_no_color"]

# =========================================================================== #

# TODO: DEPRECATED

# MTSD_VAL_LABEL_COUNTS_DICT = {
#     "circle-750.0": 2999,
#     "triangle-900.0": 711,
#     "triangle_inverted-1220.0": 347,
#     "diamond-600.0": 176,
#     "diamond-915.0": 1278,
#     "square-600.0": 287,
#     "rect-458.0-610.0": 585,
#     "rect-762.0-915.0": 117,
#     "rect-915.0-1220.0": 135,
#     "pentagon-915.0": 30,
#     "octagon-915.0": 181,
#     "other-0.0-0.0": 19241,
# }
# MTSD_VAL_TOTAL_LABEL_COUNTS = sum(MTSD_VAL_LABEL_COUNTS_DICT.values())

# MAPILLARY_LABEL_COUNTS_DICT = {
#     "circle-750.0": 18144,
#     "triangle-900.0": 1473,
#     "triangle_inverted-1220.0": 1961,
#     "diamond-600.0": 1107,
#     "diamond-915.0": 3539,
#     "square-600.0": 1898,
#     "rect-458.0-610.0": 1580,
#     "rect-762.0-915.0": 839,
#     "rect-915.0-1220.0": 638,
#     "pentagon-915.0": 204,
#     "octagon-915.0": 1001,
#     "other-0.0-0.0": 60104,
# }
# MAPILLARY_TOTAL_LABEL_COUNTS = sum(MAPILLARY_LABEL_COUNTS_DICT.values())

# Counts of images where sign is present in
MAPILLARY_IMG_COUNTS_DICT = {
    "circle-750.0": 5325,
    "triangle-900.0": 548,
    "triangle_inverted-1220.0": 706,
    "diamond-600.0": 293,
    "diamond-915.0": 1195,
    "square-600.0": 729,
    "rect-458.0-610.0": 490,
    "rect-762.0-915.0": 401,
    "rect-915.0-1220.0": 333,
    "pentagon-915.0": 116,
    "octagon-915.0": 564,
    "other-0.0-0.0": 0,
}

# Compute results
# ANNO_LABEL_COUNTS_DICT = {
#     "circle-750.0": 7971,
#     "triangle-900.0": 636,
#     "triangle_inverted-1220.0": 824,
#     "diamond-600.0": 317,
#     "diamond-915.0": 1435,
#     "square-600.0": 1075,
#     "rect-458.0-610.0": 715,
#     "rect-762.0-915.0": 544,
#     "rect-915.0-1220.0": 361,
#     "pentagon-915.0": 133,
#     "octagon-915.0": 637,
# }
# ANNO_NOBG_LABEL_COUNTS_DICT = {
#     "circle-750.0": 7902,
#     "triangle-900.0": 578,
#     "triangle_inverted-1220.0": 764,
#     "diamond-600.0": 263,
#     "diamond-915.0": 1376,
#     "square-600.0": 997,
#     "rect-458.0-610.0": 646,
#     "rect-762.0-915.0": 482,
#     "rect-915.0-1220.0": 308,
#     "pentagon-915.0": 78,
#     "octagon-915.0": 585,
# }
# ANNO_NOBG_LABEL_COUNTS_DICT_200 = {
#     "circle-750.0": 7669,
#     "triangle-900.0": 405,
#     "triangle_inverted-1220.0": 584,
#     "diamond-600.0": 0,
#     "diamond-915.0": 1201,
#     "square-600.0": 788,
#     "rect-458.0-610.0": 412,
#     "rect-762.0-915.0": 275,
#     "rect-915.0-1220.0": 150,
#     "pentagon-915.0": 0,
#     "octagon-915.0": 405,
# }
