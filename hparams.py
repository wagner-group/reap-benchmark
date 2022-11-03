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
    "circle": ["white", "blue", "red"],  # (1) white+red, (2) blue+white
    "triangle": ["white", "yellow"],  # (1) white, (2) yellow
    "up-triangle": [],  # (1) white+red
    "diamond-s": [],  # (1) white+yellow
    "diamond-l": [],  # (1) yellow
    "square": [],  # (1) blue
    "rect-s": [
        "white",
        "other",
    ],  # (1) chevron (also multi-color), (2) white
    "rect-m": [],  # (1) white
    "rect-l": [],  # (1) white
    "pentagon": [],  # (1) yellow
    "octagon": [],  # (1) red
    "other": [],
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
    "circle": (750.0, 750.0),
    "triangle": (789.0, 900.0),
    "up-triangle": (1072.3, 1220.0),
    "diamond-s": (600.0, 600.0),
    "diamond-l": (915.0, 915.0),
    "square": (600.0, 600.0),
    "rect-s": (610.0, 458.0),
    "rect-m": (915.0, 762.0),
    "rect-l": (1220.0, 915.0),
    "pentagon": (915.0, 915.0),
    "octagon": (915.0, 915.0),
}
_MPL_NO_COLOR_SIZE_MM = list(_MPL_NO_COLOR_SIZE_MM.values())

# Geometric shape of objects
# This is straightforward for our traffic sign classes, but to extend to other
# dataset in general, we need a mapping from class names to shapes.
_MPL_NO_COLOR_SHAPE = {
    "circle": "circle",
    "triangle": "triangle",
    "up-triangle": "triangle_inverted",
    "diamond-s": "diamond",
    "diamond-l": "diamond",
    "square": "square",
    "rect-s": "rect",
    "rect-m": "rect",
    "rect-l": "rect",
    "pentagon": "pentagon",
    "octagon": "octagon",
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
#     "circle": 2999,
#     "triangle": 711,
#     "up-triangle": 347,
#     "diamond-s": 176,
#     "diamond-l": 1278,
#     "square": 287,
#     "rect-s": 585,
#     "rect-m": 117,
#     "rect-l": 135,
#     "pentagon": 30,
#     "octagon": 181,
#     "other": 19241,
# }
# MTSD_VAL_TOTAL_LABEL_COUNTS = sum(MTSD_VAL_LABEL_COUNTS_DICT.values())

# MAPILLARY_LABEL_COUNTS_DICT = {
#     "circle": 18144,
#     "triangle": 1473,
#     "up-triangle": 1961,
#     "diamond-s": 1107,
#     "diamond-l": 3539,
#     "square": 1898,
#     "rect-s": 1580,
#     "rect-m": 839,
#     "rect-l": 638,
#     "pentagon": 204,
#     "octagon": 1001,
#     "other": 60104,
# }
# MAPILLARY_TOTAL_LABEL_COUNTS = sum(MAPILLARY_LABEL_COUNTS_DICT.values())

# Counts of images where sign is present in
MAPILLARY_IMG_COUNTS_DICT = {
    "circle": 5325,
    "triangle": 548,
    "up-triangle": 706,
    "diamond-s": 293,
    "diamond-l": 1195,
    "square": 729,
    "rect-s": 490,
    "rect-m": 401,
    "rect-l": 333,
    "pentagon": 116,
    "octagon": 564,
    "other": 0,
}

# Compute results
# ANNO_LABEL_COUNTS_DICT = {
#     "circle": 7971,
#     "triangle": 636,
#     "up-triangle": 824,
#     "diamond-s": 317,
#     "diamond-l": 1435,
#     "square": 1075,
#     "rect-s": 715,
#     "rect-m": 544,
#     "rect-l": 361,
#     "pentagon": 133,
#     "octagon": 637,
# }
# ANNO_NOBG_LABEL_COUNTS_DICT = {
#     "circle": 7902,
#     "triangle": 578,
#     "up-triangle": 764,
#     "diamond-s": 263,
#     "diamond-l": 1376,
#     "square": 997,
#     "rect-s": 646,
#     "rect-m": 482,
#     "rect-l": 308,
#     "pentagon": 78,
#     "octagon": 585,
# }
# ANNO_NOBG_LABEL_COUNTS_DICT_200 = {
#     "circle": 7669,
#     "triangle": 405,
#     "up-triangle": 584,
#     "diamond-s": 0,
#     "diamond-l": 1201,
#     "square": 788,
#     "rect-s": 412,
#     "rect-m": 275,
#     "rect-l": 150,
#     "pentagon": 0,
#     "octagon": 405,
# }
