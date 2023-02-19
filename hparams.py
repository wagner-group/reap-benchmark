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
PATH_DUPLICATE_FILES = "./scripts_gen_reap/similar_files_df.csv"
DEFAULT_PATH_BG_FILE_NAMES = "./bg_txt_files/"
DEFAULT_PATH_DEBUG_PATCH = f"{DEFAULT_SYN_OBJ_DIR}/debug.png"

# DEPRECATED: move to args in the future
SAVE_DIR_YOLO = "./runs/val/"

# Allowed interpolation methods
INTERPS = ("nearest", "bilinear", "bicubic")

# TODO(enhancement): Unify relighting transform API
RELIGHT_METHODS = [
    "color_transfer",
    "color_transfer_hsv-sv",
    "color_transfer_lab-l",
    "polynomial",
    "polynomial_max",
    "polynomial_mean",
    "polynomial_hsv-sv",
    "polynomial_lab-l",
    "percentile",
]

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
for shape, colors in TS_COLOR_DICT.items():
    TS_COLOR_OFFSET_DICT[shape] = idx
    idx += max(1, len(colors))

# Generate dictionary of traffic sign class: name -> idx
TS_COLOR_LABEL_DICT = {}
idx = 0
for shape, colors in TS_COLOR_DICT.items():
    if len(colors) == 0:
        TS_COLOR_LABEL_DICT[f"{shape}-none"] = idx
        idx += 1
    else:
        for color in colors:
            TS_COLOR_LABEL_DICT[f"{shape}-{color}"] = idx
            idx += 1

# Make sure that ordering is correct
TS_COLOR_LABEL_LIST = list(TS_COLOR_LABEL_DICT.keys())
TS_NO_COLOR_LABEL_LIST = list(TS_COLOR_DICT.keys())

# Metadata for the MTSD-100/REAP-100 dataset
MTSD100_TO_SHAPE = {
    "complementary--chevron-left--g1": "rect-s",
    "complementary--chevron-left--g2": "rect-s",
    "complementary--chevron-left--g5": "rect-s",
    "complementary--chevron-right--g1": "rect-s",
    "information--parking--g3": "circle",
    "information--pedestrians-crossing--g1": "square",
    "regulatory--bicycles-only--g1": "circle",
    "regulatory--go-straight--g1": "circle",
    "regulatory--go-straight-or-turn-left--g1": "circle",
    "regulatory--go-straight-or-turn-right--g1": "circle",
    "regulatory--height-limit--g1": "circle",
    "regulatory--keep-left--g1": "circle",
    "regulatory--keep-right--g1": "circle",
    "regulatory--maximum-speed-limit-100--g1": "circle",
    "regulatory--maximum-speed-limit-20--g1": "circle",
    "regulatory--maximum-speed-limit-25--g2": "rect-l",
    "regulatory--maximum-speed-limit-30--g1": "circle",
    "regulatory--maximum-speed-limit-35--g2": "rect-l",
    "regulatory--maximum-speed-limit-40--g1": "circle",
    "regulatory--maximum-speed-limit-50--g1": "circle",
    "regulatory--maximum-speed-limit-55--g2": "rect-l",
    "regulatory--maximum-speed-limit-60--g1": "circle",
    "regulatory--maximum-speed-limit-70--g1": "circle",
    "regulatory--maximum-speed-limit-80--g1": "circle",
    "regulatory--maximum-speed-limit-90--g1": "circle",
    "regulatory--no-entry--g1": "circle",
    "regulatory--no-heavy-goods-vehicles--g1": "circle",
    "regulatory--no-heavy-goods-vehicles--g2": "circle",
    "regulatory--no-heavy-goods-vehicles--g4": "circle",
    "regulatory--no-left-turn--g1": "circle",
    "regulatory--no-overtaking--g1": "circle",
    "regulatory--no-overtaking--g2": "circle",
    "regulatory--no-overtaking--g5": "circle",
    "regulatory--no-parking--g1": "circle",
    "regulatory--no-parking--g2": "circle",
    "regulatory--no-parking--g5": "circle",
    "regulatory--no-pedestrians--g2": "circle",
    "regulatory--no-right-turn--g1": "circle",
    "regulatory--no-stopping--g15": "circle",
    "regulatory--no-stopping--g2": "circle",
    "regulatory--no-stopping--g5": "circle",
    "regulatory--no-u-turn--g1": "circle",
    "regulatory--one-way-left--g2": "rect-m",
    "regulatory--pass-on-either-side--g1": "circle",
    "regulatory--pedestrians-only--g1": "circle",
    "regulatory--priority-road--g4": "diamond-s",
    "regulatory--road-closed-to-vehicles--g3": "circle",
    "regulatory--roundabout--g1": "circle",
    "regulatory--shared-path-pedestrians-and-bicycles--g1": "circle",
    "regulatory--stop--g1": "octagon",
    "regulatory--turn-left--g1": "circle",
    "regulatory--turn-left--g2": "rect-m",
    "regulatory--turn-right--g1": "circle",
    "regulatory--turn-right-ahead--g1": "circle",
    "regulatory--weight-limit--g1": "circle",
    "regulatory--yield--g1": "up-triangle",
    "warning--children--g1": "triangle",
    "warning--children--g2": "diamond-l",
    "warning--crossroads--g1": "triangle",
    "warning--crossroads--g3": "diamond-l",
    "warning--curve-left--g1": "triangle",
    "warning--curve-left--g2": "diamond-l",
    "warning--curve-right--g1": "triangle",
    "warning--curve-right--g2": "diamond-l",
    "warning--double-curve-first-right--g1": "triangle",
    "warning--height-restriction--g2": "diamond-l",
    "warning--junction-with-a-side-road-acute-left--g1": "diamond-l",
    "warning--junction-with-a-side-road-perpendicular-left--g1": "triangle",
    "warning--junction-with-a-side-road-perpendicular-left--g3": "diamond-l",
    "warning--junction-with-a-side-road-perpendicular-right--g1": "triangle",
    "warning--junction-with-a-side-road-perpendicular-right--g3": "diamond-l",
    "warning--kangaloo-crossing--g1": "diamond-l",
    "warning--narrow-bridge--g1": "diamond-l",
    "warning--other-danger--g1": "triangle",
    "warning--pedestrians-crossing--g1": "triangle",
    "warning--pedestrians-crossing--g4": "diamond-l",
    "warning--pedestrians-crossing--g5": "triangle",
    "warning--railroad-crossing-without-barriers--g3": "triangle",
    "warning--railroad-crossing-without-barriers--g4": "diamond-l",
    "warning--road-bump--g1": "triangle",
    "warning--road-bump--g2": "diamond-l",
    "warning--road-narrows-left--g2": "diamond-l",
    "warning--roadworks--g1": "triangle",
    "warning--roundabout--g1": "triangle",
    "warning--school-zone--g2": "pentagon",
    "warning--slippery-road-surface--g1": "triangle",
    "warning--slippery-road-surface--g2": "diamond-l",
    "warning--stop-ahead--g9": "diamond-l",
    "warning--texts--g1": "diamond-l",
    "warning--texts--g2": "diamond-l",
    "warning--texts--g3": "diamond-l",
    "warning--traffic-merges-right--g1": "diamond-l",
    "warning--traffic-merges-right--g2": "triangle",
    "warning--traffic-signals--g1": "triangle",
    "warning--traffic-signals--g3": "diamond-l",
    "warning--trucks-crossing--g1": "diamond-l",
    "warning--turn-right--g1": "diamond-l",
    "warning--winding-road-first-left--g1": "diamond-l",
    "warning--winding-road-first-right--g1": "diamond-l",
}
MTSD100_LABELS = list(MTSD100_TO_SHAPE.keys()) + ["other"]

LABEL_LIST = {
    "mtsd_color": TS_COLOR_LABEL_LIST,
    "mapillary_color": TS_COLOR_LABEL_LIST,
    "mtsd_no_color": TS_NO_COLOR_LABEL_LIST,
    "mapillary_no_color": TS_NO_COLOR_LABEL_LIST,
    "mtsd_100": MTSD100_LABELS,
}
LABEL_LIST["reap"] = LABEL_LIST["mapillary_no_color"]
LABEL_LIST["synthetic"] = LABEL_LIST["mapillary_no_color"]
LABEL_LIST["reap_100"] = LABEL_LIST["mtsd_100"]
LABEL_LIST["synthetic_100"] = LABEL_LIST["mtsd_100"]

# Get list of shape (no size, no color)
TS_SHAPE_LIST = list(
    set(shape.split("-", maxsplit=1)[0] for shape in TS_NO_COLOR_LABEL_LIST)
)

# =========================================================================== #

# Number of classes in each dataset
NUM_CLASSES = {
    "mtsd_orig": 401,
    "mtsd_no_color": len(TS_NO_COLOR_LABEL_LIST),
    "mtsd_color": len(TS_COLOR_LABEL_LIST),
    "mapillary_no_color": len(TS_NO_COLOR_LABEL_LIST),
    "mapillary_color": len(TS_COLOR_LABEL_LIST),
    "mtsd_100": len(MTSD100_LABELS),
}
NUM_CLASSES["reap"] = NUM_CLASSES["mapillary_no_color"]
NUM_CLASSES["synthetic"] = NUM_CLASSES["mapillary_no_color"]
NUM_CLASSES["reap_100"] = NUM_CLASSES["mtsd_100"]
NUM_CLASSES["synthetic_100"] = NUM_CLASSES["mtsd_100"]

# =========================================================================== #

# Configure dimension
_MPL_NO_COLOR_CLS_TO_SIZE_MM = {
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
_MPL_NO_COLOR_SIZE_MM = dict(enumerate(_MPL_NO_COLOR_CLS_TO_SIZE_MM.values()))

# Geometric shape of objects
# This is straightforward for our traffic sign classes, but to extend to other
# dataset in general, we need a mapping from class names to shapes.
_MPL_NO_COLOR_CLS_TO_SHAPE = {
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
_MPL_NO_COLOR_SHAPE = dict(enumerate(_MPL_NO_COLOR_CLS_TO_SHAPE.values()))

# Height-width ratio
_MPL_NO_COLOR_RATIO = {
    i: size[0] / size[1]
    for i, size in enumerate(_MPL_NO_COLOR_SIZE_MM.values())
}

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "mapillary_no_color": {
        "size_mm": _MPL_NO_COLOR_SIZE_MM,
        "hw_ratio": _MPL_NO_COLOR_RATIO,
        "shape": _MPL_NO_COLOR_SHAPE,
        "class_name": dict(enumerate(TS_NO_COLOR_LABEL_LIST)),
    }
}
DATASET_METADATA["reap"] = DATASET_METADATA["mapillary_no_color"]
DATASET_METADATA["synthetic"] = DATASET_METADATA["mapillary_no_color"]
DATASET_METADATA["mtsd_no_color"] = DATASET_METADATA["mapillary_no_color"]

_MTSD100_SIZE_MM = {
    i: _MPL_NO_COLOR_CLS_TO_SIZE_MM[v]
    for i, (k, v) in enumerate(MTSD100_TO_SHAPE.items())
}
_MTSD100_SIZE_RATIO = {
    i: size[0] / size[1] for i, size in enumerate(_MTSD100_SIZE_MM.values())
}
_MTSD100_SHAPE = {
    i: _MPL_NO_COLOR_CLS_TO_SHAPE[v]
    for i, (k, v) in enumerate(MTSD100_TO_SHAPE.items())
}
DATASET_METADATA["mtsd_100"] = {
    "size_mm": _MTSD100_SIZE_MM,
    "hw_ratio": _MTSD100_SIZE_RATIO,
    "shape": _MTSD100_SHAPE,
    "class_name": dict(enumerate(MTSD100_LABELS)),
}
DATASET_METADATA["mapillary_100"] = DATASET_METADATA["mtsd_100"]
DATASET_METADATA["reap_100"] = DATASET_METADATA["mtsd_100"]
DATASET_METADATA["synthetic_100"] = DATASET_METADATA["mtsd_100"]

# =========================================================================== #

# DEPRECATED: Kept for reference

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

OLD_TO_NEW_LABELS = {
    "circle-750.0": "circle",
    "triangle-900.0": "triangle",
    "triangle_inverted-1220.0": "up-triangle",
    "diamond-600.0": "diamond-s",
    "diamond-915.0": "diamond-l",
    "square-600.0": "square",
    "rect-458.0-610.0": "rect-s",
    "rect-762.0-915.0": "rect-m",
    "rect-915.0-1220.0": "rect-l",
    "pentagon-915.0": "pentagon",
    "octagon-915.0": "octagon",
    "other": "other",
}
