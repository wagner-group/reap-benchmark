#!/bin/bash
# This script is used to prepare MTSD/Mapillary Vistas datasets for REAP
GPU=0

# Set some parameters
BASE_DIR="./scripts_gen_reap/"

# Prepare MTSD dataset for classification
python $BASE_DIR/prep_mtsd_for_classification.py

# Train traffic sign classifier on MTSD
# TODO(user): Change path for other datasets/modifiers
CLF_DATA_PATH="$HOME/data/mtsd_v2_fully_annotated/cropped_signs_mtsd-100/"
NUM_CLASSES=100  # 12 for no_color
ARCH="convnext_small_in22k"
CUDA_VISIBLE_DEVICES=$GPU python train_classifier_main.py \
    --seed 0 --workers 8 --dataset "mtsd" --data "$CLF_DATA_PATH" \
    --arch $ARCH --full-precision --pretrained --epochs 100 \
    --batch-size 128 --optim "adamw" --lr 5e-5 --wd 1e-8 \
    --output-dir "./results/mtsd_full/" \
    --num-classes $NUM_CLASSES --balance-sampler

# Use the trained classifier to (pseudo) label the cropped traffic signs from
# Mapillary Vistas. Use prep_mapillary.py for REAP and prep_mapillary_100.py
# for REAP-100.
CUDA_VISIBLE_DEVICES=$GPU python $BASE_DIR/prep_mapillary_100.py

# =========================================================================== #
# DEPRECATED

# collects cropped traffic signs and saves offset csv of these cropped traffic signs
# python3 collect_traffic_signs.py --split training
# python3 collect_traffic_signs.py --split validation

# CUDA_VISIBLE_DEVICES=1 python example_transforms.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt \
#     --split training

# CUDA_VISIBLE_DEVICES=1 python example_transforms.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt \
#     --split validation

# python3 save_images_for_manual_labeling.py --

# python3 merge_annotation_dfs.py --split training
# python3 merge_annotation_dfs.py --split validation

# CUDA_VISIBLE_DEVICES=0 python fix_use_polygon_transforms.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt \
#     --split training

# CUDA_VISIBLE_DEVICES=0 python fix_use_polygon_transforms.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt \
#     --split validation

