#!/bin/bash

# Detector test script
GPU=0
NUM_GPU=1
BATCH_SIZE=1

# Dataset and model params
DATASET=mapillary-no_color
# DATASET=mtsd-no_color
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt

CONF_THRES=0.403

OUTPUT_PATH=./run/val/

# Attack params
CSV_PATH=mapillary_vistas_final_merged.csv
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
YOLO_IMG_SIZE=2016
INTERP=bilinear

shapes_arr=([0]="circle-750.0")

for index in "${!shapes_arr[@]}";
do
    echo "$index -> ${shapes_arr[$index]}"

    OBJ_CLASS=$index
    SHAPE=${shapes_arr[$index]}
    # BG_DIR="backgrounds/num_bg_50/bg_filenames_${shapes_arr[$index]}.txt"

    CUR_EXP_PATH="runs/paper_results_new/mtsd/"

    EXP_NAME=baseline

    CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
        --device $GPU --interp $INTERP --save-exp-metrics --workers 6 \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
        --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
        --tgt-csv-filepath $CSV_PATH \
        --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
        --batch-size $BATCH_SIZE --attack-type none \
        --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose \
        --project $CUR_EXP_PATH 
done

