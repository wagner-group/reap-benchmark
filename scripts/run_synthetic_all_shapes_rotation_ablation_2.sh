#!/bin/bash

# Detector test script
GPU=3
NUM_GPU=1

BATCH_SIZE=8

# Dataset and model params
DATASET=mapillary-combined-no_color
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt

CONF_THRES=0.403
OUTPUT_PATH=./run/val/

# Attack params
ATTACK_CONFIG_PATH=./configs/attack_config2.yaml
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
YOLO_IMG_SIZE=2016
INTERP=bilinear


shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")
EXP_NAME=synthetic-10x10_bottom

# rotation_degrees_arrr=(0 5 10 15 20 25 30)
rotation_degrees_arrr=(20 25 30)

# get baseline
for ROTATION in ${rotation_degrees_arrr[@]};
do
    for index in "${!shapes_arr[@]}";
    do
        echo "$index -> ${shapes_arr[$index]}"

        OBJ_CLASS=$index
        SHAPE=${shapes_arr[$index]}
        CUR_EXP_PATH="runs/paper_results/synthetic_rotation_ablation/baseline_rotation_${ROTATION}"


        CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
            --device $GPU --interp $INTERP --save-exp-metrics --workers 6 \
            --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
            --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
            --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
            --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
            --annotated-signs-only --batch-size $BATCH_SIZE --attack-type none \
            --obj-size 64 \
            --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
            --adv-patch-path ./runs/val/$EXP_NAME/$SHAPE/adv_patch.pkl \
            --syn-rotate-degree $ROTATION \
            --project $CUR_EXP_PATH
    done
done
