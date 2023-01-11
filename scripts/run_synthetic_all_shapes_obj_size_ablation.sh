#!/bin/bash

# Detector test script
GPU=0,1,2,3
NUM_GPU=4

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

EXP_NAME=synthetic-10x20_bottom

shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")

lmbd_arr=(0.00001)
patch_dim_arr=(32 48 64 96 128 192 256)

for PATCH_DIM in ${patch_dim_arr[@]};
do
    # get baseline
    for index in "${!shapes_arr[@]}";
    do
        echo "$index -> ${shapes_arr[$index]}"

        OBJ_CLASS=$index
        SHAPE=${shapes_arr[$index]}

        CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
            --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
            --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
            --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
            --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
            --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
            --annotated-signs-only --batch-size $BATCH_SIZE --attack-type none \
            --obj-size $PATCH_DIM \
            --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic 
    done
    CUR_EXP_PATH="runs/paper_results/synthetic_obj_size_ablation/baseline/patch_dim_${PATCH_DIM}"
    echo $CUR_EXP_PATH
    mv runs/val $CUR_EXP_PATH
    mv runs/results.csv $CUR_EXP_PATH/results.csv
done










# lmbd_arr=(0.00001 0.001 0.1)
# patch_dim_arr=(32 64 128)

# # Generate Adversarial Patches
# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do
#         ATTACK_CONFIG_PATH="./configs/synthetic_obj_size_ablation/attack_config_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}.yaml"

#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"
#             OBJ_CLASS=$index

#             # Generate adversarial patch
#             CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#                 --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
#                 --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
#                 --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                 --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
#                 --interp $INTERP --verbose --synthetic \
#                 --obj-size $PATCH_DIM \
#                 --imgsz $YOLO_IMG_SIZE \
#                 --mask-name 10x20
#         done

#         SYN_PATCH_PATH="runs/paper_results/synthetic_obj_size_ablation/synthetic_patches_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $SYN_PATCH_PATH
#         mv runs/val $SYN_PATCH_PATH


#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"

#             OBJ_CLASS=$index
#             SHAPE=${shapes_arr[$index]}
#             echo $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

#             CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                 --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                 --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                 --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                 --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                 --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                 --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
#                 --obj-size $PATCH_DIM \
#                 --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                 --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl
#         done
        
#         CUR_EXP_PATH="runs/paper_results/synthetic_obj_size_ablation/synthetic_tl_rt_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $CUR_EXP_PATH
#         mv runs/val $CUR_EXP_PATH
#         mv runs/results.csv $CUR_EXP_PATH/results.csv


#     done
# done
        
