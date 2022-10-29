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

configs/synthetic_color_jitter_ablation/attack_config_synthetic_lmbd_0.00001_patch_dim_64_color_jitter_0.05.yaml
configs/synthetic_color_jitter_ablation/attack_config_synthetic_lmbd_0.00001_patch_dim_64_color_jitter_0.05.yaml

shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")
EXP_NAME=synthetic-10x10_bottom

# colorjitter_intensity_arr=(0.05 0.10 0.15 0.20 0.25)
colorjitter_intensity_arr=(0.25)

# get baseline
for COLOR_INTENSITY in ${colorjitter_intensity_arr[@]};
do
    for index in "${!shapes_arr[@]}";
    do
        echo "$index -> ${shapes_arr[$index]}"

        OBJ_CLASS=$index
        SHAPE=${shapes_arr[$index]}
        CUR_EXP_PATH="runs/paper_results_new/ablation_synthetic_jitter/baseline_jitter_intensity_${COLOR_INTENSITY}"

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
            --syn-use-colorjitter --syn-colorjitter-intensity $COLOR_INTENSITY \
            --project $CUR_EXP_PATH
    done

done





# lmbd_arr=(0.00001)
# patch_dim_arr=(64)
# mask_name_arr=("10x10")

# # Generate Adversarial Patches
# for COLOR_INTENSITY in ${colorjitter_intensity_arr[@]};
# do
#     for LMBD in ${lmbd_arr[@]};
#     do
#         for PATCH_DIM in ${patch_dim_arr[@]};
#         do
#             # ATTACK_CONFIG_PATH="./configs/attack_config_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}.yaml"
#             # ATTACK_CONFIG_PATH="./configs/synthetic_rotation_ablation/attack_config_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}_rotation_${ROTATION}.yaml"
#             ATTACK_CONFIG_PATH="./configs/synthetic_color_jitter_ablation/attack_config_synthetic_lmbd_${LMBD}_color_jitter_${COLOR_INTENSITY}_patch_dim_${PATCH_DIM}.yaml"

#             for MASK_NAME in ${mask_name_arr[@]};
#             do
#                 EXP_NAME="synthetic-${MASK_NAME}_bottom"
#                 SYN_PATCH_PATH="runs/paper_results_new/ablation_synthetic_jitter/synthetic_patches_${LMBD}_patch_dim_${PATCH_DIM}_mask_name_${MASK_NAME}_jitter_intensity_${COLOR_INTENSITY}"
#                 CUR_EXP_PATH="runs/paper_results_new/ablation_synthetic_jitter/baseline_jitter_intensity_${COLOR_INTENSITY}"

#                 for index in "${!shapes_arr[@]}";
#                 do
#                     echo "$index -> ${shapes_arr[$index]}"
#                     OBJ_CLASS=$index

#                     # Generate adversarial patch
#                     CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#                         --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
#                         --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
#                         --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                         --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
#                         --interp $INTERP --verbose --synthetic \
#                         --obj-size 64 \
#                         --imgsz $YOLO_IMG_SIZE \
#                         --mask-name $MASK_NAME \
#                         --project $SYN_PATCH_PATH 

#                 done
#             done
#         done
#     done
# done
        

# # Test adversarial patches
# # transform: tl + rt
# # relight: 0
# for COLOR_INTENSITY in ${colorjitter_intensity_arr[@]};
# do	
#     for LMBD in ${lmbd_arr[@]};
#     do
#         for PATCH_DIM in ${patch_dim_arr[@]};
#         do 
#             for MASK_NAME in ${mask_name_arr[@]};
#             do
#                 # path where patch is stored
#                 SYN_PATCH_PATH="runs/paper_results_new/ablation_synthetic_jitter/synthetic_patches_${LMBD}_patch_dim_${PATCH_DIM}_mask_name_${MASK_NAME}_jitter_intensity_${COLOR_INTENSITY}"
#                 echo $SYN_PATCH_PATH

#                 # CUR_EXP_PATH="runs/paper_results/synthetic_rotation_ablation/synthetic_tl_rt_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}_mask_name_${MASK_NAME}_rotation_${ROTATION}"
#                 CUR_EXP_PATH="runs/paper_results_new/ablation_synthetic_jitter/synthetic_tl_rt_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}_mask_name_${MASK_NAME}_jitter_intensity_${COLOR_INTENSITY}"

#                 for index in "${!shapes_arr[@]}";
#                 do
#                     echo "$index -> ${shapes_arr[$index]}"

#                     OBJ_CLASS=$index
#                     SHAPE=${shapes_arr[$index]}
#                     echo $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

#                     CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                         --device $GPU --interp $INTERP --save-exp-metrics --workers 6 \
#                         --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                         --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                         --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                         --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                         --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
#                         --obj-size 64 \
#                         --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                         --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl \
#                         --syn-use-colorjitter --syn-colorjitter-intensity $COLOR_INTENSITY\
#                         --project $CUR_EXP_PATH

#                 done
#             done
#         done
#     done
# done

