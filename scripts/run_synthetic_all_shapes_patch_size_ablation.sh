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


shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")


lmbd_arr=(0.00001)
# patch_dim_arr=(32 48 64 96 128 192 256)
patch_dim_arr=(64)
mask_name_arr=("10x2" "10x4" "10x6" "10x8" "10x10")

for MASK_NAME in ${mask_name_arr[@]};
do
    echo $MASK_NAME
    EXP_NAME="synthetic-${MASK_NAME}_bottom"
    echo $EXP_NAME
done



# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do
#         ATTACK_CONFIG_PATH="./configs/synthetic_obj_size_ablation/attack_config_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}.yaml"
#         for MASK_NAME in ${mask_name_arr[@]};
#         do

#             EXP_NAME="synthetic-${MASK_NAME}_bottom"

#             for index in "${!shapes_arr[@]}";
#             do
#                 echo "$index -> ${shapes_arr[$index]}"
#                 OBJ_CLASS=$index

#                 # Generate adversarial patch
#                 CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#                     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
#                     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
#                     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                     --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
#                     --interp $INTERP --verbose --synthetic \
#                     --obj-size $PATCH_DIM \
#                     --imgsz $YOLO_IMG_SIZE \
#                     --mask-name $MASK_NAME
#             done

#             SYN_PATCH_PATH="runs/paper_results/synthetic_patch_size_ablation/synthetic_patches_lambda_${LMBD}_patch_dim_${PATCH_DIM}_mask_${MASK_NAME}"
#             echo $SYN_PATCH_PATH
#             mv runs/val $SYN_PATCH_PATH


#             for index in "${!shapes_arr[@]}";
#             do
#                 echo "$index -> ${shapes_arr[$index]}"

#                 OBJ_CLASS=$index
#                 SHAPE=${shapes_arr[$index]}
#                 echo $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

#                 CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                     --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                     --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
#                     --obj-size $PATCH_DIM \
#                     --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                     --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl
#             done
            
#             CUR_EXP_PATH="runs/paper_results/synthetic_patch_size_ablation/synthetic_tl_rt_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}_mask_${MASK_NAME}"
#             echo $CUR_EXP_PATH
#             mv runs/val $CUR_EXP_PATH
#             mv runs/results.csv $CUR_EXP_PATH/results.csv

#         done
#     done
# done



shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")
# shapes_arr=([1]="triangle-900.0")


for LMBD in ${lmbd_arr[@]};
do
    for PATCH_DIM in ${patch_dim_arr[@]};
    do
        ATTACK_CONFIG_PATH="./configs/real_patch_size_ablation/attack_config_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}.yaml"
        for MASK_NAME in ${mask_name_arr[@]};
        do

            EXP_NAME="real-${MASK_NAME}_bottom"

            for index in "${!shapes_arr[@]}";
            do
                echo "$index -> ${shapes_arr[$index]}"
                OBJ_CLASS=$index
                BG_DIR="bg_txt_files/bg_filenames_${shapes_arr[$index]}.txt"

                # Generate adversarial patch
                # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
                #     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
                #     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
                #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
                #     --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
                #     --interp $INTERP --verbose --synthetic \
                #     --obj-size $PATCH_DIM \
                #     --imgsz $YOLO_IMG_SIZE \
                #     --mask-name $MASK_NAME

                CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
                --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
                --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
                --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
                --obj-class $OBJ_CLASS --bg-dir $BG_DIR \
                --interp $INTERP --verbose --obj-size 64 --imgsz $YOLO_IMG_SIZE \
                --mask-name $MASK_NAME

            done
            
            REAL_PATCH_PATH="runs/paper_results/real_patch_size_ablation/real_patches_lambda_${LMBD}_patch_dim_${PATCH_DIM}_mask_${MASK_NAME}"
            echo $REAL_PATCH_PATH
            mv runs/val $REAL_PATCH_PATH

            for index in "${!shapes_arr[@]}";
            do
                echo "$index -> ${shapes_arr[$index]}"

                OBJ_CLASS=$index
                SHAPE=${shapes_arr[$index]}
                BG_DIR="backgrounds/num_bg_50/bg_filenames_${shapes_arr[$index]}.txt"
                echo $REAL_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl
                            
                CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
                    --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
                    --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
                    --tgt-csv-filepath $CSV_PATH \
                    --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
                    --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
                    --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
                    --img-txt-path $BG_DIR \
                    --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose \
                    --adv-patch-path $REAL_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

            done
            
            CUR_EXP_PATH="runs/paper_results/real_patch_size_ablation/real_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}_mask_${MASK_NAME}"
            echo $CUR_EXP_PATH
            mv runs/val $CUR_EXP_PATH
            mv runs/results.csv $CUR_EXP_PATH/results.csv

        done
    done
done
        

