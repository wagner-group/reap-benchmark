#!/bin/bash

# Detector test script
GPU=0,1,2,3
NUM_GPU=4
BATCH_SIZE=16

# Dataset and model params
DATASET=mapillary-combined-no_color
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
# MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt

CONF_THRES=0.403

OUTPUT_PATH=./run/val/

# Attack params
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
YOLO_IMG_SIZE=2016
INTERP=bilinear



# # baseline real attacks. only need to run on one shape to get baseline results for all shapes
# shapes_arr=([0]="circle-750.0")
# for index in "${!shapes_arr[@]}";
# do
#     echo "$index -> ${shapes_arr[$index]}"

#     OBJ_CLASS=$index
#     SHAPE=${shapes_arr[$index]}
#     BG_DIR="backgrounds/num_bg_50/bg_filenames_${shapes_arr[$index]}.txt"

#     EXP_NAME=real-10x20_bottom

#     CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#         --device $GPU --interp $INTERP --save-exp-metrics --workers 8 \
#         --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#         --tgt-csv-filepath $CSV_PATH \
#         --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#         --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#         --annotated-signs-only --batch-size 4 --attack-type none \
#         --img-txt-path $BG_DIR \
#         --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose

#     CUR_EXP_PATH="runs/paper_results/real_baseline/"
#     echo $CUR_EXP_PATH
#     mv runs/val $CUR_EXP_PATH
#     mv runs/results.csv $CUR_EXP_PATH/results.csv
# done


# debugging
# shapes_arr=([0]="circle-750.0")

lmbd_arr=(0.1)
# patch_dim_arr=(32)


shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")

# lmbd_arr=(0.00001 0.001)
patch_dim_arr=(32 64 128)


# # Generate Adversarial Patches
# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do
#         ATTACK_CONFIG_PATH="./configs/attack_config_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}.yaml"

#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"

#             OBJ_CLASS=$index
#             SHAPE=${shapes_arr[$index]}
#             BG_DIR="backgrounds/num_bg_50/bg_filenames_${shapes_arr[$index]}.txt"

#             EXP_NAME=real-10x20_bottom

#             CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#                 --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
#                 --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
#                 --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                 --obj-class $OBJ_CLASS --bg-dir $BG_DIR \
#                 --interp $INTERP --verbose --obj-size 64 --imgsz $YOLO_IMG_SIZE \
#                 --mask-name 10x20 
#         done

#         REAL_PATCH_PATH="runs/paper_results/real_patches_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $REAL_PATCH_PATH
#         mv runs/val $REAL_PATCH_PATH
#     done 
# done


EXP_NAME=real-10x20_bottom

# shapes_arr=([0]="circle-750.0")

# lmbd_arr=(0.1)
# patch_dim_arr=(32)


# Generate Adversarial Patches
for LMBD in ${lmbd_arr[@]};
do
    for PATCH_DIM in ${patch_dim_arr[@]};
    do
        REAL_PATCH_PATH="runs/paper_results/real_patches_${LMBD}_patch_dim_${PATCH_DIM}"

        for index in "${!shapes_arr[@]}";
        do
            echo "$index -> ${shapes_arr[$index]}"

            OBJ_CLASS=$index
            SHAPE=${shapes_arr[$index]}
            BG_DIR="backgrounds/num_bg_50/bg_filenames_${shapes_arr[$index]}.txt"
                
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

        CUR_EXP_PATH="runs/paper_results/real_attack_perspective_relighting_lambda_${LMBD}_patch_dim_${PATCH_DIM}"
        echo $CUR_EXP_PATH
        mv runs/val $CUR_EXP_PATH
        mv runs/results.csv $CUR_EXP_PATH/results.csv

    done 
done




    # # Test real attack using translate_scale transform
    # EXP_NAME=translate_scale_relight-10x10_bottom
    # ATTACK_CONFIG_PATH=./configs/attack_config_translate_scale_relight.yaml

    # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    #     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
    #     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --obj-class $OBJ_CLASS --bg-dir $BG_DIR \
    #     --interp $INTERP --verbose --obj-size 256 --imgsz $YOLO_IMG_SIZE

    # CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    #     --device $GPU --interp $INTERP --save-exp-metrics --workers 8 \
    #     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
    #     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
    #     --annotated-signs-only --batch-size 4 --attack-type load \
    #     --img-txt-path $BG_DIR \
    #     --imgsz $YOLO_IMG_SIZE --verbose --transform-mode translate_scale



    # # Test real attack using perspective transform no relighting
    # EXP_NAME=perspective_no_relight-10x10_bottom
    # ATTACK_CONFIG_PATH=./configs/attack_config_perspective_no_relight.yaml

    # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    #     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
    #     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --obj-class $OBJ_CLASS --bg-dir $BG_DIR \
    #     --interp $INTERP --verbose --obj-size 256 --imgsz $YOLO_IMG_SIZE
    # # --bg-dir $BG_PATH \

    # CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    #     --device $GPU --interp $INTERP --save-exp-metrics --workers 8 \
    #     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
    #     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
    #     --annotated-signs-only --batch-size 4 --attack-type load \
    #     --img-txt-path $BG_DIR \
    #     --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose



    # # Test real attack using translate_scale transform
    # EXP_NAME=translate_scale_no_relight-10x10_bottom
    # ATTACK_CONFIG_PATH=./configs/attack_config_translate_scale_no_relight.yaml

    # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    #     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
    #     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --obj-class $OBJ_CLASS --bg-dir $BG_DIR \
    #     --interp $INTERP --verbose --obj-size 256 --imgsz $YOLO_IMG_SIZE

    # CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    #     --device $GPU --interp $INTERP --save-exp-metrics --workers 8 \
    #     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
    #     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
    #     --annotated-signs-only --batch-size 4 --attack-type load \
    #     --img-txt-path $BG_DIR \
    #     --imgsz $YOLO_IMG_SIZE --verbose --transform-mode translate_scale












    # # Test synthetic attack
    # # Generate adversarial patch (add --synthetic for synthetic attack)
    # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    #     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
    #     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
    #     --interp $INTERP --verbose --synthetic --obj-size 256 --imgsz $YOLO_IMG_SIZE

    # # Test the generated patch
    # CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    #     --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
    #     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
    #     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
    #     --annotated-signs-only --batch-size 2 --attack-type load \
    #     --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
    #     --adv-patch-path ./runs/val/$EXP_NAME/$SHAPE/adv_patch.pkl

    # # Test the generated patch
    # CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    #     --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
    #     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
    #     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
    #     --annotated-signs-only --batch-size 2 --attack-type none \
    #     --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
    #     --adv-patch-path ./runs/val/$EXP_NAME/$SHAPE/adv_patch.pkl



# #!/bin/bash
# GPU=0
# EXP=1
# MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
# CSV_PATH=mapillary_vistas_final_merged.csv
# YOLO_IMG_SIZE=2016
# IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
# # NUM_BG=25

# # shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")
# # shapes_arr=([6]="rect-458.0-610.0")
# shapes_arr=([0]="circle-750.0")

# for index in "${!shapes_arr[@]}";
# do
#     echo "$index -> ${shapes_arr[$index]}"

#     OBJ_CLASS=$index
#     # OBJ_CLASS=12
#     SHAPE=${shapes_arr[$index]}
#     SYN_OBJ_PATH="attack_assets/${shapes_arr[$index]}.png"

#     PATCH_NAME="10x10_bottom_synthetic_${SHAPE}"

#     # # Generate mask for adversarial patch WITH tranforms
#     # CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
#     #     --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
#     #     --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

#     # # generate patch WITH tranforms
#     # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#     #     --device $GPU --seed 0 \
#     #     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
#     #     --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
#     #     --bg-dir ~/data/yolo_data/images/train \
#     #     --save-images --attack-config-path attack_config.yaml \
#     #     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
#     #     --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     #     --obj-size 256 --attack-type load --interp bilinear --synthetic

#     # test patch on real dataset WITH transforms
#     CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#         --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#         --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#         --weights $MODEL_PATH --exist-ok --workers 8 \
#         --conf-thres 0.001 \
#         --attack-config-path attack_config.yaml --name $PATCH_NAME \
#         --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#         --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#         --metrics-confidence-threshold 0.571 \
#         --annotated-signs-only \
#         --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear --debug --synthetic


#     # NAME="base_synthetic_${SHAPE}"
#     # # test yolo on real dataset WITHOUT patch
#     # CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     #     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     #     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     #     --weights $MODEL_PATH --exist-ok --workers 8 \
#     #     --conf-thres 0.001 \
#     #     --attack-config-path attack_config.yaml --name $NAME \
#     #     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     #     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     #     --metrics-confidence-threshold 0.571 \
#     #     --annotated-signs-only \
#     #     --attack-type none --interp bilinear --debug --synthetic \
#     #     --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl

# done