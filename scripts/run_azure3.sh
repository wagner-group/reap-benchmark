#!/bin/bash

# Detector test script
GPU=2
NUM_GPU=1
NUM_WORKERS=6
SEED=0

# Dataset and model params
DATASET=mapillary-combined-no_color # Options: mapillary-combined-no_color, mtsd-no_color
MODEL=faster_rcnn_R_50_FPN_mtsd_no_color_2
MODEL_PATH=~/adv-patch-bench/detectron_output/$MODEL/model_best.pth
DETECTRON_CONFIG_PATH=./configs/faster_rcnn_R_50_FPN_3x.yaml
CONF_THRES=0.634
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_TEST_SYN=5000

# Attack params
MASK_SIZE=2_10x20
SYN_OBJ_SIZE=64
ATK_CONFIG_PATH=./configs/attack_config_azure3.yaml
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/

INTERP=bilinear
TF_MODE=perspective
# per-sign-10x10-obj64-pd64-ld0.out
# synthetic-10x10-obj64-pd64-ld0.00001-2rt5.out
EXP_NAME=synthetic-${MASK_SIZE}-obj${SYN_OBJ_SIZE}-pd64-ld0.00001-2  # TODO: rename
# CLEAN_EXP_NAME=no_patch_syn_${SYN_OBJ_SIZE}
CLEAN_EXP_NAME=no_patch

# CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name "$CLEAN_EXP_NAME" \
#     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
#     --attack-config-path "$ATK_CONFIG_PATH" --workers $NUM_WORKERS --interp $INTERP \
#     --weights $MODEL_PATH --transform-mode $TF_MODE --eval-mode drop --annotated-signs-only \
#     --obj-class -1 --conf-thres $CONF_THRES --img-txt-path bg_filenames_all.txt

function syn_attack {

    OBJ_CLASS=$1
    # MASK_SIZE=$2
    # SEED=$3
    # DIST=$2
    # CLEAN_EXP_NAME=no_patch_syn_${SYN_OBJ_SIZE}_rt${RT}
    # EXP_NAME=synthetic-${MASK_SIZE}-obj${SYN_OBJ_SIZE}-pd64-3d${DIST}

    case $OBJ_CLASS in
    0) OBJ_CLASS_NAME=circle-750.0 ;;
    1) OBJ_CLASS_NAME=triangle-900.0 ;;
    2) OBJ_CLASS_NAME=triangle_inverted-1220.0 ;;
    3) OBJ_CLASS_NAME=diamond-600.0 ;;
    4) OBJ_CLASS_NAME=diamond-915.0 ;;
    5) OBJ_CLASS_NAME=square-600.0 ;;
    6) OBJ_CLASS_NAME=rect-458.0-610.0 ;;
    7) OBJ_CLASS_NAME=rect-762.0-915.0 ;;
    8) OBJ_CLASS_NAME=rect-915.0-1220.0 ;;
    9) OBJ_CLASS_NAME=pentagon-915.0 ;;
    10) OBJ_CLASS_NAME=octagon-915.0 ;;
    esac

    BG_FILES=bg_filenames_"$OBJ_CLASS_NAME".txt

    # Test on synthetic clean samples (should only be done once per aug method)
    # CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    #     --seed 0 --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name "$CLEAN_EXP_NAME" \
    #     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
    #     --attack-config-path "$ATK_CONFIG_PATH" --workers $NUM_WORKERS --interp $INTERP \
    #     --weights $MODEL_PATH --eval-mode drop --annotated-signs-only \
    #     --obj-class "$OBJ_CLASS" --obj-size $SYN_OBJ_SIZE --conf-thres $CONF_THRES \
    #     --img-txt-path $BG_FILES --num-test $NUM_TEST_SYN --synthetic \
    #     --syn-rotate-degree $RT &&

    # Generate adversarial patch
    CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
        --seed "$SEED" --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
        --attack-config-path "$ATK_CONFIG_PATH" --obj-class "$OBJ_CLASS" \
        --name "$EXP_NAME" --bg-dir $BG_PATH --transform-mode $TF_MODE \
        --weights $MODEL_PATH --workers $NUM_WORKERS --mask-name "$MASK_SIZE" \
        --img-txt-path $BG_FILES --save-images --obj-size $SYN_OBJ_SIZE \
        --annotated-signs-only --synthetic --verbose &&

    # Test patch on synthetic signs
    CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
        --seed "$SEED" --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path "$ATK_CONFIG_PATH" \
        --name "$EXP_NAME" --obj-class "$OBJ_CLASS" --conf-thres $CONF_THRES \
        --mask-name "$MASK_SIZE" --weights $MODEL_PATH --workers $NUM_WORKERS \
        --img-txt-path $BG_FILES --annotated-signs-only --obj-size $SYN_OBJ_SIZE \
        --num-test $NUM_TEST_SYN --synthetic --attack-type load &&

    # Test patch on real signs
    CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
        --seed "$SEED" --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path "$ATK_CONFIG_PATH" \
        --name "$EXP_NAME" --obj-class "$OBJ_CLASS" --conf-thres $CONF_THRES \
        --mask-name "$MASK_SIZE" --weights $MODEL_PATH --workers $NUM_WORKERS \
        --transform-mode $TF_MODE --img-txt-path $BG_FILES --attack-type load \
        --annotated-signs-only &&

    echo "Done with $OBJ_CLASS."
}

function syn_attack_all {
    for i in 6 7 8; do
        syn_attack "$i"
    done
    # syn_attack 0
}

syn_attack_all

exit 0

# =========================================================================== #
#                                Extra Commands                               #
# =========================================================================== #
# Evaluate on all Mapillary Vistas signs
rm ./detectron_output/mapillary_combined_coco_format.json
DATASET=mapillary-combined-no_color
CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
    --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
    --attack-config-path $ATK_CONFIG_PATH --workers $NUM_WORKERS \
    --weights $MODEL_PATH --img-txt-path $BG_FILES --eval-mode drop --obj-class -1
