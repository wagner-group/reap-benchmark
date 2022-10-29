#!/bin/bash
#SBATCH --job-name=apb
#SBATCH --account=fc_wagner
#SBATCH --partition=savio3_gpu
# Number of nodes:
#SBATCH --nodes=1
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=2
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:2
#SBATCH --time=36:00:00
#SBATCH --output slurm-%j-synthetic-10x10-obj64-pd64-ld0.00001-relight0.3-rt30.out  # TODO
## Command(s) to run:
source /global/home/users/$USER/.bash_profile
module purge
module load python
source activate /global/scratch/users/$USER/apb

# Detector test script
NUM_GPU=1
NUM_WORKERS=4

# Dataset and model params
DATASET=mapillary-combined-no_color # Options: mapillary-combined-no_color, mtsd-no_color
MODEL=faster_rcnn_R_50_FPN_mtsd_no_color_2
MODEL_PATH=~/adv-patch-bench/detectron_output/$MODEL/model_best.pth
DETECTRON_CONFIG_PATH=./configs/faster_rcnn_R_50_FPN_3x.yaml
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
CONF_THRES=0.634
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_TEST_SYN=5000

# Attack params
MASK_SIZE=10x10
SYN_OBJ_SIZE=64
ATK_CONFIG_PATH=./configs/attack_config_savio7.yaml

INTERP=bilinear
TF_MODE=perspective
EXP_NAME=synthetic-${MASK_SIZE}-obj${SYN_OBJ_SIZE}-pd64-ld0.00001-light0.3-rt30 # TODO: rename
CLEAN_EXP_NAME=no_patch_syn_${TF_MODE}_${SYN_OBJ_SIZE}

# Evaluate on all annotated Mapillary Vistas signs and compute score thres
# rm ./detectron_output/mapillary_combined_coco_format.json
# DATASET=mapillary-combined-no_color
# python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
#     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
#     --attack-config-path $ATK_CONFIG_PATH --workers $NUM_WORKERS \
#     --weights $MODEL_PATH --img-txt-path "$BG_FILES" --eval-mode drop --obj-class -1 \
#     --annotated-signs-only --conf-thres $CONF_THRES

function syn_attack {

    OBJ_CLASS=$1

    case $OBJ_CLASS in
    0) BG_FILES=bg_filenames_circle-750.0.txt ;;
    1) BG_FILES=bg_filenames_triangle-900.0.txt ;;
    2) BG_FILES=bg_filenames_triangle_inverted-1220.0.txt ;;
    3) BG_FILES=bg_filenames_diamond-600.0.txt ;;
    4) BG_FILES=bg_filenames_diamond-915.0.txt ;;
    5) BG_FILES=bg_filenames_square-600.0.txt ;;
    6) BG_FILES=bg_filenames_rect-458.0-610.0.txt ;;
    7) BG_FILES=bg_filenames_rect-762.0-915.0.txt ;;
    8) BG_FILES=bg_filenames_rect-915.0-1220.0.txt ;;
    9) BG_FILES=bg_filenames_pentagon-915.0.txt ;;
    10) BG_FILES=bg_filenames_octagon-915.0.txt ;;
    esac

    # Test on synthetic clean samples (should only be done once per aug method)
    # python -u test_detectron.py \
    #     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name $CLEAN_EXP_NAME \
    #     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
    #     --attack-config-path "$ATK_CONFIG_PATH" --workers $NUM_WORKERS --interp $INTERP \
    #     --weights $MODEL_PATH --eval-mode drop --annotated-signs-only \
    #     --obj-class "$OBJ_CLASS" --obj-size $SYN_OBJ_SIZE --conf-thres $CONF_THRES \
    #     --img-txt-path $BG_FILES --num-test $NUM_TEST_SYN --synthetic &&

    # Generate adversarial patch
    python -u gen_patch_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
        --attack-config-path "$ATK_CONFIG_PATH" --obj-class "$OBJ_CLASS" \
        --name "$EXP_NAME" --bg-dir $BG_PATH --transform-mode $TF_MODE \
        --weights $MODEL_PATH --workers $NUM_WORKERS --mask-name "$MASK_SIZE" \
        --img-txt-path $BG_FILES --save-images --obj-size $SYN_OBJ_SIZE \
        --annotated-signs-only --synthetic --verbose &&

    # Test patch on synthetic signs
    python -u test_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path "$ATK_CONFIG_PATH" \
        --name "$EXP_NAME" --obj-class "$OBJ_CLASS" --conf-thres $CONF_THRES \
        --mask-name "$MASK_SIZE" --weights $MODEL_PATH --workers $NUM_WORKERS \
        --transform-mode $TF_MODE --img-txt-path $BG_FILES --attack-type load \
        --annotated-signs-only --synthetic --obj-size $SYN_OBJ_SIZE \
        --num-test $NUM_TEST_SYN &&

    # Test patch on real signs
    python -u test_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path "$ATK_CONFIG_PATH" \
        --name "$EXP_NAME" --obj-class "$OBJ_CLASS" --conf-thres $CONF_THRES \
        --mask-name "$MASK_SIZE" --weights $MODEL_PATH --workers $NUM_WORKERS \
        --transform-mode $TF_MODE --img-txt-path $BG_FILES --attack-type load \
        --annotated-signs-only &&

    echo "Done with $OBJ_CLASS."
}

function syn_attack_all {
    for i in {0..10}; do
        syn_attack "$i"
    done
}

syn_attack_all

exit 0
