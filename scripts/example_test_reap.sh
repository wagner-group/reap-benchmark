#!/bin/bash
# Run evaluation on clean images
GPU=0
NUM_GPU=1
DATASET="reap"
MODEL_TYPE="faster_rcnn"
CONFIG_FILE="./configs/faster_rcnn_R_50_FPN_3x.yaml"
WEIGHTS="./results/train_mtsd_100-faster_rcnn-none/model_best.pth"

# To run on all classes, use {0..10}
# for i in {10,}; do
for i in {0..100}; do
    echo "obj class: ${i}"
    echo "===================================================================="
    echo "|              Running evaluation on clean images...               |"
    echo "===================================================================="
    CUDA_VISIBLE_DEVICES=$GPU python test_main.py \
    -e configs/cfg_reap_base.yaml \
    --options base.dataset=$DATASET \
        base.config_file=$CONFIG_FILE \
        base.base_dir='./results/' \
        base.model_name=$MODEL_TYPE \
        base.weights=$WEIGHTS \
        base.batch_size=1 \
        base.num_gpus=$NUM_GPU \
        base.obj_class=$i \
        base.conf_thres=None \
        base.verbosity=1 \
        base.workers=8 \
        base.attack_type='none' \
        attack.common.num_bg=5 \
        base.compute_conf_thres=True
    echo "==================================================================="
done
