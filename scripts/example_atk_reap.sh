#!/bin/bash
# Run evaluation on clean images
GPU=0
NUM_GPU=1
DATASET="reap"
MODEL_TYPE="faster_rcnn"
CONFIG_FILE="./configs/faster_rcnn_R_50_FPN_3x.yaml"
WEIGHTS="./results/train_mtsd_100-faster_rcnn-none/model_best.pth"

# Attack parameters
PATCH_SIZE="1_10x10_bottom"
ATTACK="dpatch"
OPTIMIZER="pgd"
NUM_STEPS=100
STEP_SIZE=0.01

# To run on all classes, use {0..10}
for i in {10,}; do
    echo "obj class: ${i}"
    echo "===================================================================="
    echo "|              Running evaluation on clean images...               |"
    echo "===================================================================="
    CUDA_VISIBLE_DEVICES=$GPU python gen_adv_main.py \
        -e configs/cfg_reap_base.yaml \
        --options base.dataset=$DATASET \
            base.config_file=$CONFIG_FILE \
            base.base_dir='./results/' \
            base.model_name=$MODEL_TYPE \
            base.weights=$WEIGHTS \
            base.batch_size=1 \
            base.num_gpus=$NUM_GPU \
            base.workers=8 \
            base.obj_class=$i \
            base.verbosity=2 \
            base.attack_type='load' \
            base.patch_size=$PATCH_SIZE \
            attack.common.num_bg=5 \
            attack.common.attack_name=$ATTACK \
            attack.dpatch.optimizer=$OPTIMIZER \
            attack.dpatch.num_steps=$NUM_STEPS \
            attack.dpatch.step_size=$STEP_SIZE

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
            base.attack_type='load' \
            base.patch_size=$PATCH_SIZE \
            attack.common.num_bg=5 \
            attack.common.attack_name=$ATTACK \
            attack.dpatch.optimizer=$OPTIMIZER \
            attack.dpatch.num_steps=$NUM_STEPS \
            attack.dpatch.step_size=$STEP_SIZE
    echo "==================================================================="
done
