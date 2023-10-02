#!/bin/bash
GPU=0
NUM_GPU=1
PYTHON=python

# MODEL_ID="model1"
# MODEL_NAME="mtsd_no_color-faster_rcnn-per-sign-10x10-pd64-bg50-auggeo1_15_0.3_1-augcj1_0.1-rp2_1e-05_0_1_5_adam_0.1_False-dt0.01_0"
# CONF_THRES="[0.455,0.742,0.977,0.372,0.184,0.755,0.836,0.850,0.781,0.669,0.657,0.0]"
MODEL_ID="model2"
MODEL_NAME="mtsd_no_color-faster_rcnn-none"
MODEL_CKPT="model_best.pth"
USE_REAP=1
RUN_ATTACK=0

if [ $USE_REAP == 1 ]; then
    # REAP dataset
    DATASET="reap"
    NUM_EVAL="None"
    # model1
    # CONF_THRES="[0.455,0.742,0.977,0.372,0.184,0.755,0.836,0.850,0.781,0.669,0.657,0.0]"
    # model2
    CONF_THRES="[0.949,0.950,0.898,0.906,0.769,0.959,0.732,0.538,0.837,0.862,0.823,0.0]"
elif [ "$MODEL_ID" == "model2" ]; then
    # Synthetic dataset
    DATASET="synthetic"
    NUM_EVAL=2000
    CONF_THRES="None"
    SYN_FNR=""
fi

# for MODEL_CKPT in {"model_0004999.pth","model_0009999.pth","model_0014999.pth","model_0019999.pth","model_0024999.pth","model_0029999.pth","model_0034999.pth","model_0039999.pth","model_0044999.pth","model_0049999.pth","model_0054999.pth","model_0059999.pth","model_0064999.pth","model_0069999.pth","model_0074999.pth","model_final.pth"}; do
# for MODEL_CKPT in {"model_best.pth",}; do


for ((i = 0; i < 1; i++)); do
    echo "obj class: ${i}"

    TEST_CLEAN_OPTS="
    base.dataset=$DATASET
    base.base_dir='./results/'
    base.weights='./results/train_$MODEL_NAME/$MODEL_CKPT'
    base.batch_size=1
    base.num_gpus=$NUM_GPU
    base.obj_class=$i
    base.conf_thres=$CONF_THRES
    base.name='_$MODEL_ID'
    base.verbose=False
    base.workers=8
    base.attack_type='none'
    base.num_eval=$NUM_EVAL
    base.syn_desired_fnr=$SYN_FNR
    "
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON test_main.py \
        -e configs/cfg_reap_base.yaml --options "$TEST_CLEAN_OPTS"

    if [ $RUN_ATTACK == 1 ]; then
        TEST_OPTS="
        base.dataset=$DATASET
        base.base_dir='./results/'
        base.weights='./results/train_$MODEL_NAME/$MODEL_CKPT'
        base.batch_size=1
        base.num_gpus=$NUM_GPU
        base.workers=8
        base.obj_class=$i
        base.conf_thres=$CONF_THRES
        base.verbose=False
        base.name='_$MODEL_ID'
        base.num_eval=$NUM_EVAL
        base.attack_type='load'
        base.reap_relight_method='percentile'
        base.reap_relight_percentile=10
        attack.common.attack_name='rp2'
        attack.common.aug_prob_geo=0
        attack.common.aug_translate=0.3
        attack.common.aug_prob_colorjitter=0
        attack.common.aug_colorjitter=0.1
        attack.rp2.optimizer='pgd'
        attack.rp2.num_steps=1000
        attack.rp2.step_size=0.01
        attack.rp2.iou_thres=0.01
        attack.rp2.min_conf=0
        attack.rp2.num_eot=1
        attack.dpatch.optimizer='pgd'
        attack.dpatch.num_steps=5
        attack.dpatch.step_size=0.1
        attack.dpatch.iou_thres=0.01
        "
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON gen_adv_main.py \
            -e configs/cfg_reap_base.yaml --options "$TEST_OPTS"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON test_main.py \
            -e configs/cfg_reap_base.yaml --options "$TEST_OPTS"
    fi
done

exit 0

# TEST_CLEAN_OPTS="
# base.dataset='reap'
# base.base_dir='./results/'
# base.weights='./results/train_$MODEL_NAME/$MODEL_CKPT'
# base.split_file_path='./splits/all.txt'
# base.batch_size=1
# base.num_gpus=$NUM_GPU
# base.workers=8
# base.attack_type='none'
# base.obj_class=-1
# base.conf_thres=$CONF_THRES
# base.verbose=False
# base.name=$MODEL_ID
# "
# CUDA_VISIBLE_DEVICES=$GPU $PYTHON test_main.py \
#     -e configs/cfg_reap_base.yaml --options "$TEST_CLEAN_OPTS"


