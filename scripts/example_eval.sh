#!/bin/bash
NUM_GPU=1
PYTHON=python3.10

# ========================= Select model to evaluate ========================= #
# MODEL_NAME="faster_rcnn"
# CONFIG_FILE="./configs/faster_rcnn_R_50_FPN_3x.yaml"
# MODEL_ID="model2"
# WEIGHTS="./results/train_mtsd_no_color-faster_rcnn-none/model_best.pth"

MODEL_NAME="yolof"
CONFIG_FILE="./configs/yolof_R_50_C5_1x.yaml"
MODEL_ID="model5"
WEIGHTS="./results/train_mtsd_no_color-yolof-none/model_best.pth"

# MODEL_NAME="detrex-dino"
# CONFIG_FILE="./configs/dino/dino_swin_tiny_224_4scale_12ep.py"
# MODEL_ID="model7"
# WEIGHTS="./results/train_mtsd-100-detrex-dino-none/model_best.pth"

# ============================ Select parameters ============================ #
DATASET="reap"
# DATASET="synthetic"
RUN_CLEAN=1  # 1: run on clean data, 0: skip
RUN_ADV=1  # 1: run on adversarial data, 0: skip

# Other params
NUM_VIS=5
COMPUTE_CONF_THRES="False"

if [ "$DATASET" = "reap" ]; then
    NUM_SAMPLES="None"
else
    NUM_SAMPLES=2000
fi

for i in {0..10}; do
    echo "**************************** START CLASS $i ****************************"
    if [ $RUN_CLEAN -eq 1 ]; then
        echo "==============================================================================="
        echo "|                        Evaluating on CLEAN data                             |"
        echo "==============================================================================="
        CLEAN_OPTS="
base.dataset=$DATASET
base.config_file=$CONFIG_FILE
base.model_name=$MODEL_NAME
base.name='_$MODEL_ID'
base.weights=$WEIGHTS
base.num_eval=$NUM_SAMPLES
base.compute_conf_thres=$COMPUTE_CONF_THRES
base.base_dir='./results/'
base.split_file_path=None
base.batch_size=1
base.num_gpus=$NUM_GPU
base.workers=8
base.obj_class=$i
base.verbosity=1
base.attack_type='none'
base.num_vis=$NUM_VIS
"
        # Test a model on clean data
        $PYTHON test_main.py \
            -e configs/cfg_reap_base.yaml --options "$CLEAN_OPTS"
    fi

    if [ $RUN_ADV -eq 1 ]; then
        echo "==============================================================================="
        echo "|                        Evaluating on ADVERSARIAL data                       |"
        echo "==============================================================================="
        ADV_OPTS="
base.dataset=$DATASET
base.config_file=$CONFIG_FILE
base.model_name=$MODEL_NAME
base.name='_$MODEL_ID'
base.weights=$WEIGHTS
base.num_eval=$NUM_SAMPLES
base.compute_conf_thres=False
base.base_dir='./results/'
base.split_file_path=None
base.batch_size=1
base.num_gpus=$NUM_GPU
base.workers=8
base.obj_class=$i
base.reap_relight_method='percentile'
base.reap_relight_percentile=0.2
base.verbosity=1
base.patch_size='1_10x10_bottom'
base.attack_type='load'
base.num_vis=$NUM_VIS
attack.common.attack_name='dpatch'
attack.common.aug_prob_geo=0
attack.common.aug_translate=0.3
attack.common.aug_prob_colorjitter=0
attack.common.aug_colorjitter=0.1
attack.rp2.optimizer='pgd'
attack.rp2.num_steps=1000
attack.rp2.step_size=0.01
attack.rp2.iou_thres=0.01
attack.dpatch.optimizer='pgd'
attack.dpatch.num_steps=1000
attack.dpatch.step_size=0.01
attack.dpatch.iou_thres=0.01
"
    # Generate adversarial patch
    $PYTHON gen_adv_main.py \
        -e configs/cfg_reap_base.yaml --options "$ADV_OPTS"
    # Test a model on clean data
    $PYTHON test_main.py \
        -e configs/cfg_reap_base.yaml --options "$ADV_OPTS"
    fi
    echo "**************************** FINISHED CLASS $i ****************************"
    echo -e "\n\n\n\n"
done
