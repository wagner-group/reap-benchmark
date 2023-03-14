#!/bin/bash
NUM_GPU=2
PYTHON=python3.10

# MODEL_NAME="faster_rcnn"
# CONFIG_FILE="./configs/faster_rcnn_R_50_FPN_3x.yaml"
# WEIGHTS="detectron2://ImageNetPretrained/MSRA/R-50.pkl"
# WEIGHTS="./results/train_mtsd_no_color-faster_rcnn-none/model_best.pth"
# MODEL_NAME="yolof"
# CONFIG_FILE="./configs/yolof_R_50_C5_1x.yaml"
# WEIGHTS="./results/train_mtsd_no_color-yolof-none/model_best.pth"
MODEL_NAME="detrex-dino"
CONFIG_FILE="./configs/dino/dino_swin_tiny_224_4scale_12ep.py"
WEIGHTS="https://github.com/IDEA-Research/detrex-storage/releases/download/v0.1.1/dino_swin_tiny_224_22kto1k_finetune_4scale_12ep.pth"
# WEIGHTS="./results/train_mtsd-100-detrex-dino-none/model_best.pth"

TRAIN_OPTS="
base.dataset=mtsd
base.config_file=$CONFIG_FILE
base.model_name=$MODEL_NAME
base.weights=$WEIGHTS
base.base_dir='./results/'
base.split_file_path=None
base.batch_size=8
base.num_gpus=$NUM_GPU
base.workers=16
base.resume=False
base.obj_class=-1
base.reap_relight_method='percentile'
base.reap_relight_percentile=0.2
base.verbosity=1
base.use_mixed_batch=False
base.patch_size='1_10x10_middle'
base.attack_type='none'
attack.common.attack_name='dpatch'
attack.common.aug_prob_geo=1
attack.common.aug_translate=0.3
attack.common.aug_prob_colorjitter=1
attack.common.aug_colorjitter=0.1
attack.rp2.optimizer='pgd'
attack.rp2.num_steps=5
attack.rp2.step_size=0.2
attack.rp2.iou_thres=0.01
attack.dpatch.optimizer='pgd'
attack.dpatch.num_steps=5
attack.dpatch.step_size=0.2
attack.dpatch.iou_thres=0.01
"

# Train a detector on Detectron2
$PYTHON train_detectron_main.py \
    -e configs/cfg_reap_base.yaml --options "$TRAIN_OPTS"
