#!/bin/bash
GPU=0
PYTHON=python

TEST_CLEAN_OPTS="
base.dataset='reap'
base.base_dir='./results/'
base.split_file_path='./splits/all.txt'
base.batch_size=1
base.workers=8
base.attack_type='none'
base.obj_class=-1
base.verbose=False
"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON gen_relight_coeffs_main.py \
    -e configs/cfg_reap_base.yaml --options "$TEST_CLEAN_OPTS"
