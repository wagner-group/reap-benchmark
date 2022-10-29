#!/bin/bash
GPU=0
CONFIG_PATH=configs/config_detectron_reap_1.yaml
# configs/config_detectron_reap_1.yaml

i=9

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
        -e $CONFIG_PATH --obj-class $i --attack-type load


# for i in {0..10}; do
#     # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
#     #     -e $CONFIG_PATH --obj-class $i --attack-type load
#     # CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#     #     -e $CONFIG_PATH --obj-class $i --attack-type none
#     CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#         -e $CONFIG_PATH --obj-class $i --attack-type load
# done



# syn_attack_all() {
#     for i in {0..10}; do
#         CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
#             -e $1 --obj-class $i --attack-type load
#         CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#             -e $1 --obj-class $i --attack-type none
#         CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#             -e $1 --obj-class $i --attack-type load
#     done
# }

# # syn_attack_all $CONFIG_PATH
# # CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
# #     -e $CONFIG_PATH --obj-class 0 --attack-type none
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
#     -e $CONFIG_PATH
# # CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
# #     -e $CONFIG_PATH --attack-type load
# echo "Finished."

# exit 0

# # =========================================================================== #
# #                                Extra Commands                               #
# # =========================================================================== #
# # Evaluate on all Mapillary Vistas signs
# rm ./detectron_output/mapillary_combined_coco_format.json
# DATASET=mapillary-combined-no_color
# CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
#     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
#     --attack-config-path $ATK_CONFIG_PATH --workers $NUM_WORKERS \
#     --weights $MODEL_PATH --img-txt-path $BG_FILES --eval-mode drop --obj-class -1
