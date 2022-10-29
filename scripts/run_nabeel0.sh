#!/bin/bash

# CIRCLE
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
SYN_OBJ_PATH=attack_assets/circle-750.0.png
OBJ_CLASS=0
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=200
SHAPE=circle-750.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp10 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug


# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear









GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/triangle-900.0.png
OBJ_CLASS=1
SHAPE=triangle-900.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear












GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/triangle_inverted-1220.0.png
OBJ_CLASS=2
SHAPE=triangle_inverted-1220.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear











# DIAMOND-600
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/diamond-600.0.png
OBJ_CLASS=3
SHAPE=diamond-600.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear












# DIAMOND-950
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/diamond-915.0.png
OBJ_CLASS=4
SHAPE=diamond-915.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear
















GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
SYN_OBJ_PATH=attack_assets/octagon-915.0.png
OBJ_CLASS=10
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1
SHAPE=octagon-915.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear




















# PENTAGON
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/pentagon-915.0.png
OBJ_CLASS=9
SHAPE=pentagon-915.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp10 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 


# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug


# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear

































# RECT-458.0-610.0
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/rect-458.0-610.0.png
OBJ_CLASS=6
SHAPE=rect-458.0-610.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear

















# RECT-762.0-915.0
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/rect-762.0-915.0.png
OBJ_CLASS=7
SHAPE=rect-762.0-915.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear








# RECT-915.0-1220.0
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/rect-915.0-1220.0.png
OBJ_CLASS=8
SHAPE=rect-915.0-1220.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear






















# SQUARE
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_BG=0.1

SYN_OBJ_PATH=attack_assets/square-600.0.png
OBJ_CLASS=5
SHAPE=square-600.0

# # test yolo on real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name exp11 \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --annotated-signs-only --attack-type none 

PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME=10x10_bottom_without_transform
PATCH_NAME="10x10_bottom_without_transform_${SHAPE}"
BG_FILE_PATH="runs/val/${PATCH_NAME}/${SHAPE}/bg_filenames.txt" 

# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg $NUM_BG --attack-type real --interp bilinear

# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --annotated-signs-only \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight \
    --img-txt-path $BG_FILE_PATH --debug

# PATCH_NAME="10x10_bottom_with_transform_${SHAPE}"
# PATCH_NAME_TARGETED="10x10_bottom_with_transform_targeted_${SHAPE}"
# # test patch on real dataset WITH transforms and targeted attacks
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
#     --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME_TARGETED \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type per-sign --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear

