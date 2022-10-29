#!/bin/bash
GPU=0
PATCH_NAME=yolor
EXP=46
MODEL_PATH=runs/train/yolor_p6/weights/best.pt
CSV_PATH=mapillary_vistas_final_training_merged.csv
SYN_OBJ_PATH=attack_assets/octagon-915.0.png
OBJ_CLASS=8
IMGSZ=1280
PAD_HEIGHT=992
PAD_WIDTH=1312
DATA_PATH=mapillary_vistas.yaml
DATA_PATH=yolor/data/mapillary_vistas.yaml

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
    --interp bilinear --attack-type none --min-area 0

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
    --interp bilinear --attack-type none --min-area 50

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
    --interp bilinear --attack-type none --min-area 100

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
    --interp bilinear --attack-type none --min-area 200

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
    --interp bilinear --attack-type none --min-area 400

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
    --interp bilinear --attack-type none --min-area 600

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
    --interp bilinear --attack-type none --min-area 800

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
    --interp bilinear --attack-type none --min-area 1000
    








# PATCH_NAME=yolov5
# EXP=46
# MODEL_PATH=/data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt
# CSV_PATH=mapillary_vistas_final_training_merged.csv
# SYN_OBJ_PATH=attack_assets/octagon-915.0.png

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 6 --task train --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --weights $MODEL_PATH \
#     --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
#     --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
#     --interp bilinear --attack-type none --min-area 0

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 6 --task train --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --weights $MODEL_PATH \
#     --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
#     --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
#     --interp bilinear --attack-type none --min-area 50

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 6 --task train --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --weights $MODEL_PATH \
#     --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
#     --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
#     --interp bilinear --attack-type none --min-area 100

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 6 --task train --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --weights $MODEL_PATH \
#     --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
#     --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
#     --interp bilinear --attack-type none --min-area 200

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 6 --task train --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --weights $MODEL_PATH \
#     --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
#     --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
#     --interp bilinear --attack-type none --min-area 400

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 6 --task train --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --weights $MODEL_PATH \
#     --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
#     --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
#     --interp bilinear --attack-type none --min-area 600

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 6 --task train --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --weights $MODEL_PATH \
#     --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
#     --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
#     --interp bilinear --attack-type none --min-area 800

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data $DATA_PATH --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 6 --task train --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --weights $MODEL_PATH \
#     --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
#     --imgsz $IMGSZ --padded_imgsz $PAD_HEIGHT,$PAD_WIDTH --batch-size 12 \
#     --interp bilinear --attack-type none --min-area 1000