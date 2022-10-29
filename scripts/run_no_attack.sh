# Detector test script
GPU=0
NUM_GPU=1

# Dataset and model params
# DATASET=mapillary-combined-no_color
DATASET=mtsd-no_color
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
# MODEL_PATH=./yolov5/runs/train/exp11/weights/best.pt
CONF_THRES=0.571
OUTPUT_PATH=./run/val/

# Attack params
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
YOLO_IMG_SIZE=2016
INTERP=bilinear


# shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")
shapes_arr=([0]="circle-750.0")

for index in "${!shapes_arr[@]}";
do
    echo "$index -> ${shapes_arr[$index]}"

    OBJ_CLASS=$index
    SHAPE=${shapes_arr[$index]}
    BG_DIR="backgrounds/num_bg_50/bg_filenames_${shapes_arr[$index]}.txt"

    # Test real attack using perspective transform
    EXP_NAME=perspective_relight-10x10_bottom
    ATTACK_CONFIG_PATH=./configs/attack_config_perspective_relight.yaml

    # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    #     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
    #     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --obj-class $OBJ_CLASS --bg-dir $BG_DIR \
    #     --interp $INTERP --verbose --obj-size 256 --imgsz $YOLO_IMG_SIZE

    CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
        --device $GPU --interp $INTERP --save-exp-metrics --workers 8 \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
        --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
        --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
        --batch-size 4 --attack-type none \
        --annotated-signs-only \
        --imgsz $YOLO_IMG_SIZE --verbose
        #  --conf-thres $CONF_THRES 
        # --metrics-confidence-threshold $CONF_THRES \

done
