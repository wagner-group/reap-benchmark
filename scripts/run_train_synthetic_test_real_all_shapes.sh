#!/bin/bash

# Detector test script
GPU=0,1,2,3
NUM_GPU=4

BATCH_SIZE=8

# Dataset and model params
DATASET=mapillary-combined-no_color
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt

CONF_THRES=0.403
OUTPUT_PATH=./run/val/

# Attack params
ATTACK_CONFIG_PATH=./configs/attack_config2.yaml
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
YOLO_IMG_SIZE=2016
INTERP=bilinear

EXP_NAME=synthetic-10x20_bottom

shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")
# shapes_arr=([6]="rect-458.0-610.0")

# # get baseline
# for index in "${!shapes_arr[@]}";
# do
#     echo "$index -> ${shapes_arr[$index]}"

#     OBJ_CLASS=$index
#     SHAPE=${shapes_arr[$index]}

#     CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#         --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#         --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#         --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#         --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#         --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#         --annotated-signs-only --batch-size 2 --attack-type none \
#         --obj-size 64 \
#         --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#         --adv-patch-path ./runs/val/$EXP_NAME/$SHAPE/adv_patch.pkl
# done

# CUR_EXP_PATH="runs/paper_results/synthetic_baseline"
# echo $CUR_EXP_PATH
# mv runs/val $CUR_EXP_PATH
# mv runs/results.csv $CUR_EXP_PATH/results.csv






lmbd_arr=(0.00001)
patch_dim_arr=(64)

# lmbd_arr=(0.00001 0.001 0.1)
# patch_dim_arr=(32 64 128)

# Generate Adversarial Patches
# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do
#         ATTACK_CONFIG_PATH="./configs/3d_synthetic_relighting_1_configs/attack_config_3d_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}.yaml"

#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"
#             OBJ_CLASS=$index

#             # Generate adversarial patch
#             CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#                 --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
#                 --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
#                 --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                 --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
#                 --interp $INTERP --verbose --synthetic \
#                 --obj-size 64 \
#                 --imgsz $YOLO_IMG_SIZE \
#                 --mask-name 10x20
#         done

#         SYN_PATCH_PATH="runs/paper_results/synthetic_3d_patches_relighting_1_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $SYN_PATCH_PATH
#         mv runs/val $SYN_PATCH_PATH
#     done
# done
        


# Test Adversarial Patches
# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do
#         SYN_PATCH_PATH="runs/paper_results/synthetic_3d_patches_relighting_1_${LMBD}_patch_dim_${PATCH_DIM}"

#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"

#             OBJ_CLASS=$index
#             SHAPE=${shapes_arr[$index]}
#             BG_DIR="backgrounds/num_bg_50/bg_filenames_${shapes_arr[$index]}.txt"
                
#             CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                 --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                 --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                 --tgt-csv-filepath $CSV_PATH \
#                 --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                 --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                 --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
#                 --img-txt-path $BG_DIR \
#                 --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose \
#                 --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

#         done

#         CUR_EXP_PATH="runs/paper_results/synthetic_train_relighting_1_real_test_lambda_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $CUR_EXP_PATH
#         mv runs/val $CUR_EXP_PATH
#         mv runs/results.csv $CUR_EXP_PATH/results.csv

#     done 
# done



# Generate Adversarial Patches
for LMBD in ${lmbd_arr[@]};
do
    for PATCH_DIM in ${patch_dim_arr[@]};
    do
        ATTACK_CONFIG_PATH="./configs/3d_synthetic_relighting_0_configs/attack_config_3d_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}.yaml"

        for index in "${!shapes_arr[@]}";
        do
            echo "$index -> ${shapes_arr[$index]}"
            OBJ_CLASS=$index

            # Generate adversarial patch
            CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
                --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
                --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
                --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
                --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
                --interp $INTERP --verbose --synthetic \
                --obj-size 64 \
                --imgsz $YOLO_IMG_SIZE \
                --mask-name 10x20
        done

        SYN_PATCH_PATH="runs/paper_results/synthetic_3d_patches_relighting_0_${LMBD}_patch_dim_${PATCH_DIM}"
        echo $SYN_PATCH_PATH
        mv runs/val $SYN_PATCH_PATH
    done
done
        


for LMBD in ${lmbd_arr[@]};
do
    for PATCH_DIM in ${patch_dim_arr[@]};
    do
        SYN_PATCH_PATH="runs/paper_results/synthetic_3d_patches_relighting_0_${LMBD}_patch_dim_${PATCH_DIM}"

        for index in "${!shapes_arr[@]}";
        do
            echo "$index -> ${shapes_arr[$index]}"

            OBJ_CLASS=$index
            SHAPE=${shapes_arr[$index]}
            BG_DIR="backgrounds/num_bg_50/bg_filenames_${shapes_arr[$index]}.txt"
                
            CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
                --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
                --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
                --tgt-csv-filepath $CSV_PATH \
                --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
                --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
                --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
                --img-txt-path $BG_DIR \
                --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose \
                --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

        done

        CUR_EXP_PATH="runs/paper_results/synthetic_train_relighting_0_real_test_lambda_${LMBD}_patch_dim_${PATCH_DIM}"
        echo $CUR_EXP_PATH
        mv runs/val $CUR_EXP_PATH
        mv runs/results.csv $CUR_EXP_PATH/results.csv

    done 
done





# # lmbd_arr=(0.00001 0.001)
# # patch_dim_arr=(32 64 128)
# lmbd_arr=(0.1)
# patch_dim_arr=(32)
# shapes_arr=([6]="rect-458.0-610.0")


# # Test adversarial patches
# # transform: tl + rt
# # relight: 0	
# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do 

#         # path where patch is stored
#         SYN_PATCH_PATH="runs/paper_results/synthetic_patches_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $SYN_PATCH_PATH

#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"

#             OBJ_CLASS=$index
#             SHAPE=${shapes_arr[$index]}
#             echo $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

#             CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                 --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                 --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                 --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                 --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                 --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                 --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
#                 --obj-size 64 \
#                 --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                 --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl
#         done
        
#         CUR_EXP_PATH="runs/paper_results/synthetic_tl_rt_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}/synthetic-10x20_bottom_mapillary_load_6"
#         echo $CUR_EXP_PATH
#         mv runs/val/synthetic-10x20_bottom_mapillary_load_0 $CUR_EXP_PATH

#         # CUR_EXP_PATH="runs/paper_results/synthetic_tl_rt_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}"
#         # echo $CUR_EXP_PATH
#         # mv runs/val $CUR_EXP_PATH
#         # mv runs/results.csv $CUR_EXP_PATH/results.csv
        
#     done
# done
    

# # Test adversarial patches
# # transform: tl + sc
# # relight: 1	
# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do 

#         # path where patch is stored
#         SYN_PATCH_PATH="runs/paper_results/synthetic_patches_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $SYN_PATCH_PATH

#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"

#             OBJ_CLASS=$index
#             SHAPE=${shapes_arr[$index]}
#             echo $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

#             CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                 --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                 --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                 --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                 --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                 --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                 --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
#                 --obj-size 64 \
#                 --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                 --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl \
#                 --syn-use-scale --syn-use-colorjitter 
#         done
        
        
#         CUR_EXP_PATH="runs/paper_results/synthetic_scale_relighting_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $CUR_EXP_PATH
#         mv runs/val $CUR_EXP_PATH
#         mv runs/results.csv $CUR_EXP_PATH/results.csv
        
#     done
# done



# # TODO: remove
# lmbd_arr=(0.00001 0.001 0.1)
# patch_dim_arr=(32 64 128)


# # Test adversarial patches
# # transform: 3d
# # relight: 0
# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do 

#         # path where patch is stored
#         SYN_PATCH_PATH="runs/paper_results/synthetic_patches_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $SYN_PATCH_PATH

#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"

#             OBJ_CLASS=$index
#             SHAPE=${shapes_arr[$index]}
#             echo $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

#             CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                 --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                 --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                 --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                 --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                 --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                 --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
#                 --obj-size 64 \
#                 --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                 --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl \
#                 --syn-3d-transform
                                
#         done
        
#         CUR_EXP_PATH="runs/paper_results/synthetic_3d_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $CUR_EXP_PATH
#         mv runs/val $CUR_EXP_PATH
#         mv runs/results.csv $CUR_EXP_PATH/results.csv
        
#     done
# done


# # Test adversarial patches
# # transform: 3d
# # relight: 1
# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do 

#         # path where patch is stored
#         SYN_PATCH_PATH="runs/paper_results/synthetic_patches_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $SYN_PATCH_PATH

#         for index in "${!shapes_arr[@]}";
#         do
#             echo "$index -> ${shapes_arr[$index]}"

#             OBJ_CLASS=$index
#             SHAPE=${shapes_arr[$index]}
#             echo $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl

#             CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                 --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                 --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                 --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                 --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                 --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                 --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
#                 --obj-size 64 \
#                 --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                 --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl \
#                 --syn-3d-transform --syn-use-colorjitter
                                
#         done
        
        
#         CUR_EXP_PATH="runs/paper_results/synthetic_3d_relighting_attack_lambda_${LMBD}_patch_dim_${PATCH_DIM}"
#         echo $CUR_EXP_PATH
#         mv runs/val $CUR_EXP_PATH
#         mv runs/results.csv $CUR_EXP_PATH/results.csv
        
#     done
# done