# Detector test script
GPU=0
NUM_GPU=1

# DATASET=mapillary-combined-color
# MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt


# Dataset and model params
DATASET=mapillary-combined-no_color
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt


# CONF_THRES=0.571
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
shapes_dims_arr=(35 45 42 32 43 43 30 41 39 52 58 27)

# shapes_arr=([0]="circle-750.0")


# i=0

# lmbd_arr=(0.00001 0.001 0.1)
# patch_dim_arr=(32 64 128)
# obj_size_arr=(256 128 64)


# lmbd_arr=(0.00001)
# patch_dim_arr=(32)
# obj_size_arr=(256)
# shapes_arr=([0]="circle-750.0")



for index in "${!shapes_arr[@]}";
do
    echo "$index -> ${shapes_arr[$index]}"


    OBJ_CLASS=$index
    SHAPE=${shapes_arr[$index]}

    CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
        --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
        --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
        --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
        --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
        --annotated-signs-only --batch-size 2 --attack-type none \
        --obj-size ${shapes_dims_arr[$index]} \
        --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
        --adv-patch-path ./runs/val/$EXP_NAME/$SHAPE/adv_patch.pkl

done


# CUR_EXP_PATH="runs/synthetic_3d_no_relighting_lambda_${LMBD}_patch_dim_${PATCH_DIM}_obj_size_${OBJ_SIZE}"
# # CUR_EXP_PATH="runs/synthetic_tl_scal_relighting_lambda_${LMBD}_patch_dim_${PATCH_DIM}_obj_size_${OBJ_SIZE}"
# echo $CUR_EXP_PATH
# mv runs/val $CUR_EXP_PATH
# mv runs/results.csv $CUR_EXP_PATH/results.csv



    # Generate adversarial patch (add --synthetic for synthetic attack)
    # CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    #     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
    #     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    #     --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
    #     --interp $INTERP --verbose --synthetic --obj-size 256 --imgsz $YOLO_IMG_SIZE \
    #     --mask-name 10x20








# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do
#         for OBJ_SIZE in ${obj_size_arr[@]};
#         do  
#             # ORIGINAL_EXP_PATH="runs/val_exp${i}"
#             # # echo $LMBD $PATCH_DIM $OBJ_SIZE

#             # path where patch is stored
#             NEW_EXP_PATH="runs/synthetic_tl_rt_lambda_${LMBD}_patch_dim_${PATCH_DIM}_obj_size_${OBJ_SIZE}"
#             echo $NEW_EXP_PATH

            
#             # echo $ORIGINAL_EXP_PATH $NEW_EXP_PATH  
#             # cp -r $ORIGINAL_EXP_PATH $NEW_EXP_PATH 
#             # ((i=i+1))

#             for index in "${!shapes_arr[@]}";
#             do
#                 echo "$index -> ${shapes_arr[$index]}"


#                 OBJ_CLASS=$index
#                 SHAPE=${shapes_arr[$index]}
#                 echo $NEW_EXP_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl
#                 # Test the generated patch
#                 CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                     --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                     --annotated-signs-only --batch-size 2 --attack-type load \
#                     --obj-size $OBJ_SIZE \
#                     --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                     --adv-patch-path $NEW_EXP_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl \
#                     --syn-3d-transform
                    
                    
#                     # --syn-use-scale --syn-use-colorjitter
#             done

#             CUR_EXP_PATH="runs/synthetic_3d_no_relighting_lambda_${LMBD}_patch_dim_${PATCH_DIM}_obj_size_${OBJ_SIZE}"
#             # CUR_EXP_PATH="runs/synthetic_tl_scal_relighting_lambda_${LMBD}_patch_dim_${PATCH_DIM}_obj_size_${OBJ_SIZE}"
#             echo $CUR_EXP_PATH
#             mv runs/val $CUR_EXP_PATH
#             mv runs/results.csv $CUR_EXP_PATH/results.csv
#         done
#     done
# done
        







# BACKUP

# for LMBD in ${lmbd_arr[@]};
# do
#     for PATCH_DIM in ${patch_dim_arr[@]};
#     do
#         for OBJ_SIZE in ${obj_size_arr[@]};
#         do  
#             # ORIGINAL_EXP_PATH="runs/val_exp${i}"
#             # # echo $LMBD $PATCH_DIM $OBJ_SIZE

#             # path where patch is stored
#             NEW_EXP_PATH="runs/synthetic_tl_rt_lambda_${LMBD}_patch_dim_${PATCH_DIM}_obj_size_${OBJ_SIZE}"
#             echo $NEW_EXP_PATH

            
#             # echo $ORIGINAL_EXP_PATH $NEW_EXP_PATH  
#             # cp -r $ORIGINAL_EXP_PATH $NEW_EXP_PATH 
#             # ((i=i+1))

#             for index in "${!shapes_arr[@]}";
#             do
#                 echo "$index -> ${shapes_arr[$index]}"


#                 OBJ_CLASS=$index
#                 SHAPE=${shapes_arr[$index]}
#                 echo $NEW_EXP_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl
#                 # Test the generated patch
#                 CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#                     --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#                     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#                     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#                     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#                     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#                     --annotated-signs-only --batch-size 2 --attack-type load \
#                     --obj-size $OBJ_SIZE \
#                     --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#                     --adv-patch-path $NEW_EXP_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl \
#                     --syn-3d-transform
                    
                    
#                     # --syn-use-scale --syn-use-colorjitter
#             done

#             CUR_EXP_PATH="runs/synthetic_3d_no_relighting_lambda_${LMBD}_patch_dim_${PATCH_DIM}_obj_size_${OBJ_SIZE}"
#             # CUR_EXP_PATH="runs/synthetic_tl_scal_relighting_lambda_${LMBD}_patch_dim_${PATCH_DIM}_obj_size_${OBJ_SIZE}"
#             echo $CUR_EXP_PATH
#             mv runs/val $CUR_EXP_PATH
#             mv runs/results.csv $CUR_EXP_PATH/results.csv
#         done
#     done
# done
        








# for OBJ_SIZE in ${obj_size_arr[@]};
# do
#     echo $OBJ_SIZE
# done

# for index in "${!shapes_arr[@]}";
# do
#     echo "$index -> ${shapes_arr[$index]}"

#     OBJ_CLASS=$index
#     SHAPE=${shapes_arr[$index]}

#     # Test the generated patch
#     CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#         --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
#         --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#         --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#         --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#         --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#         --annotated-signs-only --batch-size 2 --attack-type load \
#         --obj-size 256 \
#         --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
#         --adv-patch-path ./runs/val/$EXP_NAME/$SHAPE/adv_patch.pkl

# done


# mv runs/results.csv runs/exp24.csv
# mv runs/val runs/val_exp24

