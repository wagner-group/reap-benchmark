# Detector train script
GPU=0,1,2,3
NUM_GPU=4

# Train a YOLO detector
# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node $NUM_GPU \
#     train_yolov5.py \
#     --exist-ok --workers 24 --device $GPU --data yolov5/data/mtsd.yaml \
#     --hyp yolov5/data/hyps/hyp.scratch.yaml \
#     --img 2560 --batch 16 --weights yolov5s.pt
# --hyp yolov5/data/hyps/hyp.finetune.yaml \
# --resume

# Train a detector on Detectron2
CUDA_VISIBLE_DEVICES=$GPU python train_detectron.py \
    --num-gpus $NUM_GPU --config-file ./configs/faster_rcnn_R_50_FPN_3x.yaml \
    --dataset mtsd_no_color --eval-mode drop --resume \
    OUTPUT_DIR ./detectron_output/faster_rcnn_R_50_FPN_mtsd_no_color_2 \
    MODEL.ROI_HEADS.NUM_CLASSES 12 \
    DATALOADER.NUM_WORKERS 24
# MODEL.ROI_HEADS.NUM_CLASSES 11 # 11, 15, 401
# MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
# MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
# MODEL.WEIGHTS
# --resume --eval-only
# --eval-only MODEL.WEIGHTS /path/to/checkpoint_file

# CUDA_VISIBLE_DEVICES=$GPU python train_detectron.py \
#     --num-gpus $NUM_GPU --config-file ./configs/faster_rcnn_R_50_FPN_3x.yaml \
#     --dataset mtsd_orig --eval-mode drop \
#     OUTPUT_DIR ./detectron_output/temp \
#     MODEL.ROI_HEADS.NUM_CLASSES 401
