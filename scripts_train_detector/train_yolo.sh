# Train
GPU=0,1,2,3
NUM_GPU=4

# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node $NUM_GPU \
#     train_yolov5.py \
#     --hyp yolov5/data/hyps/hyp.scratch.yaml \
#     --img 1280 \
#     --batch 32 \
#     --data yolov5/data/mtsd_no_color.yaml \
#     --weights yolov5/yolov5s.pt \
#     --exist-ok \
#     --workers 24 \
#     --device $GPU \
#     --save-period 15 \
#     --epochs 100

CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node $NUM_GPU \
    train_yolov5.py \
    --hyp yolov5/data/hyps/hyp.scratch.yaml --weights yolov5/yolov5s.pt --exist-ok \
    --data yolov5/data/mtsd_no_color.yaml --img 1000 --batch 128 \
    --workers 24 --device $GPU --save-period 10 --epochs 200 --sync-bn --name exp2
# NOTE: --cache uses a lot of memeory (~340G) but significantly speed up training

# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node $NUM_GPU \
#     train_yolov5.py \
#     --hyp yolov5/data/hyps/hyp.scratch.yaml \
#     --img 1280 \
#     --batch 32 \
#     --data yolov5/data/mtsd_original.yaml \
#     --weights runs/train/exp/weights/best.pt \
#     --exist-ok \
#     --workers 24 \
#     --device $GPU \
#     --save-period 15

# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node $NUM_GPU \
#     train_yolov5.py \
#     --hyp yolov5/data/hyps/hyp.scratch.yaml \
#     --img 1280 \
#     --batch 32 \
#     --data yolov5/data/mtsd_original.yaml \
#     --weights yolov5/yolov5s.pt \
#     --exist-ok \
#     --workers 24 \
#     --device $GPU \
#     --save-period 1
#     # --resume
# --exist-ok --workers 24 --device $GPU --data yolov5/data/mtsd.yaml \
# --hyp yolov5/data/hyps/hyp.finetune.yaml \
# --img 2560 --batch 16 --weights yolov5s.pt
# --resume

# CUDA_VISIBLE_DEVICES=0,1 python \
#     train_yolov5.py \
#     --hyp yolov5/data/hyps/hyp.scratch.yaml \
#     --img 1280 \
#     --batch 32 \
#     --data yolov5/data/mtsd.yaml \
#     --weights yolov5s.pt \
#     --exist-ok \
#     --workers 8 \
#     --device 0,1

# CUDA_VISIBLE_DEVICES=$GPU torchrun \
#     --standalone --nnodes=1 --max_restarts 0 --nproc_per_node $NUM_GPU \
#     val.py \
#     --img 1280 --batch 8 --data mtsd.yaml --workers 24 --device $GPU --exist-ok \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp5/weights/best.pt
#
