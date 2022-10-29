GPU=0,1,2,3
NUM_GPU=4
NUMEXPR_MAX_THREADS=96

# CUDA_VISIBLE_DEVICES=1 python yolor/train.py \
# --batch-size 128 \
# --img 1280 960 \
# --data ./yolor/data/mapillary_vistas.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights '' \
# --device 0 \
# --name yolor_p6 \
# --hyp hyp.scratch.1280.yaml \
# --epochs 100

# --resume ./runs/train/yolor_p6/weights/epoch_049.pt

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2 \
# CUDA_VISIBLE_DEVICES=0 python3 yolor/tune.py \
# --batch-size 4 \
# --img 1280 960 \
# --data ./yolor/data/mtsd.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_w6.pt \
# --device 0 \
# --name yolor_p6 \
# --hyp hyp.finetune.1280.yaml \
# --epochs 100

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 \
# --master_port 9527 yolor/tune.py \
# --batch-size 24 \
# --img 1280 960 \
# --data ./yolor/data/mtsd.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights ./runs/train/yolor_p611/weights/best.pt \
# --device 0,1,2,3 \
# --sync-bn \
# --name yolor_p6 \
# --hyp hyp.finetune.1280.yaml \
# --epochs 100

# CUDA_LAUNCH_BLOCKING=1 python3 yolor/tune.py \
# --batch-size 4 \
# --img 1280 960 \
# --data ./yolov5/data/mtsd_original.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_p6.pt \
# --name yolor_p6_mtsd \
# --device 0 \
# --hyp hyp.finetune.1280.yaml \
# --epochs 100

# CUDA_VISIBLE_DEVICES=0 python3 yolor/tune.py \
# --batch-size 64 \
# --img 1280 960 \
# --data ./yolov5/data/mtsd_original.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_p6.pt \
# --name yolor_p6_mtsd \
# --device 0 \
# --hyp hyp.finetune.1280.yaml \
# --epochs 100

# CUDA_LAUNCH_BLOCKING=1 python3 yolor/tune.py \
# --batch-size 4 \
# --img 1280 960 \
# --data ./yolov5/data/mtsd_original.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_p6.pt \
# --name yolor_p6_mtsd \
# --device 0 \
# --hyp hyp.scratch.1280.yaml \
# --epochs 100

# CUDA_VISIBLE_DEVICES=0 python3 yolor/tune.py \
# --batch-size 4 \
# --img 1280 960 \
# --data ./yolor/data/mtsd.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_p6.pt \
# --name yolor_p6_mtsd \
# --device 0 \
# --hyp hyp.scratch.1280.yaml \
# --epochs 100

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 yolor/tune.py \
# --batch-size 24 \
# --img 1280 960 \
# --data ./yolor/data/mtsd.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_p6.pt \
# --device 0,1,2,3 \
# --sync-bn \
# --name yolor_p6_mtsd \
# --hyp hyp.scratch.1280.yaml \
# --epochs 100

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 yolor/tune.py \
# --batch-size 4 \
# --img 2560 1920 \
# --data ./yolor/data/mtsd.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_p6.pt \
# --device 0,1,2,3 \
# --sync-bn \
# --name yolor_p6_mtsd \
# --hyp hyp.scratch.1280.yaml \
# --epochs 100

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 tune_yolor.py \
# --batch-size 20 \
# --img 1280 960 \
# --data yolor/data/mtsd.yaml \
# --cfg yolor/cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_p6.pt \
# --device 0,1,2,3 \
# --sync-bn \
# --name yolor_p6_mtsd \
# --hyp hyp.scratch.1280.yaml \
# --epochs 300

CUDA_VISIBLE_DEVICES=$GPU python3 -m torch.distributed.launch \
    --standalone --nnodes 1 --max_restarts 0 --nproc_per_node $NUM_GPU \
    tune_yolor.py \
    --sync-bn --workers 96 --batch-size 20 --img 1280 960 --hyp hyp.scratch.1280.yaml \
    --cfg yolor/cfg/yolor_p6.cfg --weights yolor/scripts/yolor_p6.pt \
    --data yolov5/data/mtsd_no_color.yaml --name yolor_p6_mtsd_no_color \
    --epochs 200

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 \
# --master_port 9527 yolor/tune.py \
# --batch-size 64 \
# --img 1280 960 \
# --data yolov5/data/mtsd_original.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_p6.pt \
# --device 0,1,2,3 \
# --sync-bn \
# --name yolor_p6_mtsd \
# --hyp hyp.finetune.1280.yaml \
# --epochs 100

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

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 \
# --master_port 9527 yolor/tune.py \
# --batch-size 8 \
# --img 1280 960 \
# --data ./yolor/data/mtsd.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights ./runs/train/yolor_p6/weights/best.pt \
# --device 0,1,2,3 \
# --sync-bn \
# --name yolor_p6 \
# --hyp hyp.finetune.1280.yaml \
# --epochs 100

# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2 \
# --master_port 9527 yolor/tune.py \
# --batch-size 8 \
# --img 1280 960 \
# --data ./yolor/data/mtsd.yaml \
# --cfg cfg/yolor_p6.cfg \
# --weights yolor/scripts/yolor_w6.pt \
# --device 0,1 \
# --sync-bn \
# --name yolor_p6 \
# --hyp hyp.finetune.1280.yaml \
# --epochs 100

# --data data/coco.yaml --name yolor_p6-tune --hyp hyp.finetune.1280.yaml --epochs 450
