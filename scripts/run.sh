# #!/bin/bash
# TORCHELASTIC_MAX_RESTARTS=0
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --use_env traffic_sign_classifier.py \
#     --dist-url tcp://localhost:10005 \
#     --seed 0 \
#     --full-precision \
#     --print-freq 100 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_with_colors/ \
#     --dataset mtsd \
#     --arch resnet50 \
#     --output-dir /data/chawin/adv-patch-bench/results/5 \
#     --epochs 50 \
#     --batch-size 128 \
#     --lr 1e-2 \
#     --wd 1e-4 \
#     --pretrained \
#     --num-classes 16 \
#     --adv-train none \
#     --experiment clf
#     # --evaluate

# CUDA_VISIBLE_DEVICES=1 python example_transforms.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt

# CUDA_VISIBLE_DEVICES=0 python example_transforms_get_missed_alpha_betas.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt

# CUDA_VISIBLE_DEVICES=0,1 python fix_use_polygon_transforms.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data ~/data/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir ~/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt

# CUDA_VISIBLE_DEVICES=0 python generate_adv_patch.py \
#     --seed 0 \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --patch-name stop_sign_v1 \
#     --imgsz 1280 \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 50 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images

CUDA_VISIBLE_DEVICES=0,1 python main_classifier.py \
    --seed 0 --full-precision --batch-size 256 --arch resnet50 \
    --data ~/data/mtsd_v2_fully_annotated/cropped_signs_with_colors/ \
    --dataset mtsd --num-classes 16 --output-dir ~/adv-patch-bench/results/5/ \
    --atk-norm patch --epsilon 32 \
    --resume /data/chawin/adv-patch-bench/results/5/checkpoint_best.pt --evaluate
