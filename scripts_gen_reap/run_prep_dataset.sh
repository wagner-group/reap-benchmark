# collects cropped traffic signs and saves offset csv of these cropped traffic signs
# python3 collect_traffic_signs.py --split training
# python3 collect_traffic_signs.py --split validation

# CUDA_VISIBLE_DEVICES=1 python example_transforms.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt \
#     --split training

# CUDA_VISIBLE_DEVICES=1 python example_transforms.py \
#     --seed 0 \
#     --full-precision \
#     --batch-size 256 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/ \
#     --num-classes 12 \
#     --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt \
#     --split validation

# python3 save_images_for_manual_labeling.py --

# python3 merge_annotation_dfs.py --split training
# python3 merge_annotation_dfs.py --split validation

CUDA_VISIBLE_DEVICES=0 python fix_use_polygon_transforms.py \
    --seed 0 \
    --full-precision \
    --batch-size 256 \
    --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
    --dataset mtsd \
    --arch resnet18 \
    --output-dir /data/chawin/adv-patch-bench/results/ \
    --num-classes 12 \
    --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt \
    --split training

CUDA_VISIBLE_DEVICES=0 python fix_use_polygon_transforms.py \
    --seed 0 \
    --full-precision \
    --batch-size 256 \
    --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
    --dataset mtsd \
    --arch resnet18 \
    --output-dir /data/chawin/adv-patch-bench/results/ \
    --num-classes 12 \
    --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt \
    --split validation

