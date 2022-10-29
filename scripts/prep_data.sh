GPU=0

CUDA_VISIBLE_DEVICES=$GPU python prep_mapillary.py --split train \
    --resume ~/data/adv-patch-bench/results/6/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=$GPU python prep_mapillary.py --split val \
    --resume ~/data/adv-patch-bench/results/6/checkpoint_best.pt
