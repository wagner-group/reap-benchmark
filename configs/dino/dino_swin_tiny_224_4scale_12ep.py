"""DINO w/ Tiny Swin config.

Adapted from https://github.com/IDEA-Research/detrex/tree/main/projects/dino/configs.
"""

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detrex.config import get_config
from fvcore.common.param_scheduler import MultiStepParamScheduler

# This model will be instantiate in the main file. pylint: disable=unused-import
from .models.dino_swin_tiny_224 import model

# dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# EDIT: modify training config
# train.init_checkpoint = (
#     "/path/to/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"
# )
# train.output_dir = "./output/dino_swin_tiny_224_4scale_12ep_22kto1k_finetune"

# max training iterations
# train.max_iter = 90000

# run evaluation every 5000 iters
# train.eval_period = 5000

# log training infomation every 20 iters
# train.log_period = 20

# save checkpoint every 5000 iters
# train.checkpointer.period = 5000

# gradient clipping for training
# train.clip_grad.enabled = True
# train.clip_grad.params.max_norm = 0.1
# train.clip_grad.params.norm_type = 2
# EDIT: Use detectron2 gradient clipping
SOLVER = {
    "CLIP_GRADIENTS": {
        "ENABLED": True,
        "CLIP_TYPE": "norm",
        "CLIP_VALUE": 0.1,
        "NORM_TYPE": 2.0,
    }
}

# set training devices
train.device = "cuda"
model.device = "cuda"

# EDIT: Set num_classes
model.num_classes = 12
model.criterion.num_classes = 12

# modify optimizer config
# NOTE: We use detrex config to create optimizer and scheduler directly
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = (
    lambda module_name: 0.1 if "backbone" in module_name else 1
)

# modify dataloader config
# dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
# dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
# dataloader.evaluator.output_dir = train.output_dir

# Manually set lr schedule here to set the iteration numbers
# TODO(enhancement): Clean detrex config interface.
SOLVER = {
    **SOLVER,
    # "BASE_LR": 0.01,
    "STEPS": (50000, 70000, 75000),
    "MAX_ITER": 80000,
    "CHECKPOINT_PERIOD": 5000,
    "WARMUP_ITERS": 0,
}
scheduler = L(MultiStepParamScheduler)(
    values=[1.0, 0.1, 0.01],
    milestones=SOLVER["STEPS"],
)
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=scheduler,
    warmup_length=SOLVER["WARMUP_ITERS"] / SOLVER["MAX_ITER"],
    warmup_method="linear",
    warmup_factor=0.001,
)

DATALOADER = {
    "NUM_WORKERS": 24,
    # "SAMPLER_TRAIN": "RepeatFactorTrainingSampler",  # EDIT: default: 'TrainingSampler'
    "SAMPLER_TRAIN": "TrainingSampler",
    "REPEAT_THRESHOLD": 1.0,  # EDIT: default: 0
}

INPUT = {
    "RANDOM_FLIP": "none",  # EDIT: traffic sign can be flipped?. Option: 'none' (orig), 'horizontal' (shape)
    # EDIT: follow MTSD paper: no resizing during training but crop to (1000, 1000)
    # If not working, see https://github.com/facebookresearch/detectron2/issues/2275
    "MIN_SIZE_TRAIN": (2048,),
    "MAX_SIZE_TRAIN": 2048,  # Original training is 5000
    # Only used during traning
    "CROP": {
        "ENABLED": False,
        "TYPE": "absolute",
        "SIZE": (1000, 1000),
    },
    # EDIT: Test with max size 2048 or 4000. This is slightly different from
    # MTSD paper where images smaller than 2048 are not saled up.
    # Size of the smallest side of the image during testing. Set to zero to
    # disable resize in testing.
    "MIN_SIZE_TEST": 2048,  # This will be set by args during testing
    # Maximum size of the side of the image during testing
    "MAX_SIZE_TEST": 2048,  # This will be set by args during testing
}
