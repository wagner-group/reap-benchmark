#!/usr/bin/env python
"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import detectron2.utils.comm as comm
import torch.multiprocessing
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import verify_results

# Import this file to register MTSD for detectron
from adv_patch_bench.dataloaders.detectron.mtsd import register_mtsd
from adv_patch_bench.utils.detectron import build_evaluator
from adv_patch_bench.dataloaders.detectron.custom_sampler import (
    RepeatFactorTrainingSampler,
)
from hparams import DATASETS, OTHER_SIGN_CLASS

torch.multiprocessing.set_sharing_strategy("file_system")


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        return build_detection_train_loader(
            cfg, sampler=RepeatFactorTrainingSampler(repeat_factors)
        )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set some custom cfg from args
    cfg.eval_mode = args.eval_mode
    cfg.other_catId = OTHER_SIGN_CLASS[args.dataset]

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    register_mtsd(
        use_mtsd_original_labels="orig" in args.dataset,
        use_color="no_color" not in args.dataset,
        ignore_bg_class=args.data_no_other,
    )

    # TODO: Need to register Mapillary here?

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--data-no-other",
        action="store_true",
        help='If True, do not load "other" or "background" class to the dataset.',
    )
    parser.add_argument("--eval-mode", type=str, default="default")
    args = parser.parse_args()

    print("Command Line Args: ", args)
    assert args.dataset in DATASETS

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
