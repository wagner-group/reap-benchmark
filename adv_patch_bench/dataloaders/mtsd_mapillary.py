"""Create dataloader for both MTSD and Mapillary Vistas."""

import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from adv_patch_bench.dataloaders import eval_sampler


def get_loader_sampler(root, transform, args, split):
    """Get dataloader and sampler.

    distributed mode introduces slight variance to results if
    num_samples != 0 mod (batch size * num gpus).
    """
    dataset = datasets.ImageFolder(
        os.path.join(root, split), transform=transform
    )

    sampler = None
    if args.distributed:
        if split == "train":
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = eval_sampler.DistributedEvalSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=(split == "train"),
    )

    return loader, sampler


def load_mtsd_mapillary(args):
    """Get train and validation dataloaders from MTSD or Mapillary Vistas."""
    input_size = MTSD_MAPILLARY["input_dim"][-1]
    train_transform_list = [
        transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0
        ),
        transforms.RandomResizedCrop(input_size, scale=(0.64, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
    ]
    val_transform_list = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ]
    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose(val_transform_list)

    train_loader, train_sampler = get_loader_sampler(
        args.data, train_transform, args, "train"
    )
    val_loader, _ = get_loader_sampler(args.data, val_transform, args, "val")

    return train_loader, train_sampler, val_loader


MTSD_MAPILLARY = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_mtsd_mapillary,
    "input_dim": (3, 128, 128),
}
