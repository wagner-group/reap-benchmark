#!/usr/bin/env python
# The template of this code is generously provided by Norman Mu (@normster)
# The original version is from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.utils import save_image

import wandb
from adv_patch_bench.attacks import setup_eval_attacker, setup_train_attacker
from adv_patch_bench.attacks.patch import PatchAttackModule
from adv_patch_bench.dataloaders import load_dataset
from adv_patch_bench.models import build_classifier
from adv_patch_bench.utils import (AverageMeter, ProgressMeter, TRADESLoss,
                                   accuracy, adjust_learning_rate, get_rank,
                                   init_distributed_mode, is_main_process,
                                   save_on_master)

# Ignore warning from pytorch 1.9
warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser(description='Part classification', add_help=False)
    parser.add_argument('--data', default='~/data/shared/', type=str)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model on ImageNet-1k')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=256, type=int,
                        help='mini-batch size per device.')
    parser.add_argument('--full-precision', action='store_true')
    parser.add_argument('--warmup-epochs', default=0, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--optim', default='sgd', type=str)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB')
    # TODO
    parser.add_argument('--dataset', required=True, type=str, help='Dataset')
    parser.add_argument('--num-classes', default=10, type=int,
                        help='Number of classes')
    # parser.add_argument('--experiment', required=True, type=str,
    #                     help='Type of experiment to run')
    parser.add_argument('--adv-train', default='none', type=str,
                        help='Use adversarial training (default: none = normal training)')
    parser.add_argument('--epsilon', default=8/255, type=float,
                        help='Perturbation norm for attacks (default: 8/255)')
    parser.add_argument('--atk-norm', default='Linf', type=str,
                        help='Lp-norm of adversarial perturbation (default: Linf)')
    parser.add_argument('--trades-beta', default=6., type=float,
                        help='Beta parameter for TRADES (default: 6)')
    return parser


best_acc1 = 0


def main(args):
    init_distributed_mode(args)

    global best_acc1

    # Fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loading code
    print('=> creating dataset')
    loaders = load_dataset(args)
    if len(loaders) == 4:
        train_loader, train_sampler, val_loader, test_loader = loaders
    else:
        train_loader, train_sampler, val_loader = loaders
        test_loader = val_loader

    # Create model
    print('=> creating model')
    model, optimizer, scaler = build_classifier(args)
    cudnn.benchmark = True

    # Define loss function
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    train_criterion = criterion
    if args.adv_train == 'trades':
        train_criterion = TRADESLoss(args.trades_beta).cuda(args.gpu)

    if is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='adv-patch-bench', id=wandb_id, config=args, resume='allow')
        print('wandb step:', wandb.run.step)

    no_attack, pgd_attack = setup_eval_attacker(args, model)
    train_attack = setup_train_attacker(args, model)

    print(args)

    if not args.evaluate:
        print('=> beginning training')
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            lr = adjust_learning_rate(optimizer, epoch, args)
            print(f'=> lr @ epoch {epoch}: {lr:.2e}')

            # train for one epoch
            train_stats = train(train_loader, model, train_criterion, train_attack, optimizer, scaler, epoch, args)
            val_stats = validate(val_loader, model, criterion, no_attack, args)
            acc1, clean_acc1 = val_stats['acc1'], val_stats['acc1']
            if args.adv_train != 'none':
                val_stats = validate(val_loader, model, criterion, pgd_attack, args)
                acc1 = val_stats['acc1']

            is_best = acc1 > best_acc1 and clean_acc1 >= acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                print('=> Saving new best checkpoint')
            save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'args': args,
            }, is_best, args.output_dir)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch}

            if is_main_process():
                if args.wandb:
                    wandb.log(log_stats)
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')

    # Compute stats of best model
    best_path = f'{args.output_dir}/checkpoint_best.pt'
    print(f'=> loading best checkpoint {best_path}')
    if args.gpu is None:
        checkpoint = torch.load(best_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(best_path, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'])
    stats = validate(test_loader, model, criterion, no_attack, args)
    print(f'=> No attack: {stats}')
    stats = validate(test_loader, model, criterion, pgd_attack, args)
    print(f'=> PGD attack: {stats}')
    # stats = validate(test_loader, model, criterion, auto_attack, args)
    # print(f'=> Auto attack: {stats}')


def train(train_loader, model, criterion, attack, optimizer, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix='Epoch: [{}]'.format(epoch))

    # Switch to train mode
    model.train()

    # DEBUG
    # NUM_CLASSES = 16
    # img_list = []
    # for j in range(NUM_CLASSES):
    #     img_list.append([])

    end = time.time()
    for i, samples in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        images, targets = samples
        batch_size = images.size(0)

        # DEBUG
        # for j in range(batch_size):
        #     if len(img_list[targets[j]]) < 5:
        #         img_list[targets[j]].append(images[j])
        # if min([len(l) for l in img_list]) == 5:
        #     imgs = []
        #     for j in range(NUM_CLASSES):
        #         imgs.extend(img_list[j])
        #     save_image(imgs, 'samples.png', nrow=5)
        #     assert False

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        # Compute output
        with amp.autocast(enabled=not args.full_precision):
            images = attack(images, targets)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if args.adv_train == 'trades':
                outputs = outputs[batch_size:]

        if not math.isfinite(loss.item()):
            print('Loss is {}, stopping training'.format(loss.item()))
            sys.exit(1)

        # Measure accuracy and record loss
        acc1 = accuracy(outputs, targets)[0]
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)

        # Compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            if is_main_process() and args.wandb:
                wandb.log({'acc': acc1.item(), 'loss': loss.item(), 'scaler': scaler.get_scale()})
            progress.display(i)

    progress.synchronize()
    return {'acc1': top1.avg, 'loss': losses.avg, 'lr': optimizer.param_groups[0]['lr']}


def validate(val_loader, model, criterion, attack, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix='Test: ')
    acc_by_class = {}

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, samples in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, targets = samples

        # DEBUG
        # print(targets)
        # save_image(images[:16].view(48, 3, 128, 128), 'test.png')
        # # save_image(images, 'test.png')
        # import pdb
        # pdb.set_trace()

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        batch_size = images.size(0)

        # compute output
        with torch.no_grad():
            images = attack(images, targets)
            outputs = model(images)
            loss = criterion(outputs, targets)

        # DEBUG
        # if isinstance(attack, PatchAttackModule):
        #     save_image(images[:32].view(32, 3, 128, 128), 'test.png')
        #     import pdb
        #     pdb.set_trace()

        # measure accuracy and record loss
        acc1 = accuracy(outputs, targets)[0]
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)

        # classwise accuracy
        pred = outputs.argmax(1)
        is_correct = pred == targets
        for c in range(args.num_classes):
            num_samples = (targets == c).sum().item()
            num_correct = is_correct[targets == c].sum().item()
            if c in acc_by_class:
                acc_by_class[c][0] += num_samples
                acc_by_class[c][1] += num_correct
            else:
                acc_by_class[c] = [num_samples, num_correct]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    for c in range(args.num_classes):
        acc = acc_by_class[c][1] / (acc_by_class[c][0] + 1e-6)
        print(f'class {c} acc: {acc:.4f}')

    progress.synchronize()
    return {'acc1': top1.avg, 'loss': losses.avg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
