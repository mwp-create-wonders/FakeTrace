import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import shutil

import logging
import time
import timeit
from pathlib import Path

import gc
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune CAT-Net on custom dataset')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--pretrained',
                        help='path to pretrained model (CAT_full_v2.pth.tar)',
                        required=True,
                        type=str)
    parser.add_argument('--data-root',
                        help='path to combined dataset root',
                        required=True,
                        type=str)
    parser.add_argument('--freeze-backbone',
                        help='freeze backbone layers (RGB and DCT streams)',
                        action='store_true')
    parser.add_argument('--lr',
                        help='learning rate for fine-tuning',
                        type=float,
                        default=7e-6)
    parser.add_argument('--resume',
                        help='path to checkpoint to resume from',
                        type=str,
                        default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'finetune')

    logger.info(pprint.pformat(args))
    logger.info(config)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.module.load_state_dict(resume_checkpoint['state_dict'])
        logger.info("Resume checkpoint loaded successfully")
    else:
        logger.info(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        
        if 'state_dict' in checkpoint:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.module.load_state_dict(checkpoint)
        logger.info("Pretrained model loaded successfully")

    if args.freeze_backbone:
        logger.info("Freezing backbone layers (RGB and DCT streams)")
        for name, param in model.named_parameters():
            if not name.startswith('module.last_layer') and \
               not name.startswith('module.stage5'):
                param.requires_grad = False
        
        logger.info(f"# params with requires_grad = {len([c for c in model.parameters() if c.requires_grad])}, "
                    f"# params freezed = {len([c for c in model.parameters() if not c.requires_grad])}")

    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    if config.DATASET.DATASET == 'splicing_dataset':
        train_dataset = splicing_dataset(
            crop_size=crop_size, 
            grid_crop=True, 
            blocks=('RGB', 'DCTvol', 'qtable'), 
            mode='train', 
            DCT_channels=1, 
            read_from_jpeg=True, 
            class_weight=[0.5, 2.5], 
            user_data_root=args.data_root
        )
        logger.info(train_dataset.get_info())
    else:
        raise ValueError("Not supported dataset type.")

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False, )

    valid_dataset = splicing_dataset(
        crop_size=None, 
        grid_crop=True, 
        blocks=('RGB', 'DCTvol', 'qtable'), 
        mode="valid", 
        DCT_channels=1, 
        read_from_jpeg=True, 
        user_data_root=args.data_root
    )

    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights).cuda()
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights).cuda()

    model = FullModel(model, criterion)

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                          filter(lambda p: p.requires_grad,
                                                 model.parameters()),
                                      'lr': args.lr}],
                                    lr=args.lr,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_p_mIoU = 0
    last_epoch = 0
    end_epoch = config.TRAIN.END_EPOCH

    if args.resume:
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            logger.info("Optimizer state restored")
        if 'epoch' in resume_checkpoint:
            last_epoch = resume_checkpoint['epoch']
            logger.info(f"Resuming from epoch {last_epoch}")
        if 'best_p_mIoU' in resume_checkpoint:
            best_p_mIoU = resume_checkpoint['best_p_mIoU']
            logger.info(f"Restored best_p_mIoU: {best_p_mIoU}")

    start = timeit.default_timer()
    num_iters = end_epoch * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        train_dataset.shuffle()
        train(config, epoch, end_epoch,
              epoch_iters, args.lr, num_iters,
              trainloader, optimizer, model, writer_dict, final_output_dir)

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        if epoch % 5 == 0 or epoch >= (end_epoch - 10):
            print("Start Validating..")
            writer_dict['valid_global_steps'] = epoch
            valid_loss, mean_IoU, avg_mIoU, avg_p_mIoU, IoU_array, pixel_acc, mean_acc, confusion_matrix = \
                validate(config, validloader, model, writer_dict, "valid")

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(3.0)

            if avg_p_mIoU > best_p_mIoU:
                best_p_mIoU = avg_p_mIoU
                torch.save({
                    'epoch': epoch + 1,
                    'best_p_mIoU': best_p_mIoU,
                    'state_dict': model.model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(final_output_dir, 'best.pth.tar'))
                logger.info("best.pth.tar updated.")

            msg = '(Valid) Loss: {:.3f}, MeanIU: {: 4.4f}, Best_p_mIoU: {: 4.4f}, avg_mIoU: {: 4.4f}, avg_p_mIoU: {: 4.4f}, Pixel_Acc: {: 4.4f}, Mean_Acc: {: 4.4f}'.format(
                valid_loss, mean_IoU, best_p_mIoU, avg_mIoU, avg_p_mIoU, pixel_acc, mean_acc)
            logging.info(msg)
            logging.info(IoU_array)
            logging.info("confusion_matrix:")
            logging.info(confusion_matrix)

        else:
            logging.info("Skip validation.")

        logger.info('=> saving checkpoint to {}'.format(
            os.path.join(final_output_dir, 'checkpoint.pth.tar')))
        torch.save({
            'epoch': epoch + 1,
            'best_p_mIoU': best_p_mIoU,
            'state_dict': model.model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))


if __name__ == '__main__':
    main()