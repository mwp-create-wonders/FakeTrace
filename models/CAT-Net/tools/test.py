#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for CAT-Net on user dataset
"""

import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

from lib import models
from lib.config import config
from lib.config import update_config
from lib.utils.utils import create_logger
from Splicing.data.dataset_user import UserDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--data-root',
                        help='path to user dataset root',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                        help='path to checkpoint file',
                        required=True,
                        type=str)
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
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.module.load_state_dict(checkpoint['state_dict'])
    logger.info("Checkpoint loaded successfully")

    # prepare data
    test_dataset = UserDataset(
        crop_size=None, 
        grid_crop=True, 
        blocks=('RGB', 'DCTvol', 'qtable'), 
        DCT_channels=1, 
        data_root=args.data_root, 
        mode='test', 
        read_from_jpeg=True
    )
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples")

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False
    )

    # Test
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for _, (image, label, qtable) in enumerate(tqdm(testloader)):
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            # Forward pass
            pred = model(image, qtable)
            pred = nn.functional.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')

            # Convert to probabilities and predictions
            prob = nn.functional.softmax(pred, dim=1)
            seg_prob = prob[:, 1, :, :]  # Probability of tampered class
            seg_pred = torch.argmax(pred, dim=1)

            # Flatten and collect results
            label_flat = label.cpu().numpy().flatten()
            pred_flat = seg_pred.cpu().numpy().flatten()
            prob_flat = seg_prob.cpu().numpy().flatten()

            # Filter out ignore labels if any
            valid_mask = label_flat != config.TRAIN.IGNORE_LABEL
            all_labels.extend(label_flat[valid_mask])
            all_preds.extend(pred_flat[valid_mask])
            all_probs.extend(prob_flat[valid_mask])

    # Calculate metrics
    pixel_f1 = f1_score(all_labels, all_preds, average='binary')
    pixel_auc = roc_auc_score(all_labels, all_probs)

    # Log results
    logger.info(f"Pixel-level F1 score: {pixel_f1:.4f}")
    logger.info(f"Pixel-level AUC: {pixel_auc:.4f}")

    print(f"Pixel-level F1 score: {pixel_f1:.4f}")
    print(f"Pixel-level AUC: {pixel_auc:.4f}")


if __name__ == '__main__':
    main()
