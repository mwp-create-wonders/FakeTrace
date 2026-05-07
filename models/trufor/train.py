import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse

import logging
import time
import timeit

import gc
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
torch.autograd.set_detect_anomaly(True)
from tensorboardX import SummaryWriter

# 导入默认参数和更新参数函数
from lib.config import config, update_config
from lib.core.function import train, validate
from lib.utils import get_model, get_optimizer
from lib.utils import create_logger, FullModel, adjust_learning_rate

from dataset.data_core import myDataset
import albumentations


def main():
    # 设置单次实验的初始化参数
    parser = argparse.ArgumentParser(description='Train TruFor')
    
    # args.experiment
    parser.add_argument('-exp', '--experiment', type=str)
    
    # GPU
    parser.add_argument('-g',   '--gpu', type=int, default=[0], nargs="+", help='device(s)')

    # 其他参数，在lib/config/default中
    parser.add_argument('opts', help='other options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # 选择GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    args.gpu = range(len(args.gpu))

    # 初始参数+本实验特殊参数-->全部整合到config
    update_config(config, args)

    # 记录日志
    logger, final_output_dir, tb_log_dir = create_logger(config, f'{args.experiment}', 'train')
    logger.info(config)
    logger.info('\n')

    # cudnn setting
    cudnn.benchmark     = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled       = config.CUDNN.ENABLED

    gpus = list(config.GPUS)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # 是否执行力训练增强，默认none
    if config.TRAIN.AUG is not None:
        aug_train = albumentations.load(config.TRAIN.AUG, data_format='yaml')
    else:
        aug_train = None

    # 是否执行验证增强，默认none
    if config.VALID.AUG is not None:
        aug_valid = albumentations.load(config.VALID.AUG, data_format='yaml')
    else:
        aug_valid = None

    # 日志格式设置
    logger.info(f'Train augmentation: {config.TRAIN.AUG} {aug_train}')
    logger.info(f'Validation augmentation: {config.VALID.AUG} {aug_valid}')

    # 设置训练集的裁剪大小，512*512
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    
    # 导入数据集
    train_dataset = myDataset(config, crop_size=crop_size, grid_crop=False, mode='train', aug=aug_train)
    valid_dataset = myDataset(config, crop_size=None, grid_crop=False, mode="valid", aug=aug_valid, max_dim=config.VALID.MAX_SIZE)

    # 设置数据集首选项（bantchsize，是否打乱，worker数量）
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle     = config.TRAIN.SHUFFLE,
        num_workers = config.WORKERS)

    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size  = 1,      # 1 to allow arbitrary input sizes
        shuffle     = False,  # must be False to get accurate filename
        num_workers = config.WORKERS)

    # model
    model = get_model(config)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model = FullModel(model, config)

    # optimizer
    optimizer = get_optimizer(model, config)

    epoch_iters = np.int32(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_key = config.VALID.BEST_KEY
    # 判断是否是loss这个指标，如果是，这先初始化为最大inf，否则最小0
    if 'loss' in best_key:
        best_value = np.inf
    else:
        best_value = 0
    logger.info(f'best valid key: {best_key}')

    # 记录上次可能中断点
    last_epoch = 0
    # 存在预训练的模型
    if not config.TRAIN.PRETRAINING == '' and not config.TRAIN.PRETRAINING == None:
        model_state_file = config.TRAIN.PRETRAINING
        assert os.path.isfile(model_state_file)
        checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage, weights_only=False)
        state_dict = checkpoint['state_dict']
        try:
            model.model.module.load_state_dict(state_dict, strict=False)
        except:
            state_dict = {k: state_dict[k] for k in state_dict if not k.startswith('detection')}
            model.model.module.load_state_dict(state_dict, strict=False)
        del checkpoint
        del state_dict
        logger.info("=> loaded pretraining ({})".format(model_state_file))

    # 中断恢复    
    if config.TRAIN.RESUME:
        # 设置输出文件夹
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage, weights_only=False)
            best_value = checkpoint['best_value']
            assert checkpoint['best_key']==best_key
            last_epoch = checkpoint['epoch']
            model.model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            writer_dict['train_global_steps'] = last_epoch
        else:
            logger.info("No previous checkpoint.")


    # 结束轮
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    
    # 从上一次的断点开始
    start_epoch = last_epoch
    
    # 在继续训练之前执行一次验证
    if config.VALID.FIRST_VALID:
        start_epoch = start_epoch -1

    for epoch in range(start_epoch, end_epoch):
        # train
        if epoch>=last_epoch:
            train_dataset.shuffle()  # for class-balanced sampling

            print(f'TRAINING epoch {epoch}:')
            # 训练
            train(epoch, config.TRAIN.END_EPOCH,
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict,
                  adjust_learning_rate=adjust_learning_rate)

            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1.0)
            
            logger.info('=> saving checkpoint to {}'.format(
                os.path.join(final_output_dir, 'checkpoint.pth.tar')))

            # 每一轮均保存
            torch.save({
                'epoch': epoch + 1,
                'best_value': best_value,
                'best_key': best_key,
                'state_dict': model.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))


        # valid
        print(f'VALIDATION epoch {epoch}:')
        writer_dict['valid_global_steps'] = epoch

        value_valid, IoU_array, confusion_matrix = \
            validate(config, validloader, model, writer_dict, "valid")

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        # 如果损失在关键参数里面，那么越小越好
        if 'loss' in best_key:
            if value_valid[best_key] < best_value:  # smallest loss
                best_value = value_valid[best_key]
                torch.save({
                    'epoch': epoch + 1,
                    'best_value': best_value,
                    'best_key': best_key,
                    'state_dict': model.model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(final_output_dir, 'best.pth.tar'))
                logger.info("best.pth.tar updated.")
        
        # 如果损失不在关键参数里面，那么越大越好
        elif value_valid[best_key] > best_value:  # highest metric
            best_value = value_valid[best_key]
            torch.save({
                'epoch': epoch + 1,
                'best_value': best_value,
                'best_key': best_key,
                'state_dict': model.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'best.pth.tar'))
            logger.info("best.pth.tar updated.")

        msg = '(Valid) Loss: {:.3f}, Best_{:s}: {: 4.4f}'.format(
            value_valid['loss'], best_key, best_value)
        logging.info(msg)
        logging.info(IoU_array)
        logging.info("confusion_matrix:")
        logging.info(confusion_matrix)




if __name__ == '__main__':
    main()
