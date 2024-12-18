# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from datasets.base_trainer import BaseTrainer
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel, get_parameter_number


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name', 
                        default="configs/DDD17/SpikeBRGNet_small_DDD17.yaml",
                        type=str)   
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()
    
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    
    # imgnet = 'imagenet' in config.MODEL.PRETRAINED
    # model = models.SpikeBRGNet.get_seg_model(config, imgnet_pretrained=imgnet)
    pretrained = config.MODEL.PRETRAINED
    model = models.SpikeBRGNet.get_seg_model(config, imgnet_pretrained=pretrained)
    params = get_parameter_number(model)
    logger.info(params)
 

    # DDD17/DSEC datasets
    base_trainer_instance = BaseTrainer()
    trainloader, testloader = base_trainer_instance.createDataLoaders()

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        # weight=train_dataset.class_weights
                                        )
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    # weight=train_dataset.class_weights
                                    )

    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model = model.cuda(config.GPU_DEVICE)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    if config.TRAIN.OPTIMIZER == 'adam':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.Adam(params,
                                lr=config.TRAIN.LR,
                                weight_decay=config.TRAIN.WD
                                )
    else:
        raise ValueError('Only Support SGD optimizer')
    
    # scheduler_lr
    # ddd17
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.97, last_epoch=-1)
    # dsec
    # scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, 10 ,gamma=0.97, last_epoch=-1)

    epoch_iters = len(trainloader)
               
    best_mIoU = 0
    best_IoU_array = np.zeros(config.DATASET.NUM_CLASSES)
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            best_IoU_array = checkpoint['best_IoU_array']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.load_state_dict(dct)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' == config.DATASET.DATASET else end_epoch
    
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, scheduler_lr, model, writer_dict)

        valid_loss, mean_IoU, IoU_array = validate(config, 
                        testloader, model, writer_dict)

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'best_IoU_array': best_IoU_array,
            # 'state_dict': model.module.state_dict(),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            best_IoU_array = IoU_array
            # torch.save(model.module.state_dict(),
            #         os.path.join(final_output_dir, 'best.pt'))
            torch.save(model.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))
         
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(best_IoU_array)
    

    # torch.save(model.module.state_dict(),
    #         os.path.join(final_output_dir, 'final_state.pt'))
    torch.save(model.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
