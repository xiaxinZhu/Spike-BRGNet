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

import _init_paths
import models
import datasets
from datasets.base_trainer import BaseTrainer
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger
from utils.function import test, testval
from ptflops import get_model_complexity_info
from thop import profile

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/DDD17/SpikeBRGNet_DDD17.yaml",
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

    logger, final_output_dir, _ = create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = models.SpikeBRGNet.get_pred_model(config, name=config.MODEL.NAME, num_classes=config.DATASET.NUM_CLASSES)
    
    # macs, params = get_model_complexity_info(model, (5,1,200,352), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # from nni.compression.pytorch.utils.counter import count_flops_params
    # dummy_input = torch.randn(1,5,1,200,352)
    # flops, params, results = count_flops_params(model, dummy_input)
    # print(flops, params, results)
    
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pt')  
   
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda(config.GPU_DEVICE)
    
    
    base_trainer_instance = BaseTrainer()
    trainloader, testloader = base_trainer_instance.createDataLoaders()
    
    start = timeit.default_timer()
   
    # 测试并保留预测可视化图
    # test(config, testloader, model, sv_dir=final_output_dir, sv_pred=True)
    
    # 测试并评估性能，无可视化图
    mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, testloader, model, sv_dir=final_output_dir) 
    msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
        pixel_acc, mean_acc)
    logging.info(msg)
    logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Mins: %d' % int((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
