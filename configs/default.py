# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPU_DEVICE = 0
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'SpikeBRGNet_s'
_C.MODEL.PRETRAINED = 'pretrained_models/imagenet/SpikeBRGNet_S_ImageNet.pth.tar'
# _C.MODEL.PRETRAINED = False
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 2


_C.LOSS = CN()
_C.LOSS.USE_OHEM = True
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]
_C.LOSS.SB_WEIGHTS = 0.5

# DATASET related params
_C.DATASET = CN()
# _C.DATASET.ROOT = 'data/'
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
# event
_C.DATASET.DATASET_PATH = '/home/cqu/zxx/datasets/ddd17_seg'
_C.DATASET.split = 'train'
_C.DATASET.shape =  [200, 346]
_C.DATASET.nr_events_data = 20
_C.DATASET.delta_t_per_data = 50
_C.DATASET.nr_events_window = 32000
_C.DATASET.data_augmentation_train = True
_C.DATASET.event_representation = 'voxel_grid'
_C.DATASET.nr_temporal_bins = 5
_C.DATASET.require_paired_data_train = False
_C.DATASET.require_paired_data_val = True
_C.DATASET.separate_pol = False
_C.DATASET.normalize_event = False
_C.DATASET.fixed_duration = False

# training
_C.TRAIN = CN()
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True


# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False

_C.TEST.OUTPUT_INDEX = -1


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

