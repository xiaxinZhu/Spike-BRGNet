"""
Adapted from: https://github.com/uzh-rpg/rpg_ev-transfer
"""
from __future__ import division

import torch
import numpy as np
from configs import config


class BaseTrainer(object):
    """BaseTrainer class to be inherited"""

    def __init__(self):
        # override this function to define your model, optimizers etc.
        super(BaseTrainer, self).__init__()

    
    def getDataloader(self, dataset_name):
        """Returns the dataset loader specified in the settings file"""
        if dataset_name == 'DSEC_events':
            from datasets.DSEC_events_loader import DSECEvents
            return DSECEvents
        elif dataset_name == 'Cityscapes_gray':
            from datasets.cityscapes_loader import CityscapesGray
            return CityscapesGray
        elif dataset_name == 'DDD17_events':
            from datasets.ddd17_events_loader import DDD17Events
            return DDD17Events

    def createDataLoaders(self):
        if config.DATASET.DATASET == 'DSEC_events':
            out = self.createDSECDataset(config.DATASET.DATASET,
                                         config.DATASET.DATASET_PATH,
                                         config.TRAIN.BATCH_SIZE_PER_GPU,
                                         config.DATASET.nr_events_data,
                                         config.DATASET.delta_t_per_data,
                                         config.DATASET.nr_events_window,
                                         config.DATASET.data_augmentation_train,
                                         config.DATASET.event_representation,
                                         config.DATASET.nr_temporal_bins,
                                         config.DATASET.require_paired_data_train,
                                         config.DATASET.require_paired_data_val,
                                         config.DATASET.separate_pol,
                                         config.DATASET.normalize_event,
                                         config.DATASET.NUM_CLASSES,
                                         config.DATASET.fixed_duration)
        elif config.DATASET.DATASET == 'DDD17_events':
            out = self.createDDD17EventsDataset(config.DATASET.DATASET,
                                                config.DATASET.DATASET_PATH,
                                                config.DATASET.split,
                                                config.TRAIN.BATCH_SIZE_PER_GPU,
                                                config.DATASET.nr_events_data,
                                                config.DATASET.delta_t_per_data,
                                                config.DATASET.nr_events_window,
                                                config.DATASET.data_augmentation_train,
                                                config.DATASET.event_representation,
                                                config.DATASET.nr_temporal_bins,
                                                config.DATASET.require_paired_data_train,
                                                config.DATASET.require_paired_data_val,
                                                config.DATASET.separate_pol,
                                                config.DATASET.normalize_event,
                                                config.DATASET.fixed_duration)
        train_loader, val_loader = out
        return train_loader, val_loader


    def createDSECDataset(self, dataset_name, dsec_dir, batch_size, nr_events_data, delta_t_per_data, nr_events_window,
                          augmentation, event_representation, nr_bins_per_data, require_paired_data_train,
                          require_paired_data_val, separate_pol, normalize_event, semseg_num_classes, fixed_duration):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(dsec_dir=dsec_dir,
                                        nr_events_data=nr_events_data,
                                        delta_t_per_data=delta_t_per_data,
                                        nr_events_window=nr_events_window,
                                        augmentation=augmentation,
                                        mode='train',
                                        event_representation=event_representation,
                                        nr_bins_per_data=nr_bins_per_data,
                                        require_paired_data=require_paired_data_train,
                                        separate_pol=separate_pol,
                                        normalize_event=normalize_event,
                                        semseg_num_classes=semseg_num_classes,
                                        fixed_duration=fixed_duration)
        val_dataset = dataset_builder(dsec_dir=dsec_dir,
                                      nr_events_data=nr_events_data,
                                      delta_t_per_data=delta_t_per_data,
                                      nr_events_window=nr_events_window,
                                      augmentation=False,
                                      mode='val',
                                      event_representation=event_representation,
                                      nr_bins_per_data=nr_bins_per_data,
                                      require_paired_data=require_paired_data_val,
                                      separate_pol=separate_pol,
                                      normalize_event=normalize_event,
                                      semseg_num_classes=semseg_num_classes,
                                      fixed_duration=fixed_duration)

        dataset_loader = torch.utils.data.DataLoader
        train_loader = dataset_loader(train_dataset, 
                                      batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                                      num_workers=config.WORKERS,
                                      pin_memory=False, 
                                      shuffle=config.TRAIN.SHUFFLE, 
                                      drop_last=True)
        val_loader = dataset_loader(val_dataset, 
                                    batch_size=config.TEST.BATCH_SIZE_PER_GPU,
                                    num_workers=config.WORKERS,
                                    pin_memory=False, 
                                    shuffle=False, 
                                    drop_last=True)
        print('DSEC num of batches: ', len(train_loader), len(val_loader))

        return train_loader, val_loader

    def createDDD17EventsDataset(self, dataset_name, root, split_train, batch_size, nr_events_data, delta_t_per_data,
                                 nr_events_per_data,
                                 augmentation, event_representation,
                                 nr_bins_per_data, require_paired_data_train, require_paired_data_val, separate_pol,
                                 normalize_event, fixed_duration):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(root=root,
                                        split=split_train,
                                        event_representation=event_representation,
                                        nr_events_data=nr_events_data,
                                        delta_t_per_data=delta_t_per_data,
                                        nr_bins_per_data=nr_bins_per_data,
                                        require_paired_data=require_paired_data_train,
                                        separate_pol=separate_pol,
                                        normalize_event=normalize_event,
                                        augmentation=augmentation,
                                        fixed_duration=fixed_duration,
                                        nr_events_per_data=nr_events_per_data)
        val_dataset = dataset_builder(root=root,
                                      split='valid',
                                      event_representation=event_representation,
                                      nr_events_data=nr_events_data,
                                      delta_t_per_data=delta_t_per_data,
                                      nr_bins_per_data=nr_bins_per_data,
                                      require_paired_data=require_paired_data_val,
                                      separate_pol=separate_pol,
                                      normalize_event=normalize_event,
                                      augmentation=False,
                                      fixed_duration=fixed_duration,
                                      nr_events_per_data=nr_events_per_data)
        
        dataset_loader = torch.utils.data.DataLoader
        train_loader = dataset_loader(train_dataset, 
                                      batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                                      num_workers=config.WORKERS,
                                      pin_memory=False, 
                                      shuffle=config.TRAIN.SHUFFLE, 
                                      drop_last=True)
        val_loader = dataset_loader(val_dataset, 
                                    batch_size=config.TEST.BATCH_SIZE_PER_GPU,
                                    num_workers=config.WORKERS,
                                    pin_memory=False, 
                                    shuffle=False, 
                                    drop_last=True)
        print('DDD17Events num of batches: ', len(train_loader), len(val_loader))

        return train_loader, val_loader
