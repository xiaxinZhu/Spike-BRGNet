"""
Adapted from: https://github.com/uzh-rpg/rpg_ev-transfer
"""
from __future__ import division

import torch
import numpy as np


class BaseDataset(object):
    """BaseTrainer class to be inherited"""

    def __init__(self):
        # override this function to define your model, optimizers etc.
        super(BaseDataset, self).__init__()

    
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


    def createDataset(self, dataset_name, dataset_path, towns, img_size, nr_events_data, nr_events_files,
                      nr_events_window, augmentation, event_representation,
                      nr_temporal_bins, require_paired_data_train, require_paired_data_val, read_two_imgs, separate_pol,
                      normalize_event, semseg_num_classes):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(root=dataset_path,
                                        towns=towns,
                                        height=img_size[0],
                                        width=img_size[1],
                                        nr_events_data=nr_events_data,
                                        nr_events_files=nr_events_files,
                                        nr_events_window=nr_events_window,
                                        augmentation=augmentation,
                                        mode='train',
                                        event_representation=event_representation,
                                        nr_temporal_bins=nr_temporal_bins,
                                        require_paired_data=require_paired_data_train,
                                        read_two_imgs=read_two_imgs,
                                        separate_pol=separate_pol,
                                        normalize_event=normalize_event,
                                        semseg_num_classes=semseg_num_classes,
                                        fixed_duration=self.settings.fixed_duration_b)
        test_dataset = dataset_builder(root=dataset_path,
                                       height=img_size[0],
                                       width=img_size[1],
                                       nr_events_data=nr_events_data,
                                       nr_events_files=nr_events_files,
                                       nr_events_window=nr_events_window,
                                       augmentation=False,
                                       mode='val',
                                       event_representation=event_representation,
                                       nr_temporal_bins=nr_temporal_bins,
                                       require_paired_data=require_paired_data_val,
                                       read_two_imgs=read_two_imgs,
                                       separate_pol=separate_pol,
                                       normalize_event=normalize_event,
                                       semseg_num_classes=semseg_num_classes,
                                       fixed_duration=self.settings.fixed_duration_b)

        return train_dataset, test_dataset


    def createDSECDataset(self, dataset_name, dsec_dir, nr_events_data, delta_t_per_data, nr_events_window,
                          augmentation, event_representation, nr_bins_per_data, require_paired_data_train,
                          require_paired_data_val, separate_pol, normalize_event, semseg_num_classes, 
                          fixed_duration):
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
        test_dataset = dataset_builder(dsec_dir=dsec_dir,
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

        return train_dataset, test_dataset


    def createDDD17EventsDataset(self, dataset_name, root, split_train, nr_events_data, 
                                 delta_t_per_data, nr_events_per_data, augmentation, 
                                 event_representation, nr_bins_per_data, require_paired_data_train, 
                                 require_paired_data_val, separate_pol,
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
        test_dataset = dataset_builder(root=root,
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
        
        return train_dataset, test_dataset


    def createCityscapesDataset(self, dataset_name, dataset_path, img_size, semseg_num_classes,
                                augmentation, random_crop):
        """
        Creates the validation and the training data based on the provided paths and parameters.
        """
        dataset_builder = self.getDataloader(dataset_name)

        train_dataset = dataset_builder(root=dataset_path,
                                        height=img_size[0],
                                        width=img_size[1],
                                        augmentation=augmentation,
                                        split='train',
                                        semseg_num_classes=semseg_num_classes,
                                        random_crop=random_crop)
        test_dataset = dataset_builder(root=dataset_path,
                                      height=img_size[0],
                                      width=img_size[1],
                                      augmentation=False,
                                      split='val',
                                      semseg_num_classes=semseg_num_classes,
                                      random_crop=random_crop)

        return train_dataset, test_dataset
