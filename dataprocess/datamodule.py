import os
from math import log2
import random

import numpy as np
import yaml
import json
import pyvips
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# from used_models.MRCPS.utils import get_strong_aug, get_weak_aug, get_preprocess
from dataprocess.transform import get_strong_aug, get_weak_aug, get_preprocess
from dataprocess.HisPathDataset import HisPathDataset

class DataModule():
    def __init__(self, cfgpath: str, client_id=0):
        '''
        Perpare train, valid, test dataloader
        '''
        #--Load config--
        with open(cfgpath, 'r') as fp:
            self.traincfg = yaml.load(fp, Loader=yaml.FullLoader)
        
        #--For federated learning--
        # self.client_id = client_id
        # print("DataModule_ClientID", client_id)

        self.preprocess = get_preprocess()
        self.dataset =  HisPathDataset
        self.num_workers = 4

        # self.label_batchsize = self.traincfg['traindl']['batchsize'] // 2 // 4 * 4
        # self.unlabel_batchsize = self.traincfg['traindl']['batchsize'] - self.label_batchsize
        self.label_batchsize = self.traincfg['traindl']['batchsize']
        self.unlabel_batchsize = self.traincfg['traindl']['batchsize']
        print(f"label batchsize: {self.label_batchsize}\tunlabel batchsize: {self.unlabel_batchsize}")


        settings = {
            "dataroot":self.traincfg['rootset']['dataroot'],
            "datalist":self.traincfg['rootset']['datalist'],
            "classes":len(self.traincfg['classes']),
            "patchsize":self.traincfg['traindl']['patchsize'],
            "stridesize":self.traincfg['traindl']['stridesize'],
            "tifpage":self.traincfg['traindl']['tifpage'],
            "lr_ratio":self.traincfg['lr_ratio'],
            "preprocess":get_preprocess()
        }

        label_aug = get_strong_aug() if self.traincfg['traindl'].get('sda', False) \
                else get_weak_aug()

        unlabel_aug = get_weak_aug()

        # print('*****training label dataset')
        self.train_label_dataset = self.dataset(
            stage='train_label',
            transform=label_aug,
            **settings
            )

        # print('*****training unlabel dataset')
        self.train_unlabel_dataset = self.dataset(
            stage='train_unlabel',
            transform=unlabel_aug,
            **settings
            )
        self.train_size = len(self.train_label_dataset)+len(self.train_unlabel_dataset)
        
        # print('*****training valid dataset')
        self.valid_dataset = self.dataset(
            stage='valid',
            **settings
            )
        
        # print('*****training test dataset')
        self.test_dataset = self.dataset(
            stage='test',
            **settings
            )

    def seed_worker(self,worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)        

    def train_dataloader(self):
        # print('*****training dataloader')
        settings = {
            "shuffle":True,
            "drop_last":True,
            "pin_memory":True,
            "persistent_workers":True,
        }
        ## get dataset subset
        num_of_samples_l = len(self.train_label_dataset)
        # subset_l = list(range(0, len(self.train_label_dataset)))
        # random.shuffle(subset_l)
        # subset_l = subset_l[0:num_of_samples_l]
        # trainset_l = torch.utils.data.Subset(self.train_label_dataset, subset_l)
        
        num_of_samples_u = len(self.train_unlabel_dataset)
        # subset_u = list(range(0, len(self.train_unlabel_dataset)))
        # random.shuffle(subset_u)
        # subset_u = subset_u[0:num_of_samples_u]
        # trainset_u = torch.utils.data.Subset(self.train_unlabel_dataset, subset_u)
        
        dataloader_dict = {}

        if num_of_samples_l>0:
            labeled_dataloader = DataLoader(
                dataset=self.train_label_dataset,
                # dataset=trainset_l,  # use subset
                batch_size = self.label_batchsize,
                num_workers = self.num_workers,
                worker_init_fn=self.seed_worker,
                **settings
                )
            dataloader_dict['label'] = labeled_dataloader
        if num_of_samples_u>0:
            unlabeled_dataloader = DataLoader(
                dataset=self.train_unlabel_dataset,
                # dataset=trainset_u,  # use subset
                batch_size = self.unlabel_batchsize,
                num_workers = self.num_workers,
                worker_init_fn=self.seed_worker,
                **settings
                )
            dataloader_dict['unlabel'] = unlabeled_dataloader
        return dataloader_dict

    def val_dataloader(self):
        # print('*****valid dataloader')
        return DataLoader(
                dataset=self.valid_dataset,
                # dataset = sub_val,
                batch_size = self.traincfg['testdl']['batchsize'],
                num_workers=self.num_workers,
                worker_init_fn=self.seed_worker,
                pin_memory=True,
                persistent_workers=True,
                )

    def test_dataloader(self):
        # print('*****test dataloader')
        # subset = list(range(0, 100))
        # sub_test = torch.utils.data.Subset(self.test_dataset, subset)
        
        return DataLoader(
                dataset=self.test_dataset,
                # dataset=sub_test,
                batch_size= self.traincfg['testdl']['batchsize'],
                num_workers=self.num_workers,
                worker_init_fn=self.seed_worker,
                pin_memory=True,
                persistent_workers=True,
                )



