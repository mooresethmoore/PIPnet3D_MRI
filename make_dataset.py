import argparse
import os
import math
import numpy as np
import SimpleITK as sitk
import random
import pandas as pd
from typing import Tuple, Dict

import torch
from torch import Tensor
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import json

from monai.transforms import (
    Compose,
    Resize,
    RandRotate,
    Affine,
    RandGaussianNoise,
    RandZoom,
    RepeatChannel,
)

import joblib
import h5py


cuda=torch.cuda.is_available()



pretrain_batchsize = 5
channels=3
# Data augmentation (on-the-fly) parameters
aug_prob = 1
rand_rot = 10                       # random rotation range [deg]
rand_rot_rad = rand_rot*math.pi/180 # random rotation range [rad]
rand_noise_std = 0.01               # std random Gaussian noise
rand_shift = 5                      # px random shift
min_zoom = 0.9
max_zoom = 1.1


useGPU=True





def kFold_TrainTestSplit(arr,k:int,testP=.1,shuffle=True):
    assert len(arr)>1 and testP<=1
    folds=[] # [([...Tr_k...],[...Te_k...])]
    testSize=round(testP*len(arr))
    
    for i in range(k):
        testBin=list(np.random.choice(arr,size=testSize,replace=False,))
        trainBin=list(set(arr)-set(testBin))
        if shuffle:
            np.random.shuffle(testBin)
            np.random.shuffle(trainBin)
        folds.append({"train":trainBin,"test":testBin})

    return folds

def gen_kFold_TrainTestSplit(arr,k,trP,fileName="kFoldGen.sav"):
    folds=kFold_TrainTestSplit(arr,k,trP)
    joblib.dump(folds,fileName)

def load_kFold_TrainTestSplit(fileName="kFoldGen.sav"):
    """
    returns a list of size k:
    [{"train":[tr_k],"test":[te_k]},...]
    """
    
    return joblib.load(fileName)




transforms_dic = {
    'train': Compose([
        RandRotate(range_x=rand_rot_rad, 
                    range_y=rand_rot_rad, 
                    range_z=rand_rot_rad, 
                    prob=aug_prob),
        RandGaussianNoise(std=rand_noise_std, prob=aug_prob),
        Affine(translate_params=(rand_shift,
                                    rand_shift,
                                    rand_shift), 
                image_only=True),
        RandZoom(min_zoom=min_zoom, max_zoom=max_zoom, prob=aug_prob),
        RepeatChannel(repeats=channels),
    ]),
    'train_noaug': Compose([RepeatChannel(repeats=channels)]),
    'project_noaug':Compose([RepeatChannel(repeats=channels)]),
    'val': Compose([RepeatChannel(repeats=channels)]),
    'test': Compose([RepeatChannel(repeats=channels)]),
    'test_projection': Compose([RepeatChannel(repeats=channels)]),
}



class AugSupervisedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_h5path,
                 yflag_dict,
                 subsetKeys=None,
                 transform = None,

                 ):
        self.dataset_path=dataset_h5path

        self.yflag_dict=yflag_dict
        self.classes=[0,1]
        self.class_to_idx={0:0,1:1} #classname -> id for network
        if subsetKeys:
            self.subsetKeys=subsetKeys
        else:
            with h5py.File(self.dataset_path, 'r') as f:
                self.subsetKeys=list(f.keys())
                self.subsetKeys.sort()
        
        #if transform is None:
        #    self.transform = lambda x: x
        #else:
        self.transform=transform
        
    def __len__(self):
        if hasattr(self.subsetKeys, '__len__'):
            return len(self.subsetKeys)
        elif self.subsetKeys is None:
            length=0
            with h5py.File(self.dataset_path, 'r') as f:
                length=len(f.keys)
            return length
        

    def __getitem__(self,idx):
        id=self.subsetKeys[idx]
        label=self.yflag_dict[f"Breast_MRI_{str(id).zfill(3)}"]
        with h5py.File(self.dataset_path, 'r') as f:
            image=f[str(id)]['data'][:]
        
        volume = torch.tensor(image) # torch.Size([160, 229, 193])
        #volume = torch.unsqueeze(volume, 0) # add channel dimension
        #volume = volume.float()
        
        if self.transform:
            volume = self.transform(volume)
            img_min = volume.min()
            img_max = volume.max()
            volume = (volume-img_min)/(img_max-img_min)

        return volume.type(torch.float32), label

    
        

class TwoAugSelfSupervisedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_h5path,
                 yflag_dict,
                 subsetKeys=None,
                 transform = None,

                 ):
        self.dataset_path=dataset_h5path

        self.yflag_dict=yflag_dict
        self.classes=[0,1]
        self.class_to_idx={0:0,1:1} #classname -> id for network
        if subsetKeys:
            self.subsetKeys=subsetKeys
        else:
            with h5py.File(self.dataset_path, 'r') as f:
                self.subsetKeys=list(f.keys())
                self.subsetKeys.sort()
        
        #if transform is None:
        #    self.transform = lambda x: x
        #else:
        self.transform=transform
        
    def __len__(self):
        if hasattr(self.subsetKeys, '__len__'):
            return len(self.subsetKeys)
        elif self.subsetKeys is None:
            length=0
            with h5py.File(self.dataset_path, 'r') as f:
                length=len(f.keys)
            return length
        

    def __getitem__(self,idx):
        id=self.subsetKeys[idx]
        label=self.yflag_dict[f"Breast_MRI_{str(id).zfill(3)}"]
        with h5py.File(self.dataset_path, 'r') as f:
            image=f[str(id)]['data'][:]
        
        volume = torch.tensor(image) # torch.Size([160, 229, 193])
        #volume = torch.unsqueeze(volume, 0) # add channel dimension
        #volume = volume.float()
        volumes=[]

        if self.transform:
            for i in range(2):
                newVolume = self.transform(volume)
                img_min = newVolume.min()
                img_max = newVolume.max()
                newVolume  = (newVolume -img_min)/(img_max-img_min)
                volumes.append(newVolume)
        else:
            volumes=[volume,volume]
            

        return volumes[0].type(torch.float32),volumes[1].type(torch.float32), label



def construct_dataLR(
        dataset_h5path,
        k_fold = 5,
        test_p=.2,
        val_p=.05,
        seed=42,
        
    ):

    with h5py.File(dataset_h5path, 'r') as f:
        validPatients=list(f.keys())
    
    


def construct_data(
        dataset_h5path,
        yflag_df,
        yLabelColumn='StagingNodes',
        k_fold = 5,
        test_p=.2,
        val_p=.05,
        seed=42,
        kMeansSaveDir="kMeans.json"
    ):
    """
    k-fold and returns dataset classes for 
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, valset, testset, testset_projection 
    """
    patientNums=[int(i.split("_")[-1]) for i in yflag_df.index] #following Breast_MRI_"i".zfill(3) convention
    validPatients=[i for i in patientNums if not np.isnan(yflag_df[yLabelColumn][f"Breast_MRI_{str(i).zfill(3)}"])]
    with h5py.File(dataset_h5path, 'r') as f:
        validPatients=[i for i in validPatients if str(i) in f.keys()]
    
    np.random.seed(seed)

    #not using below right.. implementation of 1 of k folds.



    trainTestFolds=kFold_TrainTestSplit(validPatients,k=k_fold,testP=test_p)[0]
    trainValFolds=kFold_TrainTestSplit(trainTestFolds['train'],k=k_fold,testP=val_p/(1-test_p))[0]
    trainValFolds={'train':trainValFolds['train'],'val':trainValFolds['test']}
    trainTestFolds.update(trainValFolds)

    folds=trainTestFolds # keys 'train' 'test' 'val'
    if type(kMeansSaveDir)==str:
        with open(kMeansSaveDir,"w") as f:
            json.dump(folds,f)

    #
    # modify yflag_dict expression here if you want to modify the way we define classification
    yflag_dict={ind:1 if yflag_df[yLabelColumn][ind]>=1 else 0 for ind in yflag_df.index}

    trainset = TwoAugSelfSupervisedDataset(
        dataset_h5path,
                 yflag_dict,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['train'],
        )
    trainset_pretraining = TwoAugSelfSupervisedDataset(
        dataset_h5path,
                 yflag_dict,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['train'],
        )
    trainset_normal = AugSupervisedDataset(
        dataset_h5path,
                 yflag_dict,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['train_noaug'],
        )
    trainset_normal_augment = AugSupervisedDataset(
        dataset_h5path,
                 yflag_dict,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['train'],
        )
    projectset = AugSupervisedDataset(
        dataset_h5path,
                 yflag_dict,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['project_noaug'],
        )
    valset = AugSupervisedDataset(
        dataset_h5path,
                 yflag_dict,
                 subsetKeys=folds['val'],
                 transform = transforms_dic['val'],
        )
    testset = AugSupervisedDataset(
        dataset_h5path,
                 yflag_dict,
                 subsetKeys=folds['test'],
                 transform = transforms_dic['test'],
        )
    testset_projection = AugSupervisedDataset(
        dataset_h5path,
                 yflag_dict,
                 subsetKeys=folds['test'],
                 transform = transforms_dic['test_projection'],
        )
    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, valset, testset, testset_projection 



def get_dataloaders(dataset_h5path,
        yflag_df,
        yLabelColumn='StagingNodes',
        k_fold = 5,
        test_p=.2,
        val_p=.05,
        batchSize=25,
        num_workers=1,
        seed=42,):
    """
    calls get_data and returns DataLoaders
    """
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, valset, testset, testset_projection = construct_data(dataset_h5path,
        yflag_df,
        yLabelColumn=yLabelColumn,
        k_fold = k_fold,
        test_p=test_p,
        val_p=val_p,
        seed=seed,)
    
    usePins= useGPU and torch.cuda.is_available()
    to_shuffle = True
    sampler = None


    pretrain_batchsize = batchSize
    
    trainloader = torch.utils.data.DataLoader(
        dataset = trainset,
        batch_size = batchSize,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = True)
           
    trainloader_pretraining = torch.utils.data.DataLoader(
        dataset = trainset_pretraining,
        batch_size = pretrain_batchsize,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = True)
    
    trainloader_normal = torch.utils.data.DataLoader(
        dataset = trainset_normal,
        batch_size = batchSize,
        shuffle = False, 
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = True)
        
    trainloader_normal_augment = torch.utils.data.DataLoader(
        dataset = trainset_normal_augment,
        batch_size = batchSize,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = True)
    
    projectloader = torch.utils.data.DataLoader(
        dataset = projectset,
        batch_size = 1,
        shuffle = False, 
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = True)
    
    valloader = torch.utils.data.DataLoader(
        dataset = valset,
        batch_size = 1,
        shuffle = True, 
        pin_memory = usePins,
        num_workers = num_workers,                
        worker_init_fn = np.random.seed(seed),
        drop_last = False)

    testloader = torch.utils.data.DataLoader(
        dataset = testset,
        batch_size = 1,
        shuffle = False, 
        pin_memory = usePins,
        num_workers = num_workers,                
        worker_init_fn = np.random.seed(seed),
        drop_last = False)
    
    test_projectloader = torch.utils.data.DataLoader(
        dataset = testset_projection,
        batch_size = 1,
        shuffle = False, 
        pin_memory = usePins,
        num_workers = num_workers,                
        worker_init_fn = np.random.seed(seed),
        drop_last = False)

    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, valloader, testloader, test_projectloader

