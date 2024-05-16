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
import json


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
                 subsetKeys=None,
                 transform = None,
                 flipFlag=False,
                 flipProb=.5,
                 ):
        self.dataset_path=dataset_h5path

        self.classes=[0,1]
        self.class_to_idx={0:0,1:1} #classname -> id for network
        if subsetKeys:
            self.subsetKeys=subsetKeys  ## expecting ["19_L","23_R",...]
        else:
            with h5py.File(self.dataset_path, 'r') as f:
                self.subsetKeys=list(f.keys())
                self.subsetKeys.sort(key=lambda x:int(x.split("_")[0]))
            subsetLR=[]
            for i in self.subsetKeys:
                subsetLR.append(f"{i}_L")
                subsetLR.append(f"{i}_R")
            self.subsetKeys=subsetLR
                
        
        #if transform is None:
        #    self.transform = lambda x: x
        #else:
        self.transform=transform
        self.flipFlag=flipFlag
        self.flipProb=flipProb
        
    def __len__(self):
        if hasattr(self.subsetKeys, '__len__'):
            return len(self.subsetKeys)
        elif self.subsetKeys is None:
            length=0
            with h5py.File(self.dataset_path, 'r') as f:
                length=len(f.keys)
            return length
        

    def __getitem__(self,idx):
        if type(idx)==str:
            key,subKey=idx.split("_")
        else:
            id=self.subsetKeys[idx]
            #label=self.yflag_dict[f"Breast_MRI_{str(id).zfill(3)}"]
            key,subKey=id.split("_")
        assert subKey in {"L","R"}
        with h5py.File(self.dataset_path, 'r') as f:
            image=f[key][subKey][:]
            if subKey=="L":
                label=f[key].attrs['LRflag'][0]
            else:
                label=f[key].attrs['LRflag'][1]
        
        if self.flipFlag:
            if random.random()<=self.flipProb:
                image=image[::-1,:,:].copy()
            if random.random()<=self.flipProb:
                image=image[:,::-1,:].copy()
            if random.random()<=self.flipProb:
                image=image[:,:,::-1].copy()

        volume = torch.tensor(image) # torch.Size([160, 229, 193])
        volume = torch.unsqueeze(volume, 0) # add channel dimension
        #volume = volume.float()
        


        if self.transform:
            volume = self.transform(volume)
            img_min = volume.min()
            img_max = volume.max()
            volume = (volume-img_min)/(img_max-img_min)

        return volume.type(torch.float32), torch.tensor(label).type(torch.LongTensor)

    
        

class TwoAugSelfSupervisedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_h5path,
                 subsetKeys=None,
                 transform = None,
                 flipFlag=False,
                 flipProb=.5,
                 ):
        self.dataset_path=dataset_h5path

        self.classes=[0,1]
        self.class_to_idx={0:0,1:1} #classname -> id for network
        if subsetKeys:
            self.subsetKeys=subsetKeys  ## expecting ["19_L","23_R",...]
        else:
            with h5py.File(self.dataset_path, 'r') as f:
                self.subsetKeys=list(f.keys())
                self.subsetKeys.sort(key=lambda x:int(x.split("_")[0]))
            subsetLR=[]
            for i in self.subsetKeys:
                subsetLR.append(f"{i}_L")
                subsetLR.append(f"{i}_R")
            self.subsetKeys=subsetLR
                
        
        #if transform is None:
        #    self.transform = lambda x: x
        #else:
        self.transform=transform
        self.flipFlag=flipFlag
        self.flipProb=flipProb
        
    def __len__(self):
        if hasattr(self.subsetKeys, '__len__'):
            return len(self.subsetKeys)
        elif self.subsetKeys is None:
            length=0
            with h5py.File(self.dataset_path, 'r') as f:
                length=len(f.keys)
            return length
        

    def __getitem__(self,idx):
        if type(idx)==str:
            key,subKey=idx.split("_")
        else:
            id=self.subsetKeys[idx]
            #label=self.yflag_dict[f"Breast_MRI_{str(id).zfill(3)}"]
            key,subKey=id.split("_")
        assert subKey in {"L","R"}
        with h5py.File(self.dataset_path, 'r') as f:
            image=f[key][subKey][:]
            if subKey=="L":
                label=f[key].attrs['LRflag'][0]
            else:
                label=f[key].attrs['LRflag'][1]
        
        if self.flipFlag:
            if random.random()<=self.flipProb:
                image=image[::-1,:,:].copy()
            if random.random()<=self.flipProb:
                image=image[:,::-1,:].copy()
            if random.random()<=self.flipProb:
                image=image[:,:,::-1].copy()

        volume = torch.tensor(image) # torch.Size([160, 229, 193])
        volume = torch.unsqueeze(volume, 0) # add channel dimension
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
            

        return volumes[0].type(torch.float32),volumes[1].type(torch.float32), torch.tensor(label).type(torch.LongTensor)

def getAllDataset(dataset_h5path):
    return AugSupervisedDataset(dataset_h5path=dataset_h5path) #assuming this is all we need

def getAllDataloader(dataset_h5path,
                    batch_size = 1,
                    to_shuffle = True,
                    sampler = None,
                    pin_memory = torch.cuda.is_available(),
                    num_workers = 1,
                    worker_init_fn = lambda: np.random.seed(42),
                    drop_last = False):
    return torch.utils.data.DataLoader(
        dataset = getAllDataset(dataset_h5path),
        batch_size = batch_size,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = pin_memory,
        num_workers = num_workers,
        worker_init_fn = worker_init_fn,
        drop_last = drop_last)


def construct_data(
        dataset_h5path,
        k_fold = 5,
        test_p=.2,
        val_p=.05,
        seed=42,
        kMeansSaveDir=None,
        flipTrain=True
    ):

    with h5py.File(dataset_h5path, 'r') as f:
        validPatients=list(f.keys())
    subsetLR=[]
    for i in validPatients:
        subsetLR.append(f"{i}_L")
        subsetLR.append(f"{i}_R")
    validPatients=subsetLR
    
    np.random.seed(seed)

    trainTestFolds=kFold_TrainTestSplit(validPatients,k=k_fold,testP=test_p)[0]
    trainValFolds=kFold_TrainTestSplit(trainTestFolds['train'],k=k_fold,testP=val_p/(1-test_p))[0]
    trainValFolds={'train':trainValFolds['train'],'val':trainValFolds['test']}
    trainTestFolds.update(trainValFolds)

    folds=trainTestFolds # keys 'train' 'test' 'val'
    
    if type(kMeansSaveDir)==str:
        with open(kMeansSaveDir,"w") as f:
            json.dump(folds,f)

    trainset = TwoAugSelfSupervisedDataset(
        dataset_h5path,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['train'],
                 flipFlag=flipTrain,
                 flipProb=.5
        )
    trainset_pretraining = TwoAugSelfSupervisedDataset(
        dataset_h5path,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['train'],
                 flipFlag=flipTrain,
                 flipProb=.5
        )
    trainset_normal = AugSupervisedDataset(
        dataset_h5path,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['train_noaug'],
        )
    trainset_normal_augment = AugSupervisedDataset(
        dataset_h5path,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['train'],
                 flipFlag=flipTrain,
                 flipProb=.5
        )
    projectset = AugSupervisedDataset(
        dataset_h5path,
                 subsetKeys=folds['train'],
                 transform = transforms_dic['project_noaug'],
        )
    valset = AugSupervisedDataset(
        dataset_h5path,
                 subsetKeys=folds['val'],
                 transform = transforms_dic['val'],
        )
    testset = AugSupervisedDataset(
        dataset_h5path,
                 subsetKeys=folds['test'],
                 transform = transforms_dic['test'],
        )
    testset_projection = AugSupervisedDataset(
        dataset_h5path,
                 subsetKeys=folds['test'],
                 transform = transforms_dic['test_projection'],
        )
    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, valset, testset, testset_projection 



def get_dataloaders(dataset_h5path,
        k_fold = 5,
        test_p=.2,
        val_p=.05,
        batchSize=15,
        num_workers=1,
        seed=42,
        kMeansSaveDir=None,
        flipTrain=True
        ):
    """
    calls get_data and returns DataLoaders
    """
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, valset, testset, testset_projection = construct_data(dataset_h5path,
        k_fold = k_fold,
        test_p=test_p,
        val_p=val_p,
        seed=seed,
        kMeansSaveDir=kMeansSaveDir,
        flipTrain=flipTrain)
    
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
        drop_last = False)
           
    trainloader_pretraining = torch.utils.data.DataLoader(
        dataset = trainset_pretraining,
        batch_size = pretrain_batchsize,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = False)
    
    trainloader_normal = torch.utils.data.DataLoader(
        dataset = trainset_normal,
        batch_size = batchSize,
        shuffle = False, 
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = False)
        
    trainloader_normal_augment = torch.utils.data.DataLoader(
        dataset = trainset_normal_augment,
        batch_size = batchSize,
        shuffle = to_shuffle,
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = False)
    
    projectloader = torch.utils.data.DataLoader(
        dataset = projectset,
        batch_size = 1,
        shuffle = False, 
        sampler = sampler,
        pin_memory = usePins,
        num_workers = num_workers,
        worker_init_fn = np.random.seed(seed),
        drop_last = False)
    
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

