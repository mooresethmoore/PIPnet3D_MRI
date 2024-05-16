import os
import sys
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
#import random
from imageio import imread
#import json
import torch
torch.cuda.empty_cache()

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

current_dir = os.path.dirname(os.path.realpath('__file__'))
import utils
from utils import plot_3d_slices
from utils import set_seeds
from utils import set_device
#from utils import get_optimizer_nn
from utils import init_weights_xavier
from utils import get_patch_size
from utils import Log
data_dir = os.path.join(current_dir, 'data')

from training_pipnet_LR import get_network,get_optimizer_nn
sys.path.append(data_dir)
#from make_dataset import get_dataloaders
import make_dataset_LR
from make_dataset_LR import get_dataloaders,getAllDataloader,getAllDataset
# Construct the path to the models directory
models_dir = os.path.join(current_dir, 'models')

# Add the models directory to sys.path
sys.path.append(models_dir)
from resnet_features import video_resnet18_features
from pipnet import PIPNet,NonNegLinear
from train_model_custom import train_pipnet

#from test_model import eval_pipnet




from monai.transforms import (
    Compose,
    Resize,
    RandRotate,
    Affine,
    RandGaussianNoise,
    RandZoom,
    RepeatChannel,
)
import math
import joblib
import h5py
from importlib import reload



args={
    'seed':44,
    'experiment_folder':'data/experiment_1',
    'lr':.0001,
    'lr_net':.0001,
    'lr_block':.0001,
    'lr_class':.0001,
    'lr_backbone':.0001,
    'weight_decay':0,
    'gamma':.1,
    'step_size':1,
    'batch_size':15,
    'epochs':160,
    'epochs_pretrain':30,
    'freeze_epochs':0,
    'epochs_finetune':10,
    'num_classes':2,
    'channels':3,
    'net':"3Dresnet18",
    'num_features':0,
    'bias':False,
    'out_shape':1,
    'disable_pretrained':False,
    'optimizer':'Adam',
    'state_dict_dir_net':'',
    'log_dir':"logs/PT_tan5_backbone1en4_flipTrain",
    "dic_classes":{False:0,True:1},
    'val_split':.05,
    'test_split':.2,
    'defaultFinetune':True,
    'lr_finetune':.05,
    'flipTrain':True,
    'img_shape':[51,101,72],
    'wshape':5,
    'hshape':7,
    'dshape':7,
    'patchsize':15
}
args['checkpointFile']=checkpointFile=f"{args['log_dir']}/checkpoints/net_trained_last"

topkProtos=[26, 30, 58, 76, 80, 95, 137, 172, 189, 190, 196, 206, 211, 229, 236, 252, 278, 282, 303, 327, 347, 366, 372, 384, 424, 450, 452, 458, 479, 507, 509]
## later we want to regen these on the fly ^^^ 


channels=3
aug_prob = 1
rand_rot = 10                       # random rotation range [deg]
rand_rot_rad = rand_rot*math.pi/180 # random rotation range [rad]
rand_noise_std = 0.01               # std random Gaussian noise
rand_shift = 5                      # px random shift
min_zoom = 0.9
max_zoom = 1.1
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

downSample=3.2
lowerBound=.2
inputData=f'data/FP923_LR_avgCrop_DS{int(downSample*10)}_point{int(lowerBound*100)}Thresh.h5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#yflags=pd.read_csv("../duke/ClinicalFlags.csv",index_col=0)


dataloaders=get_dataloaders(dataset_h5path=inputData,
                            k_fold=5,
                            test_p=.2,
                            val_p=.05,
                            batchSize=args['batch_size'],
                            seed=args['seed'],
                            kMeansSaveDir="data/kMeans_DS32.json")

trainloader = dataloaders[0]
trainloader_pretraining = dataloaders[1]
trainloader_normal = dataloaders[2] 
trainloader_normal_augment = dataloaders[3]
projectloader = dataloaders[4]
valloader = dataloaders[5]
testloader = dataloaders[6] 
test_projectloader = dataloaders[7]

allData=getAllDataset(inputData)
inputKeys=allData.subsetKeys

useGPU=True
devID=0
if useGPU:
    device=torch.device(f'cuda:{devID}')
else:
    device=torch.device('cpu')

network_layers = get_network(num_classes=args['num_classes'], args=args)
feature_net = network_layers[0]
add_on_layers = network_layers[1]
pool_layer = network_layers[2]
classification_layer = network_layers[3]
num_prototypes = network_layers[4]
newFeatures=feature_net
net = PIPNet(
        num_classes = args['num_classes'],
        num_prototypes = num_prototypes,
        feature_net = newFeatures,
        args = args,
        add_on_layers = add_on_layers,
        pool_layer = pool_layer,
        classification_layer = classification_layer
        )
net = net.to(device=device)
net = nn.DataParallel(net, device_ids = [0])  

optimizer = get_optimizer_nn(net, args)
optimizer_net = optimizer[0]
optimizer_classifier = optimizer[1] 
params_to_freeze = optimizer[2] 
params_to_train = optimizer[3] 
params_backbone = optimizer[4]   

checkpoint = torch.load(checkpointFile, map_location = device)
net.load_state_dict(checkpoint['model_state_dict'], strict = True) 
net.module._multiplier.requires_grad = False
try:
    optimizer_net.load_state_dict(
        checkpoint['optimizer_net_state_dict']) 
except:
    print("optimizer failed load")


def returnInputGradient(patientKey='109_R',softmaxThresh=.5,returnNet=False):

    gradients=[]
    
    arr,label=projectloader.dataset[inputKeys[10]]
    xs=arr.unsqueeze(0).to(device)
    xs.requires_grad=True
    features = net.module._net(xs)
    proto_features = net.module._add_on(features) #does any form of gradient accrue on features here?
    proto_features=proto_features.detach().cpu()
    protoThresh=proto_features > softmaxThresh

    for proto in topkProtos:
        releventIndices=[(i,j,k) for i in range(features.shape[2]) for j in range(features.shape[3]) for k in range(features.shape[4]) if protoThresh[0][proto][i][j][k]==1]
        if len(releventIndices)>0:
            
            i,j,k=releventIndices[0]
            gradient=torch.autograd.grad(features[0][proto][i][j][k],xs,retain_graph=True)[0]
            for i,j,k in releventIndices[1:]:
                gradient+=torch.autograd.grad(features[0][proto][i][j][k],xs,retain_graph=True)[0]
            gradient=gradient.detach().cpu().numpy()
            gradients.append(gradient)
    
    gradients=np.array(gradients)
    if returnNet:
        return gradients,net 
    else:
        return gradients

print(returnInputGradient())