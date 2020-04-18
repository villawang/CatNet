'''
This script evaluates the lifelong RGB and Depth model by using 
the exemplar sets selected by RGB and Depth. Final features are 
concatenated like [RGB_feature, Depth_feature] which is 2048 * 2
 '''
import shutil
from random import randint
import argparse
import glob
import pdb
import random
import math
import time
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pickle
import numpy as np


import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset
from torchsummary import summary


from models import resnext 
from model_RGB_D import generate_model
from opts import parse_opts
import utils
from spatial_transforms import *
from temporal_transforms import *
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'dataset'))
import dataset_class
from train_R3D_CatNet import iCaRL, compute_mean, feature_extractor

import warnings
warnings.filterwarnings("ignore")

args = parse_opts()

annot_dir = 'dataset'
load_dir = 'models/CatNet_models/{}-{}/'.format(args.arch, args.n_frames_per_clip)


os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
device = 'cuda:0'
if isinstance(eval(args.cuda_id), int):
    device_ids = [eval(args.cuda_id)]
else:
    device_ids = [i for i in eval(args.cuda_id)]

def classfier_fusion(net_D, net_RGB, exemplar_means_D, exemplar_means_RGB, 
                    x_D, x_RGB):
    """Classify images by neares-means-of-exemplars
    Args:
        x: input video batch
    Returns:
        preds: Tensor of size (batch_size,)
    """        
    batch_size = x_D.size(0)

    feature_D = feature_extractor(net_D, x_D).detach().cpu()
    feature_D = feature_D / torch.norm(feature_D, p=2, dim=1, keepdim=True)
    feature_D = feature_D.unsqueeze(2) # (batch_size, feature_size, 1)
    feature_RGB = feature_extractor(net_RGB, x_RGB).detach().cpu()
    feature_RGB = feature_RGB / torch.norm(feature_RGB, p=2, dim=1, keepdim=True)
    feature_RGB = feature_RGB.unsqueeze(2) # (batch_size, feature_size, 1)

    

    exemplar_means_D = torch.stack([exemplar_means_D] * batch_size)  # (batch_size, n_classes, feature_size)
    exemplar_means_D = exemplar_means_D.transpose(1, 2)
    exemplar_means_RGB = torch.stack([exemplar_means_RGB] * batch_size)  # (batch_size, n_classes, feature_size)
    exemplar_means_RGB = exemplar_means_RGB.transpose(1, 2)

    feature_D = feature_D.expand_as(exemplar_means_D) # (batch_size, feature_size, n_classes)
    feature_RGB = feature_RGB.expand_as(exemplar_means_RGB) # (batch_size, feature_size, n_classes)

    feature_fusion = torch.cat([feature_D, feature_RGB], 1)
    exemplar_means_fusion = torch.cat([exemplar_means_D, exemplar_means_RGB], 1)

    dists = (feature_fusion - exemplar_means_fusion).pow(2).sum(1) #(batch_size, n_classes)
    _, preds = dists.min(1)
    return preds


def test(net_D, net_RGB, load_dir, ExemplarSet_D, checkpoint_file_D,
        ExemplarSet_RGB, checkpoint_file_RGB, dataloader):
    in_features = net_D.module.fc.in_features
    net_D.module.fc = nn.Linear(in_features, 83)
    net_RGB.module.fc = nn.Linear(in_features, 83)
    net_D.to(device)
    net_RGB.to(device)


    ExemplarSet_file_D = os.path.join(load_dir, ExemplarSet_D)
    ExemplarSet_file_RGB = os.path.join(load_dir, ExemplarSet_RGB)   
    checkpoint_file_D = os.path.join(load_dir, checkpoint_file_D)
    checkpoint_file_RGB = os.path.join(load_dir, checkpoint_file_RGB)
    with open(ExemplarSet_file_D, 'rb') as f:
        exemplar_sets_D = pickle.load(f)
    with open(ExemplarSet_file_RGB, 'rb') as f:
        exemplar_sets_RGB = pickle.load(f)


    checkpoint_D = torch.load(checkpoint_file_D)
    net_D.load_state_dict(checkpoint_D['state_dict'])
    checkpoint_RGB = torch.load(checkpoint_file_RGB)
    net_RGB.load_state_dict(checkpoint_RGB['state_dict'])
    net_D.eval()
    net_RGB.eval()

    print('Computing Depth exemplar means for {} classes........'.format(len(exemplar_sets_D)))
    exemplar_means_D = compute_mean(net_D, exemplar_sets_D)    
    print('Computing RGB exemplar means for {} classes........'.format(len(exemplar_sets_RGB)))
    exemplar_means_RGB = compute_mean(net_RGB, exemplar_sets_RGB)

    acc_exemplar = []
    acc_exemplar_class = utils.AverageMeter()
    for dataloader_i in tqdm(dataloader):
        acc_exemplar_class.reset()
        for data in dataloader_i:   
            rgb, depth, labels = data
            depth = depth.to(device, non_blocking=True).float()  
            rgb = rgb.to(device, non_blocking=True).float()
            preds = classfier_fusion(net_D, net_RGB, 
                                    exemplar_means_D, exemplar_means_RGB, 
                                    depth, rgb)       
            acc_exemplar_class.update(utils.calculate_accuracy_ForIcarl(preds, labels))
        acc_exemplar.append(acc_exemplar_class.avg) 
    return acc_exemplar



if __name__ == '__main__':
    # activitynet mean value
    mean = [114.7, 107.7, 99.4]

    norm_method = Normalize(mean, [1, 1, 1])

    trans_test = Compose([
                Scale([112,112]),
                CenterCrop([112, 112]),
                ToTensor(1), norm_method
                ])

    temporal_transform_test = Compose([
            TemporalCenterCrop(args.n_frames_per_clip)
            ])  


    modality = 'Depth'
    net_D, parameters_D = generate_model(args, modality)
    print('\n')
    modality = 'RGB'
    net_RGB, parameters_RGB = generate_model(args, modality)



    class_init = [i for i in range(1, 41)]
    class_eval = [class_init]
    class_step = 5 # incremental steps for new class
    # evaluate the final model for each task
    for task_i, incremenral_class in enumerate(range(41, 84, class_step)):
        if incremenral_class == range(41, 84, class_step)[-1]:
            class_id2 = [i for i in range(incremenral_class, 84)]
        else:
            class_id2 = [i for i in range(incremenral_class, incremenral_class+class_step)]
        class_eval.append(class_id2)

    dataset_test = [dataset_class.dataset_video_class(annot_dir, 'test', 
                                        class_id = class_eval[i],
                                        n_frames_per_clip=args.n_frames_per_clip,
                                        img_size=(args.w, args.h), 
                                        reverse=False, transform=trans_test,
                                        temporal_transform = temporal_transform_test,
                                        modality = None)
                                        for i in range(len(class_eval))]
    dataloader_test = [DataLoader(dataset_test[i], batch_size=args.batch_size_val, 
                            num_workers=args.num_workers,pin_memory=True)
                            for i in range(len(class_eval))]

    acc_exemplar = test(net_D, net_RGB, load_dir, 
                        'Depth_ExemplarSet_83.pkl', 
                        '{}-{}-{}.pth'.format(args.arch, 'Depth', 83), 
                        'RGB_ExemplarSet_83.pkl', 
                        '{}-{}-{}.pth'.format(args.arch, 'RGB', 83),  
                        dataloader_test)
    print(np.mean(acc_exemplar))
    pdb.set_trace()
