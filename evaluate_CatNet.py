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
from model import generate_model
from opts import parse_opts
import utils
from spatial_transforms import *
from temporal_transforms import *
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'dataset'))
import dataset_class
from train_R3D_CatNet import iCaRL, compute_mean, feature_extractor, classfier

import warnings
# os.environ['CUDA_VISIBLE_DEVICES']='3'
warnings.filterwarnings("ignore")

args = parse_opts()

load_dir = 'models/CatNet_models/{}-{}/'.format(args.arch, args.n_frames_per_clip)
annot_dir = 'dataset'

# annot_dir = '/home/zhengwei/dataset/egogesture/FramesInMeadia'


os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
device = 'cuda:0'
if isinstance(eval(args.cuda_id), int):
    device_ids = [eval(args.cuda_id)]
else:
    device_ids = [i for i in eval(args.cuda_id)]


def test(model, load_dir, ExemplarSet_file, checkpoint_file, dataloader):
    in_features = model.module.fc.in_features
    model.module.fc = nn.Linear(in_features, 83)
    model.to(device)
    # print('Model {} and exemplar sets {} loaded'.format(checkpoint_file, ExemplarSet_file))
    ExemplarSet_file = os.path.join(load_dir, ExemplarSet_file)
    checkpoint_file = os.path.join(load_dir, checkpoint_file)
    with open(ExemplarSet_file, 'rb') as f:
        exemplar_sets = pickle.load(f)

    checkpoint = torch.load(checkpoint_file)
    # model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    print('Computing exemplar means for {} classes........'.format(len(exemplar_sets)))
    exemplar_means = compute_mean(model, exemplar_sets)

    acc_exemplar = []
    for dataloader_i in tqdm(dataloader):
        acc_exemplar_class = utils.AverageMeter()
        for data in dataloader_i:   
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True).float()
            preds = classfier(model, exemplar_means, inputs)
            acc_exemplar_class.update(utils.calculate_accuracy_ForIcarl(preds, labels))
        acc_exemplar.append(acc_exemplar_class.avg)

    acc_softmax = []
    for dataloader_i in tqdm(dataloader):
        acc_softmax_class = utils.AverageMeter()
        for data in dataloader_i:   
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            probs, logits = model(inputs)
            acc_softmax_class.update(utils.calculate_accuracy(probs, labels))
        acc_softmax.append(acc_softmax_class.avg)
    return acc_exemplar, acc_softmax




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

    net, parameters = generate_model(args)
    # checkpoint = utils.load_checkpoint(model_dir, 'resnet101-{}-task0.pth'.format(args.modality))
    # net.load_state_dict(checkpoint['state_dict'])
    net_train = deepcopy(net)
    net_train.to(device)
    net_test = deepcopy(net)
    icarl = iCaRL(net_train, 2000)




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
                                        modality = args.modality)
                                        for i in range(len(class_eval))]
    dataloader_test = [DataLoader(dataset_test[i], batch_size=args.batch_size_val, 
                            num_workers=args.num_workers,pin_memory=True)
                            for i in range(len(class_eval))]



    acc_exemplar, acc_softmax = test(net_test, load_dir, 
                                    '{}_ExemplarSet_{}.pkl'.format(args.modality, 83), 
                                    '{}-{}-{}.pth'.format(args.arch, args.modality, 83), 
                                     dataloader_test)
    print(np.mean(acc_exemplar))
    
    pdb.set_trace()
    # with open(os.path.join(save_dir, 
    #         'T_{}_exemplar.pkl'.format(args.modality)), 'wb') as f:
    #     pickle.dump(T_exemplar, f)
    # with open(os.path.join(save_dir, 
    #         'T_{}_softmax.pkl'.format(args.modality)), 'wb') as f:
    #     pickle.dump(T_softmax, f)