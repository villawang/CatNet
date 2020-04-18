import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm, trange
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
import sys
sys.path.append(os.path.join(os.getcwd(), 'dataset'))


import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchsummary import summary


from models import resnext 
from model import generate_model
from opts import parse_opts
import utils
from spatial_transforms import *
from temporal_transforms import *
import dataset_class

import warnings
import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
warnings.filterwarnings("ignore")

annot_dir = './dataset'
save_dir = 'output_task0'
model_test_dir = 'models/task0_model'


args = parse_opts()
os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
device = 'cuda:0'
if isinstance(args.cuda_id, list):
    device_ids = [i for i in eval(args.cuda_id)]
else:
    device_ids = [eval(args.cuda_id)]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)



def forward(model, data):
    rgbs, depths, labels = data
    if args.modality == 'RGB':
        inputs = rgbs.to(device, non_blocking=True).float()
    elif args.modality == 'Depth':
        inputs = depths.to(device, non_blocking=True).float()
    elif args.modality == 'RGB-D':
        inputs = torch.cat((rgbs, depths), 1).to(device, non_blocking=True).float()
    probs, logits = model(inputs)
    labels = labels.to(device, non_blocking=True).long()
    return probs, logits, labels

def model_test(model, save_dir, filename, dataloader, num_class):
    model.module.fc = nn.Linear(model.module.fc.in_features, num_class)
    model.module.fc.to(device)
    checkpoint = utils.load_checkpoint(save_dir, filename)
    # model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('Evaluating for model {}........'.format(filename))
    acc = utils.AverageMeter()
    for data in tqdm(dataloader):
        probs, logits, labels = forward(model, data)
        acc.update(utils.calculate_accuracy(probs, labels))
    print('val_acc:{:.3f}'.format(acc.avg))


def model_train(model, save_dir, dataloader_train, dataloader_val):
    model.train()
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()

    # determine optimizer
    fc_lr_layers = list(map(id, model.module.fc.parameters()))
    pretrained_lr_layers = [p for p in model.parameters() 
                            if id(p) not in fc_lr_layers and p.requires_grad==True]
    # pretrained_lr_layers = filter(lambda p: 
    #                               id(p) not in fc_lr_layers, model.parameters())
    # optimizer = torch.optim.SGD([
    #     {"params": model.module.fc.parameters()},
    #     {"params": pretrained_lr_layers, "lr": 1e-4, 'weight_decay':1e-3}
    # ], lr=1e-3, momentum=0.9, weight_decay=1e-3)    
    learning_rate = 1e-3
    # lr_steps = [10,15,20]
    lr_steps = [25]
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9, weight_decay=1e-3)   

    train_logger = utils.Logger(os.path.join(save_dir, '{}-{}-{}.log'.format(args.arch, args.n_frames_per_clip, args.modality)),
                                ['step', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                                'lr_feature', 'lr_fc'])
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    

    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()


    step = 0
    for epoch in trange(num_epochs):  # loop over the dataset multiple times
        train_loss.reset()
        train_acc.reset()
        for data in dataloader_train:
            probs, outputs, labels = forward(model, data)
            optimizer.zero_grad()
            loss_ = criterion(outputs, labels)
            loss_.backward()
            optimizer.step()
            train_loss.update(loss_.item())
            train_acc.update(utils.calculate_accuracy(probs, labels))
            if step % 100 == 0:
                val_loss.reset()
                val_acc.reset()
                model.eval()
                for data_val in dataloader_val:
                    probs_val, outputs_val, labels_val = forward(model, data_val)
                    val_loss_ = criterion(outputs_val, labels_val)
                    val_loss.update(val_loss_.item())
                    val_acc.update(utils.calculate_accuracy(probs_val, labels_val))
                model.train()
                print('epoch{}/{} train_acc:{:.3f} train_loss:{:.3f} val_acc:{:.3f} val_loss:{:.3f}'.format(
                    epoch + 1, num_epochs,
                    train_acc.val, train_loss.val,
                    val_acc.avg, val_loss.avg
                    ))
                train_logger.log({
                    'step': step,
                    'train_loss': train_loss.val,
                    'train_acc': train_acc.val,
                    'val_loss': val_loss.avg,
                    'val_acc': val_acc.avg,
                    # 'lr_feature': optimizer.param_groups[1]['lr'],
                    'lr_feature': 0,
                    'lr_fc': optimizer.param_groups[0]['lr']
                })
            step += 1     
        utils.save_checkpoint(model, optimizer, step, save_dir,
                                '{}-{}-{}.pth'.format(args.arch, args.n_frames_per_clip, args.modality))
        # scheduler.step()
        utils.adjust_learning_rate(learning_rate, optimizer, epoch, lr_steps)



if __name__ == '__main__':
    # keep shuffling be constant every time
    seed = 1
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # activitynet mean value
    mean = [114.7, 107.7, 99.4]
    # std = [38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255] 

    # kinetics
    # [110.6, 103.1, 96.2]

    # norm_method1 = Normalize(mean, std)
    norm_method = Normalize(mean, [1, 1, 1])
    # scales = [1, 1/(math.pow(2, .25)), 1/(math.pow(2, .75)), 1/2]
    scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        scales.append(scales[-1] * args.scale_step)

    trans_train = Compose([
                Scale([112,112]),
                MultiScaleRandomCrop(scales, [112,112]),
                SpatialElasticDisplacement(),
                # RandomHorizontalFlip(),
                ToTensor(1), norm_method
                ])
    temporal_transform_train = Compose([
            TemporalRandomCrop(args.n_frames_per_clip)
            ])    
    trans_test = Compose([
                Scale([112,112]),
                CenterCrop([112, 112]),
                ToTensor(1), norm_method
                ])
    temporal_transform_test = Compose([
            TemporalCenterCrop(args.n_frames_per_clip)
            ])

    # load dataset
    if args.is_train:
        print('Loading training data.....')
        class_id1 = [i for i in range(1, 41)]
        dataset_train = dataset_class.dataset_video_class(annot_dir, 'train_plus_val',
                                            class_id = class_id1,
                                            n_frames_per_clip=args.n_frames_per_clip,
                                            img_size=(args.w, args.h),
                                            reverse=False, transform=trans_train,
                                            temporal_transform = temporal_transform_train)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                        shuffle=True, 
                                        num_workers=args.num_workers, pin_memory=True)

        print('\n')
        print('Loading validating data.....')
        dataset_val = dataset_class.dataset_video_class(annot_dir, 'test', 
                                            class_id = class_id1,
                                            n_frames_per_clip=args.n_frames_per_clip,
                                            img_size=(args.w, args.h), 
                                            reverse=False, transform=trans_test,
                                            temporal_transform = temporal_transform_test)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size_val, 
                                    num_workers=args.num_workers,pin_memory=True)

        
    else:
        print('Loading testing data.....')
        class_id1 = [i for i in range(1, 41)]
        dataset_test = dataset_class.dataset_video_class(annot_dir, 'test', 
                                            class_id = class_id1,
                                            n_frames_per_clip=args.n_frames_per_clip,
                                            img_size=(args.w, args.h), 
                                            reverse=False, transform=trans_test,
                                            temporal_transform = temporal_transform_test)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size_val, 
                                    num_workers=args.num_workers,pin_memory=True)


    model, parameters = generate_model(args)
    model.to(device)
    if args.is_train:
        if args.modality == 'RGB':
            summary(model, (3,args.n_frames_per_clip,112,112))
        elif args.modality == 'Depth':
            summary(model, (1,args.n_frames_per_clip,112,112))
        elif args.modality == 'RGB-D':
            summary(model, (4,args.n_frames_per_clip,112,112))
        model_train(model, save_dir, dataloader_train, dataloader_val)
        pdb.set_trace()
    else:
        model_test(model, model_test_dir, '{}-{}-{}.pth'.format(args.arch, args.n_frames_per_clip, args.modality), dataloader_test, args.n_finetune_classes)
        pdb.set_trace()