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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.getcwd(), 'dataset'))


import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset
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
args = parse_opts()

annot_dir = 'dataset'
save_dir = 'output_CatNet/{}-{}'.format(args.arch, args.n_frames_per_clip)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


model_dir = 'models/task0_model'


os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
device = 'cuda:0'
if isinstance(eval(args.cuda_id), int):
    device_ids = [eval(args.cuda_id)]
else:
    device_ids = [i for i in eval(args.cuda_id)]




class iCaRL(object):
    def __init__(self, net, K):
        self.net = net
        self.K = K # number of cached exemplar sets for all observed classes
        self.exemplar_sets = [] # list contains exemplar sets for different classes
        self.exemplar_labels = []
        self.criterion = nn.CrossEntropyLoss()


    def increment_classes(self, num_AddedClasses):
        """Add n classes in the final fc layer"""
        in_features = self.net.module.fc.in_features
        out_features = self.net.module.fc.out_features
        weight = self.net.module.fc.weight.data
        # self.net.module.fc = nn.Linear(in_features, out_features+num_AddedClasses, bias=False)
        self.net.module.fc = nn.Linear(in_features, out_features+num_AddedClasses)
        self.net.module.fc.weight.data[:out_features] = weight
        self.net.module.fc.to(device)


    def update_representation(self, exemplar_dataset, new_class_dataset):
        print('Updating representation........')
        # DO NOT CHANGE batch size HERE!!!!!!!!!!!!!!!
        exemplar_loader = torch.utils.data.DataLoader(exemplar_dataset, batch_size=1, num_workers=args.num_workers)
        # exemplar_dataset_UpdateLables contains stored network predicted label and new class label
        # replace the ground truth label with the predicted labels by the stored model
        exemplar_dataset_UpdateLables = []
        for data in exemplar_loader:
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True).float()
            probs, logits = self.net(inputs)
            update_labels = int(probs.max(1)[1].detach().cpu().item())
            exemplar_dataset_UpdateLables.append((inputs[0].detach().cpu(), update_labels))
        D_dataset = ConcatDataset([dataset_class.make_dataset_instance(exemplar_dataset_UpdateLables)]+new_class_dataset)
        D_loader = DataLoader(D_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True) 
        del inputs
        torch.cuda.empty_cache()
        return D_loader

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
            self.exemplar_labels[y] = self.exemplar_labels[y][:m]

    def train(self, new_class_train_dataset, new_class_test_dataset, added_class_ids, dataloader_val):
        # representation step
        exemplar_dataset = self.combine_dataset_with_exemplars(self.exemplar_sets, self.exemplar_labels)
        self.net.eval()
        D_loader = self.update_representation(exemplar_dataset, new_class_train_dataset)
        self.net.train()
        self.increment_classes(len(added_class_ids))

        # training
        learning_rate = 1e-3
        lr_steps = [6]
        num_epochs = 12
        step = 0


        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, 
                                    weight_decay=1e-3, dampening=0.9)   


        train_loss = utils.AverageMeter()
        train_acc = utils.AverageMeter()
        val_loss = utils.AverageMeter()
        val_acc = utils.AverageMeter()

        print('Start training.......')
        for epoch in trange(num_epochs):
            train_loss.reset()
            train_acc.reset()
            for data in D_loader:
                inputs, labels = data
                inputs = inputs.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True)
                probs, logits = self.net(inputs)
                self.optimizer.zero_grad()
                loss_ = self.criterion(logits, labels)
                loss_.backward()
                self.optimizer.step()
                train_loss.update(loss_.item())
                if step % 100 == 0:
                    loss_val, acc_val_old, acc_val_new, acc_val = self.val_model(dataloader_val)
                    train_logger.log({
                        'num_classes': self.net.module.fc.out_features,
                        'train_loss': train_loss.val,
                        'val_loss': loss_val,
                        'val_acc_old': acc_val_old,
                        'val_acc_new': acc_val_new,
                        'val_acc': acc_val,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                    print('train loss: {:.3f}'.format(train_loss.avg), 
                          'val loss: {:.3f}'.format(loss_val), 
                          'acc val old: {:.3f}'.format(acc_val_old),
                          'acc val new: {:.3f}'.format(acc_val_new),
                          'acc val mean: {:.3f}'.format(acc_val))
                step += 1 
            utils.save_checkpoint(self.net, self.optimizer, step, save_dir,
                                  '{}-{}-{}.pth'.format(args.arch, args.modality, self.net.module.fc.out_features))
            utils.adjust_learning_rate(learning_rate, self.optimizer, epoch, lr_steps)  


        # print('Updating the exemplar sets.......')
        # m = int(self.K / self.net.module.fc.out_features)
        # shrink the exemplar sets for old class
        # self.reduce_exemplar_sets(m)

        
        new_class_dataloader = []
        ###########################Dirty way to construct the dataloader list. No idea why the second method does not work......#####################################################
        for i in range(len(new_class_test_dataset)):
            new_class_dataloader.append(DataLoader(new_class_test_dataset[i], batch_size=args.batch_size, 
                                                    num_workers=args.num_workers))
        # new_class_dataloader = [DataLoader(new_class_dataset[i], batch_size=16, 
        #                                                     num_workers=4) for i in range(len(new_class_dataset))] 
        #############################################################################################
        # attach the new class representation to exemplar sets
        self.net.eval()
        for dataloader_class in new_class_dataloader:
            self.construct_exemplar_set(dataloader_class)
        self.net.train()
        # need to be saved for testing phase
        # exemplar_dataset = self.combine_dataset_with_exemplars(self.exemplar_sets, self.exemplar_labels)
        with open(os.path.join(save_dir, 
        '{}_ExemplarSet_{}.pkl'.format(args.modality, self.net.module.fc.out_features)), 'wb') as f:
            pickle.dump(self.exemplar_sets, f)
        del inputs, labels, probs, logits
        torch.cuda.empty_cache()


    def val_model(self, dataloader):
        self.net.eval()
        acc = utils.AverageMeter()
        acc_class = utils.AverageMeter()
        acc_class_cache = []
        loss_val = utils.AverageMeter()
        for class_i, dataloader_i in enumerate(dataloader):
            acc_class.reset()
            for data in dataloader_i:   
                inputs, labels = data
                inputs = inputs.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True)
                probs, logits = self.net(inputs)
                val_loss_ = self.criterion(logits, labels)
                acc.update(utils.calculate_accuracy(probs, labels))
                acc_class.update(utils.calculate_accuracy(probs, labels))
                loss_val.update(val_loss_.item())
            acc_class_cache.append(acc_class.avg)
        self.net.train()
        return loss_val.avg, acc_class_cache[0], acc_class_cache[1], acc.avg


    # function prepare exemplar dataloader for update_representation
    def combine_dataset_with_exemplars(self, exemplar_sets, exemplar_labels):
        exemplar_dataset = []
        for y, P_y in enumerate(exemplar_sets):
            for i in range(P_y.size(0)): 
                exemplar_dataset.append((P_y[i], exemplar_labels[y][i]))
        return exemplar_dataset

    def construct_exemplar_set(self, dataloader):
        """Construct an exemplar set for videos frames set
        Args:
            dataloader: dataloader containing videos frames of a class
        """
        # m = int(self.K / self.net.module.fc.out_features)
        m = int(self.K / 40)
        # assert m <= len(dataloader) * 16
        features = []
        frames = []
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            feature = feature_extractor(self.net, inputs).detach().cpu()
            feature = feature/torch.norm(feature, p=2, dim=1, keepdim=True)
            features.append(feature)
            frames.append(inputs.detach().cpu())
        features = torch.cat(features)
        class_mean = torch.mean(features, dim=0, keepdim=True)
        class_mean = class_mean/torch.norm(class_mean, p=2, dim=1, keepdim=True)
        frames = torch.cat(frames)


        exemplar_set = []
        exemplar_features = [] # list of Variables of shape (feature_size,)
        for k in range(m):
            if k == 1:
                S = torch.cat(exemplar_features).view(1,-1)
            elif k == 0:
                S = 0.
            else:
                S = torch.sum(torch.stack(exemplar_features), dim=0, keepdim=True)               
            phi = features
            mu = class_mean           
            mu_p = 1/(k+1) * (phi + S)
            selected_indice = torch.argmin(torch.sqrt(torch.sum((mu - mu_p)**2, dim=1))).item()
            exemplar_set.append(frames[selected_indice:selected_indice+1])
            exemplar_features.append(features[selected_indice])
        exemplar_set = torch.cat(exemplar_set)
        exemplar_features = torch.cat(exemplar_features)
        exemplar_label = torch.tensor([labels[0]]*m).long()
        self.exemplar_sets.append(exemplar_set)   
        self.exemplar_labels.append(exemplar_label)
        del inputs
        torch.cuda.empty_cache()

     
def compute_mean(net, exemplar_sets):  
    # prior knowledge of the statistics in the exemplar dataset
    exemplar_means = []
    for P_y in exemplar_sets:
        loader = torch.utils.data.DataLoader(P_y, batch_size=args.batch_size, num_workers=args.num_workers)
        features = []
        for inputs in loader:
            feature = feature_extractor(net, inputs.to(device,non_blocking=True).float()).detach().cpu()
            feature = feature / torch.norm(feature, p=2, dim=1, keepdim=True) # batch_size * feature_size
            features.append(feature)
        features = torch.cat(features)  # batch_size * feature_size
        mu_y = torch.mean(features, dim=0, keepdim=True)  # 1 * feature_size
        mu_y = mu_y/torch.norm(mu_y, p=2, dim=1, keepdim=True)
        exemplar_means.append(mu_y) 
    # save gpu memory
    del feature
    torch.cuda.empty_cache()  
    exemplar_means = torch.cat(exemplar_means)  # (n_classes, feature_size)
    return exemplar_means


def feature_extractor(net, x):
    """Classify images by neares-means-of-exemplars
    Args:
        x: input video batch
    Returns:
        feature: Tensor of extracted features (batch_size,)
    """    
    net_FeatureExtractor = nn.Sequential(*list([i for i in net.children()][0].children())[:-2])
    feature = net_FeatureExtractor(x)
    feature = feature.view(feature.size(0), -1)
    return feature


def test(model, load_dir, ExemplarSet_file, checkpoint_file, dataloader, class_id1, class_id2):
    in_features = model.module.fc.in_features
    out_features = model.module.fc.out_features
    # model.module.fc = nn.Linear(in_features, len(class_id1)+len(class_id2), bias=False)
    model.module.fc = nn.Linear(in_features, len(class_id1)+len(class_id2))
    model.to(device)
    print('Start testing for class {}.....'.format(range(class_id1[0], class_id2[-1])))
    print('Model {} and exemplar sets {} loaded'.format(checkpoint_file, ExemplarSet_file))
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

    acc = utils.AverageMeter()
    acc_class = utils.AverageMeter()
    acc_class_cache = []
    for class_i, dataloader_i in enumerate(dataloader):
        acc_class.reset()
        for data in dataloader_i:   
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True).float()
            preds = classfier(model, exemplar_means, inputs)
            acc.update(utils.calculate_accuracy_ForIcarl(preds, labels))
            acc_class.update(utils.calculate_accuracy_ForIcarl(preds, labels))
        acc_class_cache.append(acc_class.avg)
    print('Accuracy for old classes:')
    print(acc_class_cache[0])
    print('Accuracy for new classes:')
    print(acc_class_cache[1])
    print('Mean accuracy')
    print(acc.avg)


    acc.reset()
    acc_class.reset()
    acc_class_cache = []
    for class_i, dataloader_i in enumerate(dataloader):
        acc_class.reset()
        for data in dataloader_i:   
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            probs, logits = model(inputs)
            acc.update(utils.calculate_accuracy(probs, labels))
            acc_class.update(utils.calculate_accuracy(probs, labels))
        acc_class_cache.append(acc_class.avg)
    print('\n')
    print('Accuracy for old classes:')
    print(acc_class_cache[0])
    print('Accuracy for new classes:')
    print(acc_class_cache[1])
    print('Mean accuracy')
    print(acc.avg)
    del inputs, labels, probs, logits, model
    torch.cuda.empty_cache()


def classfier(net, exemplar_means, x):
    """Classify images by neares-means-of-exemplars
    Args:
        x: input video batch
    Returns:
        preds: Tensor of size (batch_size,)
    """        
    batch_size = x.size(0)

    # feature = self.feature_extractor(x).detach().cpu() # (batch_size, feature_size)
    # feature = feature / torch.norm(feature, p=2, dim=1, keepdim=True)

    # feature_extractor = nn.Sequential(*list([i for i in net.children()][0].children())[:-2])
    feature = feature_extractor(net, x).detach().cpu()
    feature = feature / torch.norm(feature, p=2, dim=1, keepdim=True)
    feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)

    exemplar_means = torch.stack([exemplar_means] * batch_size)  # (batch_size, n_classes, feature_size)
    exemplar_means = exemplar_means.transpose(1, 2)
    feature = feature.expand_as(exemplar_means) # (batch_size, feature_size, n_classes)
    dists = (feature - exemplar_means).pow(2).sum(1) #(batch_size, n_classes)
    _, preds = dists.min(1)
    return preds

def get_confusion_matrix(model, load_dir, ExemplarSet_file, checkpoint_file, dataloader, class_id):
    in_features = model.module.fc.in_features
    out_features = model.module.fc.out_features
    # model.module.fc = nn.Linear(in_features, len(class_id1)+len(class_id2), bias=False)
    model.module.fc = nn.Linear(in_features, len(class_id))
    model.to(device)
    print('Model {} and exemplar sets {} loaded'.format(checkpoint_file, ExemplarSet_file))
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
    labels_all = []
    preds_exemplar_all = []
    preds_softmax_all = []
    # for class_i, dataloader_i in enumerate(dataloader):
    for data in tqdm(dataloader):   
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)
        preds_exemplar = classfier(model, exemplar_means, inputs)
        probs, logits = model(inputs)
        pred_softmax = probs.max(1)[1]
        labels_all.append(labels.detach().cpu())
        preds_exemplar_all.append(preds_exemplar.detach().cpu())
        preds_softmax_all.append(pred_softmax.detach().cpu())
    preds_exemplar_all = torch.cat(preds_exemplar_all)
    preds_softmax_all = torch.cat(preds_softmax_all)
    labels_all = torch.cat(labels_all)
    preds_exemplar_all = preds_exemplar_all.numpy()
    preds_softmax_all = preds_softmax_all.numpy()
    labels_all = labels_all.numpy()
    C_exemplar = confusion_matrix(labels_all,preds_exemplar_all)
    C_softmax = confusion_matrix(labels_all,preds_softmax_all)
    plt.figure()
    plt.matshow(C_exemplar)
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(os.path.join(load_dir, 'confusion_matrix_exemplar.jpg'))


    plt.figure()
    plt.matshow(C_softmax)
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(os.path.join(load_dir, 'confusion_matrix_softmax.jpg'))





if __name__ == '__main__':
    # keep shuffling be constant every time
    seed = 1
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # activitynet mean value
    mean = [114.7, 107.7, 99.4]

    norm_method = Normalize(mean, [1, 1, 1])

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

    net, parameters = generate_model(args)
    checkpoint = utils.load_checkpoint(model_dir, '{}-{}-{}.pth'.format(args.arch, args.n_frames_per_clip, args.modality))
    net.load_state_dict(checkpoint['state_dict'])
    # # # set fine tune parameters: Conv5_x and fc layer from original paper
    # for param in net.module.parameters():
    #     param.requires_grad = False
    # for named_child in net.module.named_children():
    #     if named_child[0] == 'fc' or named_child[0] == 'layer4' or named_child[0] == 'layer3':
    #     # if named_child[0] == 'fc':
    #         for param in named_child[1].parameters():
    #             param.requires_grad = True

    net_train = deepcopy(net)
    net_train.to(device)
    net_test = deepcopy(net)
    icarl = iCaRL(net_train, 2000)


    # load dataset
    if args.is_train:
        train_logger = utils.Logger(os.path.join(save_dir, '{}.log'.format(args.modality)),
                                    ['num_classes', 'train_loss', 'val_loss', 
                                    'val_acc_old', 'val_acc_new','val_acc', 'lr'])
        class_id1 = [i for i in range(1, 41)] # initial learned class 

        print('Preparing initial exemplar sets........')
        print('Loading the initial class training data..... class {}'.format(range(class_id1[0], class_id1[-1])))
        # class_id1 dataloader for creating the exemplar set  
        dataset_init = [dataset_class.dataset_video_class(annot_dir, 'train_plus_val',
                                            class_id = [class_id1[i]],
                                            n_frames_per_clip=args.n_frames_per_clip,
                                            img_size=(args.w, args.h),
                                            reverse=False, transform=trans_test,
                                            temporal_transform = temporal_transform_test,
                                            modality = args.modality)
                                            for i in range(len(class_id1))]
        dataloader_init = [DataLoader(dataset_init[i], batch_size=args.batch_size,
                                        shuffle=True, 
                                        num_workers=args.num_workers, pin_memory=True)
                                        for i in range(len(class_id1))]
        icarl.net.eval()
        for dataloader_init_class in tqdm(dataloader_init):
            icarl.construct_exemplar_set(dataloader_init_class)
        icarl.net.train()

        class_step = 5 # incremental steps for new class
        for incremenral_class in trange(41, 84, class_step):
            if incremenral_class == range(41, 84, class_step)[-1]:
                class_id2 = [i for i in range(incremenral_class, 84)]
            else:
                class_id2 = [i for i in range(incremenral_class, incremenral_class+class_step)]
            class_all = class_id1 + class_id2

            print('Loading new class training data..... class {}'.format(range(class_id2[0], class_id2[-1])))
            # this dataset is used for preparing the exemplar set. DO NOT use any augmentation rules e.g. transforms
            dataset_new = [dataset_class.dataset_video_class(annot_dir, 'train_plus_val',
                                                class_id = [class_id2[i]],
                                                n_frames_per_clip=args.n_frames_per_clip,
                                                img_size=(args.w, args.h),
                                                reverse=False, transform=trans_test,
                                                temporal_transform = temporal_transform_test,
                                                modality = args.modality)
                                                for i in range(len(class_id2))]
            # new train dataset does not use augmentation (otherwise may causes some unstable issues)
            dataset_new_train = [dataset_class.dataset_video_class(annot_dir, 'train_plus_val',
                                                class_id = [class_id2[i]],
                                                n_frames_per_clip=args.n_frames_per_clip,
                                                img_size=(args.w, args.h),
                                                reverse=False, transform=trans_test,
                                                temporal_transform = temporal_transform_test,
                                                modality = args.modality)
                                                for i in range(len(class_id2))]

            print('Loading validating data..... class {}'.format(range(class_all[0], class_all[-1])))


            print('Loading testing data..... class_id1 {} class_id2 {}'.format(range(class_id1[0], class_id1[-1]),
                                                               range(class_id2[0], class_id2[-1]) ))
            dataset_test = [dataset_class.dataset_video_class(annot_dir, 'test', 
                                                class_id = class_id,
                                                n_frames_per_clip=args.n_frames_per_clip,
                                                img_size=(args.w, args.h), 
                                                reverse=False, transform=trans_test,
                                                temporal_transform = temporal_transform_test,
                                                modality = args.modality)
                                                for class_id in [class_id1, class_id2]]
            dataloader_test = [DataLoader(dataset_test[i], batch_size=args.batch_size_val, 
                                        num_workers=args.num_workers,pin_memory=True)
                                        for i in range(len(dataset_test))]
            

            icarl.train(dataset_new_train, dataset_new, class_id2, dataloader_test)
            test(net_test, save_dir, 
                '{}_ExemplarSet_{}.pkl'.format(args.modality, len(class_all)), 
                '{}-{}-{}.pth'.format(args.arch, args.modality, len(class_all)), 
                dataloader_test, class_id1, class_id2)
            class_id1 = deepcopy(class_all) # update learned class    
        pdb.set_trace()  
    else:
        class_id1 = [i for i in range(1, 41)]
        class_id2 = [i for i in range(41, 46)]
        # class_all = class_id1 + class_id2
        class_all = [i for i in range(1, 84)]
        print('Loading validating data.....')                                    
        dataset_test = dataset_class.dataset_video_class(annot_dir, 'test', 
                                            class_id = class_all,
                                            n_frames_per_clip=args.n_frames_per_clip,
                                            img_size=(args.w, args.h), 
                                            reverse=False, transform=trans_test,
                                            temporal_transform = temporal_transform_test,
                                            modality = args.modality)
        dataloader_test = DataLoader(dataset_test, batch_size=16, 
                                    num_workers=args.num_workers,pin_memory=True)

        test(net_test, save_dir, 
            '{}_ExemplarSet_{}.pkl'.format(args.modality, len(class_all)), 
            'resnext-101-{}-{}.pth'.format(args.modality, len(class_all)), 
            dataloader_test, class_id1, class_id2)

        # get_confusion_matrix(net_test, save_dir, 
        #     '{}_ExemplarSet_{}.pkl'.format(args.modality, len(class_all)), 
        #     'resnext-101-{}-{}.pth'.format(args.modality, len(class_all)), 
        #     dataloader_test, class_all)


