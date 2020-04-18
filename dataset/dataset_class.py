'''
This script loads the EgoGesture dataset through different classes
'''
import os 
import sys
import pickle
import numpy as np
import pandas as pd
import random
import torch
import pdb
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import skimage.util as ski_util
from sklearn.utils import shuffle
from temporal_transforms import *
from spatial_transforms import *

# self-defined modules
import opts




class make_dataset_instance(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        img = self.data[idx][0]
        label = self.data[idx][1]
        return img, torch.Tensor([label])[0].long()

    def __len__(self):
        return len(self.data)



def load_video_class(csv_path, mode, class_id, reverse=False):
    # mode: train, val, test
    csv_file = os.path.join(csv_path, '{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    # annot_df = DownSample(raw_annot_df, 40)
    rgb_samples = []
    depth_samples = []
    labels = []
    # get task index in dataframe
    task_ind = []
    for i in range(annot_df.shape[0]):
        if annot_df['label'][i] in class_id:
            task_ind.append(i)       
    annot_df_task = annot_df.iloc[task_ind] 
    for frame_i in range(annot_df_task.shape[0]):
        rgb_list = annot_df_task['rgb'].iloc[frame_i] # convert string in dataframe to list
        depth_list = annot_df_task['depth'].iloc[frame_i] # convert string in dataframe to list
        rgb_samples.append(rgb_list)
        depth_samples.append(depth_list)
        labels.append(annot_df_task['label'].iloc[frame_i])
        # data augmentation by reversing the sequence of the video
        if reverse:
            rgb_samples.append(rgb_list[::-1])
            depth_samples.append(depth_list[::-1])
            labels.append(annot_df_task['label'].iloc[frame_i])
    return rgb_samples, depth_samples, labels


class dataset_video_class(Dataset):
    def __init__(self, root_path, mode, class_id, n_frames_per_clip, img_size, 
                reverse=False, transform=None, temporal_transform=None, modality=None):
        self.root_path = root_path
        self.rgb_samples, self.depth_samples, self.labels = load_video_class(root_path, mode, 
                                                                            class_id, reverse)
        self.w = img_size[0]
        self.h = img_size[1]
        self.sample_num = len(self.rgb_samples)
        self.transform = transform
        self.n_frames_per_clip = n_frames_per_clip
        self.temporal_transform = temporal_transform
        self.modality = modality
        # print('{} {} samples have been loaded'.format(class_id, self.sample_num))

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        depth_name = self.depth_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        if self.temporal_transform is None:
        # repeat frame sequence less than the n_frames_per_clip
            if len(rgb_name) < self.n_frames_per_clip: 
                RepeatTimes = int(self.n_frames_per_clip/len(rgb_name))+1
                rgb_name = rgb_name*RepeatTimes
                depth_name = depth_name*RepeatTimes
            rgb = torch.zeros([3, len(rgb_name), self.h, self.w]).float()
            depth = torch.zeros([1, len(rgb_name), self.h, self.w]).float()
            for frame_name_i in range(len(rgb_name)):
                rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB").resize((171, 128), Image.BILINEAR)  # C * W * H
                # rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
                rgb_cache = self.transform(rgb_cache)
                rgb[:, frame_name_i, :, :] = rgb_cache
                depth_cache = Image.open(depth_name[frame_name_i]).convert("L").resize((171, 128), Image.BILINEAR)  # C * W * H
                # depth_cache = Image.open(depth_name[frame_name_i]).convert("L")
                depth_cache = self.transform(depth_cache)
                depth[:, frame_name_i, :, :] = depth_cache
        else:
            rgb = torch.zeros([3, self.n_frames_per_clip, self.h, self.w]).float()
            depth = torch.zeros([1, self.n_frames_per_clip, self.h, self.w]).float()
            selected_indice = self.temporal_transform(indices)
            for i, frame_name_i in enumerate(selected_indice):
                rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB").resize((171, 128), Image.BILINEAR)  # C * W * H
                # rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
                rgb_cache = self.transform(rgb_cache)
                rgb[:, i, :, :] = rgb_cache
                depth_cache = Image.open(depth_name[frame_name_i]).convert("L").resize((171, 128), Image.BILINEAR)  # C * W * H
                # depth_cache = Image.open(depth_name[frame_name_i]).convert("L")
                depth_cache = self.transform(depth_cache)
                depth[:, i, :, :] = depth_cache
        if self.modality == 'Depth':
            return depth, (torch.tensor(label)-1).long()
        elif self.modality == 'RGB':
            return rgb, (torch.tensor(label)-1).long()
        elif self.modality == 'RGB-D':
            return torch.cat([rgb, depth], 0), (torch.tensor(label)-1).long()
        else:
            return rgb, depth, (torch.tensor(label)-1).long()
        # return rgb, mask, (torch.tensor(label)-1).long()

    def __len__(self):
        return int(self.sample_num)


# root_path = './'
# class_id = [i for i in range(0,41)]
# args = opts.parse_opts()
# trans_train = Compose([
#             Scale([112,112]),
#             SpatialElasticDisplacement(),
#             ToTensor(1)
#             ])
# temporal_transform_ = Compose([
#         TemporalCenterCrop(100)
#         ])

# dataset_train = dataset_video_class(root_path, 'train',
#                                     n_frames_per_clip=100,
#                                     class_id = class_id,
#                                     img_size=(args.w, args.h),
#                                     reverse=False, transform=trans_train,
#                                     temporal_transform = temporal_transform_)
# rgb, depth, label = dataset_train.__getitem__(0)



# # dataloader_train = DataLoader(dataset_train, batch_size=32,
# #                                 shuffle=True, 
# #                                 num_workers=args.num_workers, pin_memory=True)
# # trainiter = iter(dataloader_train)
# # rgbs, masks, labels = trainiter.next()

# pdb.set_trace()