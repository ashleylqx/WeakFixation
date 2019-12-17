import os
import pdb
import sys
import cv2
import pickle
import scipy.misc
import scipy.io
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import numpy as np
import random

# import torch.nn.functional as F
# from torch_geometric.data import Data, Batch
import torch.nn.functional as F

# --- coco api --------------
import json
import time
from collections import defaultdict
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


import matplotlib.pylab as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import *

from data_aug.data_aug_map import *
from data_aug.bbox_util import *


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # rgb_mean,
                             std=[0.229, 0.224, 0.225])])

# TODO: discuss if the resize should preserve aspect ratio; now image and sal_img does not preserve
def imageProcessing(image, saliency, h=input_h, w=input_w):
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    saliency = cv2.resize(saliency, (output_w, output_h), interpolation=cv2.INTER_AREA).astype(np.float32)

    # remove mean value
    # image -= bgr_mean
    # transform to tensor
    # image = torch.FloatTensor(image)

    # swap channel dimensions to RGB
    # image = image.permute(2, 0, 1)
    # if np.max(saliency) != 0:
    #     saliency = saliency*0.2/np.max(saliency)

    return transform(image/255.), torch.FloatTensor(saliency/255.)
    # return transform(image/255.), torch.FloatTensor(saliency)

def resize_fixation_mat(fix, rows=output_h, cols=output_w):
    out = np.zeros((rows, cols))
    resolution = fix["resolution"][0]
    # print('resolution:', resolution)
    factor_scale_r = rows*1.0 / resolution[0]
    factor_scale_c = cols*1.0 / resolution[1]
    # print('factor_scale_r:', factor_scale_r)
    # print('factor_scale_c:', factor_scale_c)
    coords = []
    for gaze in fix["gaze"][0]: # previous (1, N, 3)
        coords.extend(gaze[2])

    # for gaze in fix["gaze"]:  # new(N, 1, 3)
    #     coords.extend(gaze[0][2])

    for coord in coords:
        # print('coord:', coord[0], coord[1])
        r = np.round(coord[1] * factor_scale_r).astype(np.int32)
        c = np.round(coord[0] * factor_scale_c).astype(np.int32)
        if r >= rows:
            r = rows-1
        if c >= cols:
            c = cols-1
        out[r, c] = 1

    return out

# preserve aspect ratio
def fixationProcessing_mat(fix_mat, h=output_h, w=output_w):
    fix = np.zeros((h, w))

    original_shape = fix_mat['resolution'][0]
    rows_rate = original_shape[0]*1.0 / h
    cols_rate = original_shape[1]*1.0 / w

    if rows_rate > cols_rate:
        new_cols = int((original_shape[1] * float(h)) // original_shape[0])
        fix_mat = resize_fixation_mat(fix_mat, rows=h, cols=new_cols)
        if new_cols > w:
            new_cols = w
        fix[:, ((fix.shape[1] - new_cols) // 2):((fix.shape[1] - new_cols) // 2 + new_cols), ] = fix_mat
    else:
        new_rows = int((original_shape[0] * float(w)) // original_shape[1])
        fix_mat = resize_fixation_mat(fix_mat, rows=new_rows, cols=w) # the result seems right
        if new_rows > h:
            new_rows = h
        fix[((fix.shape[0] - new_rows) // 2):((fix.shape[0] - new_rows) // 2 + new_rows), :] = fix_mat

    return fix


class SALICON_full(Dataset):
    def __init__(self, mode='train', return_path=False, N=None,
                 img_h = input_h, img_w = input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_SALICON
        self.path_images = os.path.join(self.path_dataset, 'images', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features_2', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500', mode)
        self.path_saliency = os.path.join(self.path_dataset, 'maps', mode)
        self.path_fixation = os.path.join(self.path_dataset, 'fixations', mode)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names
        # list_names = np.array(['COCO_val2014_000000030785',
        #                        'COCO_val2014_000000061507',
        #                        'COCO_val2014_000000091883',
        #                        'COCO_val2014_000000123480',
        #                        'COCO_val2014_000000031519',
        #                        'COCO_val2014_000000061520',
        #                        'COCO_val2014_000000091909',
        #                        'COCO_val2014_000000123580'])
        # self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]


        # self.coco = COCO(os.path.join(PATH_COCO, 'annotations', 'instances_%s2014.json'%mode))
        self.imgNsToCat = pickle.load(open(os.path.join(PATH_COCO, 'imgNsToCat_{}.p'.format(mode)), "rb"))

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init SALICON full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        sal_path = os.path.join(self.path_saliency, self.list_names[index]+'.png')
        fix_path = os.path.join(self.path_fixation, self.list_names[index]+'.mat')
        box_path = os.path.join(self.edge_boxes, self.list_names[index]+'_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        # image = cv2.imread(rgb_ima)
        # saliency = cv2.imread(sal_path, 0)
        fixation = scipy.io.loadmat(fix_path)
        fix_processed = fixationProcessing_mat(fixation)
        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        if os.path.exists(sal_path):
            saliency = cv2.imread(sal_path, 0)
        else:
            saliency = gaussian_filter(fix_processed, sigma=5)

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)
        # img_processed, sal_processed, ori_img = imageProcessing_img(image, saliency, h=self.img_h, w=self.img_w)

        # get coco label
        label_indices = self.imgNsToCat[self.list_names[index]]
        # label_indices = self.coco.imgNsToCat[self.list_names[index]]
        label = torch.zeros(coco_num_classes)
        if len(label_indices)>0:
            label[label_indices] = 1
        else:
            label[0] = 1

        # -------------------------------
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max==0.] = 1e-8
        #     box_features = box_features / box_f_max
        # boxes = np.load(os.path.join(self.path_features, '{}_boxes.npy'.format(self.list_names[index])))
        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.return_path:
            return img_processed, label, boxes, sal_processed, fix_processed, self.list_names[index]
        else:
            return img_processed, label, boxes, sal_processed, fix_processed

class MS_COCO_full(Dataset):
    def __init__(self, mode='train', return_path=False, N=None,
                 img_h = input_h, img_w = input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_COCO
        self.path_images = os.path.join(self.path_dataset, mode+'2014')
        # self.path_features = os.path.join(self.path_dataset, 'features_2', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, mode + '2014_eb500')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        # if mode=='train':
        #     list_names = np.array(['COCO_train2014_000000001108',
        #                            'COCO_train2014_000000002148',
        #                            'COCO_train2014_000000003348',
        #                            'COCO_train2014_000000004575'])
        # elif mode=='val':
        #     list_names = np.array(['COCO_val2014_000000005586',
        #                            'COCO_val2014_000000011122',
        #                            'COCO_val2014_000000016733',
        #                            'COCO_val2014_000000022199'])
        #
        # self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]


        # self.coco = COCO(os.path.join(PATH_COCO, 'annotations', 'instances_%s2014.json'%mode))
        self.imgNsToCat = pickle.load(open(os.path.join(PATH_COCO, 'imgNsToCat_{}.p'.format(mode)), "rb"))

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init MS_COCO full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')

        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA).astype(np.float32)

        img_processed = transform(image/255.)

        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        # get coco label
        label_indices = self.imgNsToCat[self.list_names[index]]
        # label_indices = self.coco.imgNsToCat[self.list_names[index]]
        label = torch.zeros(coco_num_classes)
        if len(label_indices)>0:
            label[label_indices] = 1
        else:
            label[0] = 1

        # -------------------------------
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max == 0.] = 1e-8
        #     box_features = box_features / box_f_max
        # boxes = np.load(os.path.join(self.path_features, '{}_boxes.npy'.format(self.list_names[index])))
        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.return_path:
            return img_processed, label, boxes, self.list_names[index]
        else:
            return img_processed, label, boxes

class MS_COCO_map_full(Dataset):
    def __init__(self, mode='train', return_path=False, N=None, prior = 'nips08',
                 img_h = input_h, img_w = input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_COCO
        self.path_images = os.path.join(self.path_dataset, mode+'2014')
        # self.path_features = os.path.join(self.path_dataset, 'features_2', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, mode + '2014_eb500')

        self.path_saliency = os.path.join(self.path_dataset, mode + '2014_%s'%prior)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        # list_names = os.listdir(self.edge_boxes)
        # list_names = np.array([n.split('.')[0][:-7] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]


        # self.coco = COCO(os.path.join(PATH_COCO, 'annotations', 'instances_%s2014.json'%mode))
        self.imgNsToCat = pickle.load(open(os.path.join(PATH_COCO, 'imgNsToCat_{}.p'.format(mode)), "rb"))

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init MS_COCO full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        sal_path = os.path.join(self.path_saliency, self.list_names[index] + '.png')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        saliency = cv2.imread(sal_path, 0)

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        # get coco label
        label_indices = self.imgNsToCat[self.list_names[index]]
        # label_indices = self.coco.imgNsToCat[self.list_names[index]]
        label = torch.zeros(coco_num_classes)
        if len(label_indices)>0:
            label[label_indices] = 1
        else:
            label[0] = 1

        # -------------------------------
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max == 0.] = 1e-8
        #     box_features = box_features / box_f_max
        # boxes = np.load(os.path.join(self.path_features, '{}_boxes.npy'.format(self.list_names[index])))
        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.return_path:
            return img_processed, label, boxes, sal_processed, self.list_names[index]
        else:
            return img_processed, label, boxes, sal_processed

class MS_COCO_map_full_aug(Dataset):
    def __init__(self, mode='train', return_path=False, N=None, prior = 'nips08',
                 img_h = input_h, img_w = input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_COCO
        self.path_images = os.path.join(self.path_dataset, mode+'2014')
        # self.path_features = os.path.join(self.path_dataset, 'features_2', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, mode + '2014_eb500')

        self.path_saliency = os.path.join(self.path_dataset, mode + '2014_%s'%prior)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        # list_names = os.listdir(self.edge_boxes)
        # list_names = np.array([n.split('.')[0][:-7] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]


        # self.coco = COCO(os.path.join(PATH_COCO, 'annotations', 'instances_%s2014.json'%mode))
        self.imgNsToCat = pickle.load(open(os.path.join(PATH_COCO, 'imgNsToCat_{}.p'.format(mode)), "rb"))

        #self.seq = Sequence([RandomHSV(40, 40, 40),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     RandomScale(0.2, diff=True),
        #                     RandomTranslate(0.2, diff=True),
        #                     RandomRotate(10),
        #                     RandomShear(0.2)]
        #                    ) # 1st attempt _aug

        # self.seq = Sequence([RandomHSV(20, 20, 20),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     RandomScale(0.1, diff=True),
        #                     RandomTranslate(0.1, diff=True),
        #                     RandomRotate(10),
        #                     RandomShear(0.1)]
        #                    ) # 2nd attempt _aug2

        #self.seq = Sequence([RandomHSV(10, 10, 10),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     RandomScale(0.05, diff=True),
        #                     RandomTranslate(0.05, diff=True),
        #                     RandomRotate(5),
        #                     RandomShear(0.05)]
        #                    ) # _aug3

        #self.seq = Sequence([#RandomHSV(10, 10, 10),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     RandomScale(0.1, diff=True),
        #                     RandomTranslate(0.1, diff=True),
        #                     RandomRotate(5)], [0.2,0.2,0.2,0.2]
        #                     #RandomShear(0.1)]
        #                    ) # _aug4

        #self.seq = Sequence([RandomHSV(5, 5, 5),
        #                    RandomHorizontalFlip(), #p=0.5
        #                    RandomScale(0.1, diff=True),
        #                    RandomTranslate(0.1, diff=True),
        #                    RandomRotate(5)], [0.2,0.2,0.2,0.2,0.2]
        #                    #RandomShear(0.1)]
        #                   ) # _aug5

        #self.seq = RandomHorizontalFlip() # _aug6
        self.seq = RandomRotate(5) # _aug7
        # self.seq = RandomRotate(10) # _aug7_2
        # self.seq = RandomScale(0.1, diff=True) # _aug8
        # self.seq = RandomScale(0.01, diff=True) # _aug8_2


        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init MS_COCO full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        sal_path = os.path.join(self.path_saliency, self.list_names[index] + '.png')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB') # (h,w,c)
        saliency = cv2.imread(sal_path, 0)
        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        if boxes.shape[0]==0:
            img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)
            boxes_ = np.zeros_like(boxes)
            boxes_[:, 0] = boxes[:, 0] * self.img_w
            boxes_[:, 2] = boxes[:, 2] * self.img_w
            boxes_[:, 1] = boxes[:, 1] * self.img_h
            boxes_[:, 3] = boxes[:, 3] * self.img_h
        else:
            boxes[:, 0] = boxes[:, 0] * image.shape[1]
            boxes[:, 2] = boxes[:, 2] * image.shape[1]
            boxes[:, 1] = boxes[:, 1] * image.shape[0]
            boxes[:, 3] = boxes[:, 3] * image.shape[0]

            image_, saliency_, boxes_ = self.seq(image.copy(), saliency.copy(), boxes.copy())

            img_processed, sal_processed = imageProcessing(image_, saliency_, h=self.img_h, w=self.img_w)

            # if boxes_.min()<0:
            #     pdb.set_trace()

            #np.clip(boxes_[:, 0], 0., image.shape[1], out=boxes_[:, 0])
            #np.clip(boxes_[:, 2], 0., image.shape[1], out=boxes_[:, 2])
            #np.clip(boxes_[:, 1], 0., image.shape[0], out=boxes_[:, 1])
            #np.clip(boxes_[:, 3], 0., image.shape[0], out=boxes_[:, 3])

            #boxes_[boxes_[:, 0] < 0., 0] = 0.
            #boxes_[boxes_[:, 0] > image.shape[1], 0] = image.shape[1]
            #boxes_[boxes_[:, 2] < boxes_[:, 0], 2] = boxes_[:, 0]
            #boxes_[boxes_[:, 2] > image.shape[1], 2] = image.shape[1]

            #boxes_[boxes_[:, 1] < 0., 1] = 0.
            #boxes_[boxes_[:, 1] > image.shape[0], 1] = image.shape[0]
            #boxes_[boxes_[:, 3] < boxes_[:, 1], 3] = boxes_[:, 1]
            #boxes_[boxes_[:, 3] > image.shape[0], 3] = image.shape[0]

            np.clip(boxes_[:, 0], 0., image.shape[1], out=boxes_[:, 0])
            np.clip(boxes_[:, 2], 0., image.shape[1], out=boxes_[:, 2])
            boxes_[:, 1] = np.minimum(np.maximum(boxes_[:, 0], boxes_[:, 1]), image.shape[0])
            boxes_[:, 3] = np.minimum(np.maximum(boxes_[:, 2], boxes_[:, 3]), image.shape[0])

            boxes_[:, 0] = boxes_[:, 0] / image.shape[1] * self.img_w
            boxes_[:, 2] = boxes_[:, 2] / image.shape[1] * self.img_w
            boxes_[:, 1] = boxes_[:, 1] / image.shape[0] * self.img_h
            boxes_[:, 3] = boxes_[:, 3] / image.shape[0] * self.img_h

        # get coco label
        label_indices = self.imgNsToCat[self.list_names[index]]
        # label_indices = self.coco.imgNsToCat[self.list_names[index]]
        label = torch.zeros(coco_num_classes)
        if len(label_indices)>0:
            label[label_indices] = 1
        else:
            label[0] = 1


        if self.return_path:
            return img_processed, label, boxes_, sal_processed, self.list_names[index]
        else:
            return img_processed, label, boxes_, sal_processed

class MS_COCO_map_full_aug_sf(Dataset):
    def __init__(self, mode='train', return_path=False, N=None, prior = 'nips08',
                 img_h = input_h, img_w = input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_COCO
        self.path_images = os.path.join(self.path_dataset, mode+'2014')
        # self.path_features = os.path.join(self.path_dataset, 'features_2', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, mode + '2014_eb500')

        self.path_saliency = os.path.join(self.path_dataset, mode + '2014_%s'%prior)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        # list_names = os.listdir(self.edge_boxes)
        # list_names = np.array([n.split('.')[0][:-7] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]


        # self.coco = COCO(os.path.join(PATH_COCO, 'annotations', 'instances_%s2014.json'%mode))
        self.imgNsToCat = pickle.load(open(os.path.join(PATH_COCO, 'imgNsToCat_{}.p'.format(mode)), "rb"))

        #self.seq = Sequence([RandomHSV(40, 40, 40),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     RandomScale(0.2, diff=True),
        #                     RandomTranslate(0.2, diff=True),
        #                     RandomRotate(10),
        #                     RandomShear(0.2)]
        #                    ) # 1st attempt _aug

        # self.seq = Sequence([RandomHSV(20, 20, 20),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     RandomScale(0.1, diff=True),
        #                     RandomTranslate(0.1, diff=True),
        #                     RandomRotate(10),
        #                     RandomShear(0.1)]
        #                    ) # 2nd attempt _aug2

        #self.seq = Sequence([RandomHSV(10, 10, 10),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     RandomScale(0.05, diff=True),
        #                     RandomTranslate(0.05, diff=True),
        #                     RandomRotate(5),
        #                     RandomShear(0.05)]
        #                    ) # _aug3

        #self.seq = Sequence([#RandomHSV(10, 10, 10),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     RandomScale(0.1, diff=True),
        #                     RandomTranslate(0.1, diff=True),
        #                     RandomRotate(5)], [0.2,0.2,0.2,0.2]
        #                     #RandomShear(0.1)]
        #                    ) # _aug4

        #self.seq = Sequence([RandomHSV(5, 5, 5),
        #                    RandomHorizontalFlip(), #p=0.5
        #                    RandomScale(0.1, diff=True),
        #                    RandomTranslate(0.1, diff=True),
        #                    RandomRotate(5)], [0.2,0.2,0.2,0.2,0.2]
        #                    #RandomShear(0.1)]
        #                   ) # _aug5

        #self.seq = RandomHorizontalFlip() # _aug6
        self.seq = RandomRotate(5) # _aug7
        # self.seq = RandomRotate(10) # _aug7_2
        # self.seq = RandomScale(0.1, diff=True) # _aug8
        # self.seq = RandomScale(0.01, diff=True) # _aug8_2


        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init MS_COCO full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        sal_path = os.path.join(self.path_saliency, self.list_names[index] + '.png')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB') # (h,w,c)
        saliency = cv2.imread(sal_path, 0)
        tmp_boxes = scipy.io.loadmat(box_path)['bboxes']
        if tmp_boxes.shape[0]//2 > MAX_BNUM:
            box_indices = random.sample(population=range(tmp_boxes.shape[0]//2), k=MAX_BNUM)
            boxes = tmp_boxes[box_indices, :]
        else:
            boxes = tmp_boxes[:tmp_boxes.shape[0]//2, :]

        # if tmp_boxes.shape[0] > MAX_BNUM:
        #     box_indices = random.sample(population=range(tmp_boxes.shape[0]), k=MAX_BNUM)
        #     boxes = tmp_boxes[box_indices, :]
        # else:
        #     boxes = tmp_boxes

        if boxes.shape[0]==0:
            img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)
            boxes_ = np.zeros_like(boxes)
            boxes_[:, 0] = boxes[:, 0] * self.img_w
            boxes_[:, 2] = boxes[:, 2] * self.img_w
            boxes_[:, 1] = boxes[:, 1] * self.img_h
            boxes_[:, 3] = boxes[:, 3] * self.img_h
        else:
            boxes[:, 0] = boxes[:, 0] * image.shape[1]
            boxes[:, 2] = boxes[:, 2] * image.shape[1]
            boxes[:, 1] = boxes[:, 1] * image.shape[0]
            boxes[:, 3] = boxes[:, 3] * image.shape[0]

            image_, saliency_, boxes_ = self.seq(image.copy(), saliency.copy(), boxes.copy())

            img_processed, sal_processed = imageProcessing(image_, saliency_, h=self.img_h, w=self.img_w)

            # if boxes_.min()<0:
            #     pdb.set_trace()

            #np.clip(boxes_[:, 0], 0., image.shape[1], out=boxes_[:, 0])
            #np.clip(boxes_[:, 2], 0., image.shape[1], out=boxes_[:, 2])
            #np.clip(boxes_[:, 1], 0., image.shape[0], out=boxes_[:, 1])
            #np.clip(boxes_[:, 3], 0., image.shape[0], out=boxes_[:, 3])

            #boxes_[boxes_[:, 0] < 0., 0] = 0.
            #boxes_[boxes_[:, 0] > image.shape[1], 0] = image.shape[1]
            #boxes_[boxes_[:, 2] < boxes_[:, 0], 2] = boxes_[:, 0]
            #boxes_[boxes_[:, 2] > image.shape[1], 2] = image.shape[1]

            #boxes_[boxes_[:, 1] < 0., 1] = 0.
            #boxes_[boxes_[:, 1] > image.shape[0], 1] = image.shape[0]
            #boxes_[boxes_[:, 3] < boxes_[:, 1], 3] = boxes_[:, 1]
            #boxes_[boxes_[:, 3] > image.shape[0], 3] = image.shape[0]

            np.clip(boxes_[:, 0], 0., image.shape[1], out=boxes_[:, 0])
            np.clip(boxes_[:, 2], 0., image.shape[1], out=boxes_[:, 2])
            boxes_[:, 1] = np.minimum(np.maximum(boxes_[:, 0], boxes_[:, 1]), image.shape[0])
            boxes_[:, 3] = np.minimum(np.maximum(boxes_[:, 2], boxes_[:, 3]), image.shape[0])

            boxes_[:, 0] = boxes_[:, 0] / image.shape[1] * self.img_w
            boxes_[:, 2] = boxes_[:, 2] / image.shape[1] * self.img_w
            boxes_[:, 1] = boxes_[:, 1] / image.shape[0] * self.img_h
            boxes_[:, 3] = boxes_[:, 3] / image.shape[0] * self.img_h

        # get coco label
        label_indices = self.imgNsToCat[self.list_names[index]]
        # label_indices = self.coco.imgNsToCat[self.list_names[index]]
        label = torch.zeros(coco_num_classes)
        if len(label_indices)>0:
            label[label_indices] = 1
        else:
            label[0] = 1


        if self.return_path:
            return img_processed, label, boxes_, sal_processed, self.list_names[index]
        else:
            return img_processed, label, boxes_, sal_processed

# if num_tgt_cls is not None, use collate_coco since its also multi_label
class ILSVRC_full_wrong(Dataset):
    def __init__(self, mode='train', return_path=False, N=None,
                 img_h = input_h, img_w = input_w, num_tgt_cls=ilsvrc_num_tgt_classes): #'train', 'test', 'val'

        if num_tgt_cls is not None: # but this mapping is not uniform; turn to multi-label mapping later.
            if num_tgt_cls not in ILSVRC_TGT_CLS:
                print('If being specified, the num of target classes must be in {256, 385}.')
                raise NotImplementedError
        self.num_tgt_cls = num_tgt_cls
        self.mode = mode
        self.path_dataset = PATH_ILSVRC
        self.path_images = os.path.join(self.path_dataset, 'images', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features_2', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500', self.mode)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # get list images
        with open(os.path.join(self.path_dataset, '%s_list.txt'%self.mode), 'r') as f:
            tmp = f.readlines()
        list_names = [l.split(' ')[0] for l in tmp] # .JPEG
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        labels = [int(l.split(' ')[1]) for l in tmp]
        if self.num_tgt_cls is not None:
            self.lb_mappings = pickle.load(open(
                os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_train_{}.pickle'.format(num_tgt_cls)), 'rb'))
            # self.lb_mappings = pickle.load(open(
            #         os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_{}_{}.pickle'.format(self.mode, num_tgt_cls)), 'rb'))
        #     tmp = [lb_mappings[l] for l in labels]
        #     labels = tmp.copy()
        #     assert max(labels)==num_tgt_cls-1

        self.labels = labels
        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init ILSVRC full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.JPEG')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]


        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        img_processed = transform(image / 255.)

        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.num_tgt_cls is not None:
            # get coco label
            # if self.mode == 'val':
            #     label_indices = self.lb_mappings[self.list_names[index]]
            # elif self.mode == 'train':
            #     label_indices = self.lb_mappings[self.labels[index]]
            #     # label_indices = self.lb_mappings[self.list_names[index].split('/')[0]]
            label_indices = self.lb_mappings[self.labels[index]]
            # label_indices = self.coco.imgNsToCat[self.list_names[index]]
            label = torch.zeros(ilsvrc_num_tgt_classes)
            label[label_indices] = 1
        else:
            label = self.labels[index]

        if self.return_path:
            return img_processed, label, boxes, self.list_names[index]
        else:
            return img_processed, label, boxes

class ILSVRC_map_full_wrong(Dataset):
    def __init__(self, mode='train', return_path=False, N=None,
                 img_h = input_h, img_w = input_w, num_tgt_cls=ilsvrc_num_tgt_classes): #'train', 'test', 'val'

        if num_tgt_cls is not None:
            if num_tgt_cls not in ILSVRC_TGT_CLS:
                print('If being specified, the num of target classes must be in {256, 385}.')
                raise NotImplementedError
        self.num_tgt_cls = num_tgt_cls
        self.mode =  mode
        self.path_dataset = PATH_ILSVRC
        self.path_images = os.path.join(self.path_dataset, 'images', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features_2', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500', self.mode)
        self.path_saliency = os.path.join(self.path_dataset, 'images', self.mode + '_nips08')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # get list images
        with open(os.path.join(self.path_dataset, '%s_list.txt'%self.mode), 'r') as f:
            tmp = f.readlines()
        list_names = [l.split(' ')[0] for l in tmp] # .JPEG
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        labels = [int(l.split(' ')[1]) for l in tmp]
        if self.num_tgt_cls is not None:
            self.lb_mappings = pickle.load(open(
                os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_train_{}.pickle'.format(num_tgt_cls)), 'rb'))
            # self.lb_mappings = pickle.load(open(
            #     os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_{}_{}.pickle'.format(self.mode, num_tgt_cls)), 'rb'))
            # # tmp = [lb_mappings[l] for l in labels]
            # labels = tmp.copy()
            # assert max(labels)==num_tgt_cls-1

        self.labels = labels
        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init ILSVRC full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.JPEG')
        sal_path = os.path.join(self.path_saliency, self.list_names[index] + '.png')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        saliency = cv2.imread(sal_path, 0)
        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.num_tgt_cls is not None:
            # if self.mode=='val':
            #     label_indices = self.lb_mappings[self.list_names[index]]
            # elif self.mode=='train':
            #     label_indices = self.lb_mappings[self.labels[index]]
            #     # label_indices = self.lb_mappings[self.list_names[index].split('/')[0]]
            # label_indices = self.coco.imgNsToCat[self.list_names[index]]
            label_indices = self.lb_mappings[self.labels[index]]
            label = torch.zeros(ilsvrc_num_tgt_classes)
            label[label_indices] = 1
        else:
            label = self.labels[index]

        if self.return_path:
            return img_processed, label, boxes, sal_processed, self.list_names[index]
        else:
            return img_processed, label, boxes, sal_processed

class ILSVRC_map_full_aug_wrong(Dataset):
    def __init__(self, mode='train', return_path=False, N=None, prior = 'nips08',
                 img_h = input_h, img_w = input_w, normalize_feature=False, num_tgt_cls=ilsvrc_num_tgt_classes): #'train', 'test', 'val'

        if num_tgt_cls is not None:
            if num_tgt_cls not in ILSVRC_TGT_CLS:
                print('If being specified, the num of target classes must be in {256, 385}.')
                raise NotImplementedError
        self.num_tgt_cls = num_tgt_cls
        self.mode =  mode
        self.path_dataset = PATH_ILSVRC
        self.path_images = os.path.join(self.path_dataset, 'images', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features_2', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500', self.mode)
        self.path_saliency = os.path.join(self.path_dataset, 'images', self.mode + '_' + prior)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        self.normalize_feature = normalize_feature

        # get list images
        with open(os.path.join(self.path_dataset, '%s_list.txt'%self.mode), 'r') as f:
            tmp = f.readlines()
        list_names = [l.split(' ')[0] for l in tmp] # .JPEG
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        labels = [int(l.split(' ')[1]) for l in tmp]
        if self.num_tgt_cls is not None:
            self.lb_mappings = pickle.load(open(
                os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_train_{}.pickle'.format(num_tgt_cls)), 'rb'))
            # self.lb_mappings = pickle.load(open(
            #     os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_{}_{}.pickle'.format(self.mode, num_tgt_cls)), 'rb'))
            # # tmp = [lb_mappings[l] for l in labels]
            # labels = tmp.copy()
            # assert max(labels)==num_tgt_cls-1

        self.labels = labels
        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        self.seq = Sequence([RandomHSV(10, 10, 10),
                             RandomHorizontalFlip(),  # p=0.5
                             RandomScale(0.05, diff=True),
                             RandomTranslate(0.05, diff=True),
                             RandomRotate(5),
                             RandomShear(0.05)]
                            )  # _aug3

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init ILSVRC full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.JPEG')
        sal_path = os.path.join(self.path_saliency, self.list_names[index] + '.png')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        saliency = cv2.imread(sal_path, 0)
        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        if boxes.shape[0] == 0:
            img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)
            boxes_ = np.zeros_like(boxes)
            boxes_[:, 0] = boxes[:, 0] * self.img_w
            boxes_[:, 2] = boxes[:, 2] * self.img_w
            boxes_[:, 1] = boxes[:, 1] * self.img_h
            boxes_[:, 3] = boxes[:, 3] * self.img_h

        else:
            boxes[:, 0] = boxes[:, 0] * image.shape[1]
            boxes[:, 2] = boxes[:, 2] * image.shape[1]
            boxes[:, 1] = boxes[:, 1] * image.shape[0]
            boxes[:, 3] = boxes[:, 3] * image.shape[0]

            image_, saliency_, boxes_ = self.seq(image.copy(), saliency.copy(), boxes.copy())

            img_processed, sal_processed = imageProcessing(image_, saliency_, h=self.img_h, w=self.img_w)

            # np.clip(boxes_[:, 0], 0., image.shape[1], out=boxes_[:, 0])
            # np.clip(boxes_[:, 2], 0., image.shape[1], out=boxes_[:, 2])
            # np.clip(boxes_[:, 1], 0., image.shape[0], out=boxes_[:, 1])
            # np.clip(boxes_[:, 3], 0., image.shape[0], out=boxes_[:, 3])

            # boxes_[boxes_[:, 0] < 0., 0] = 0.
            # boxes_[boxes_[:, 0] > image.shape[1], 0] = image.shape[1]
            # boxes_[boxes_[:, 2] < boxes_[:, 0], 2] = boxes_[:, 0]
            # boxes_[boxes_[:, 2] > image.shape[1], 2] = image.shape[1]

            # boxes_[boxes_[:, 1] < 0., 1] = 0.
            # boxes_[boxes_[:, 1] > image.shape[0], 1] = image.shape[0]
            # boxes_[boxes_[:, 3] < boxes_[:, 1], 3] = boxes_[:, 1]
            # boxes_[boxes_[:, 3] > image.shape[0], 3] = image.shape[0]

            np.clip(boxes_[:, 0], 0., image.shape[1], out=boxes_[:, 0])
            np.clip(boxes_[:, 2], 0., image.shape[1], out=boxes_[:, 2])
            boxes_[:, 1] = np.minimum(np.maximum(boxes_[:, 0], boxes_[:, 1]), image.shape[0])
            boxes_[:, 3] = np.minimum(np.maximum(boxes_[:, 2], boxes_[:, 3]), image.shape[0])

            boxes_[:, 0] = boxes_[:, 0] / image.shape[1] * self.img_w
            boxes_[:, 2] = boxes_[:, 2] / image.shape[1] * self.img_w
            boxes_[:, 1] = boxes_[:, 1] / image.shape[0] * self.img_h
            boxes_[:, 3] = boxes_[:, 3] / image.shape[0] * self.img_h


        # load features and transfor to geometric Data
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max == 0.] = 1e-8
        #     box_features = box_features / box_f_max

        if self.num_tgt_cls is not None:
            # if self.mode=='val':
            #     label_indices = self.lb_mappings[self.list_names[index]]
            # elif self.mode=='train':
            #     label_indices = self.lb_mappings[self.labels[index]]
            #     # label_indices = self.lb_mappings[self.list_names[index].split('/')[0]]
            # label_indices = self.coco.imgNsToCat[self.list_names[index]]
            label_indices = self.lb_mappings[self.labels[index]]
            label = torch.zeros(ilsvrc_num_tgt_classes)
            label[label_indices] = 1
        else:
            label = self.labels[index]

        if self.return_path:
            return img_processed, label, boxes_, sal_processed, self.list_names[index]
        else:
            return img_processed, label, boxes_, sal_processed


class ILSVRC_full(Dataset):
    def __init__(self, mode='train', return_path=False, N=None,
                 img_h = input_h, img_w = input_w, num_tgt_cls=ilsvrc_num_tgt_classes): #'train', 'test', 'val'

        if num_tgt_cls is not None: # but this mapping is not uniform; turn to multi-label mapping later.
            if num_tgt_cls not in ILSVRC_TGT_CLS:
                print('If being specified, the num of target classes must be in {256, 385}.')
                raise NotImplementedError
        self.num_tgt_cls = num_tgt_cls
        self.mode = mode
        self.path_dataset = PATH_ILSVRC
        self.path_images = os.path.join(self.path_dataset, 'images', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features_2', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500', self.mode)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # get list images
        with open(os.path.join(self.path_dataset, '%s_list.txt'%self.mode), 'r') as f:
            tmp = f.readlines()
        list_names = [l.split(' ')[0] for l in tmp] # .JPEG
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        labels = [int(l.split(' ')[1]) for l in tmp]
        if self.num_tgt_cls is not None:
            self.lb_mappings = pickle.load(open(
                os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_train_{}.pickle'.format(num_tgt_cls)), 'rb'))
            # self.lb_mappings = pickle.load(open(
            #         os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_{}_{}.pickle'.format(self.mode, num_tgt_cls)), 'rb'))
        #     tmp = [lb_mappings[l] for l in labels]
        #     labels = tmp.copy()
        #     assert max(labels)==num_tgt_cls-1

        self.labels = labels
        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init ILSVRC full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.JPEG')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]


        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        img_processed = transform(image / 255.)

        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.num_tgt_cls is not None:
            label = self.lb_mappings[self.labels[index]][0]

        else:
            label = self.labels[index]

        if self.return_path:
            return img_processed, label, boxes, self.list_names[index]
        else:
            return img_processed, label, boxes

class ILSVRC_map_full(Dataset):
    def __init__(self, mode='train', return_path=False, N=None,
                 img_h = input_h, img_w = input_w, num_tgt_cls=ilsvrc_num_tgt_classes): #'train', 'test', 'val'

        if num_tgt_cls is not None:
            if num_tgt_cls not in ILSVRC_TGT_CLS:
                print('If being specified, the num of target classes must be in {256, 385}.')
                raise NotImplementedError
        self.num_tgt_cls = num_tgt_cls
        self.mode =  mode
        self.path_dataset = PATH_ILSVRC
        self.path_images = os.path.join(self.path_dataset, 'images', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features_2', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500', self.mode)
        self.path_saliency = os.path.join(self.path_dataset, 'images', self.mode + '_nips08')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # get list images
        with open(os.path.join(self.path_dataset, '%s_list.txt'%self.mode), 'r') as f:
            tmp = f.readlines()
        list_names = [l.split(' ')[0] for l in tmp] # .JPEG
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        labels = [int(l.split(' ')[1]) for l in tmp]
        if self.num_tgt_cls is not None:
            self.lb_mappings = pickle.load(open(
                os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_train_{}.pickle'.format(num_tgt_cls)), 'rb'))
            # self.lb_mappings = pickle.load(open(
            #     os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_{}_{}.pickle'.format(self.mode, num_tgt_cls)), 'rb'))
            # # tmp = [lb_mappings[l] for l in labels]
            # labels = tmp.copy()
            # assert max(labels)==num_tgt_cls-1

        self.labels = labels
        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init ILSVRC full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.JPEG')
        sal_path = os.path.join(self.path_saliency, self.list_names[index] + '.png')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        saliency = cv2.imread(sal_path, 0)
        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.num_tgt_cls is not None:
            label = self.lb_mappings[self.labels[index]][0]
        else:
            label = self.labels[index]

        if self.return_path:
            return img_processed, label, boxes, sal_processed, self.list_names[index]
        else:
            return img_processed, label, boxes, sal_processed

class ILSVRC_map_full_aug(Dataset):
    def __init__(self, mode='train', return_path=False, N=None, prior = 'nips08',
                 img_h = input_h, img_w = input_w, normalize_feature=False, num_tgt_cls=ilsvrc_num_tgt_classes): #'train', 'test', 'val'

        if num_tgt_cls is not None:
            if num_tgt_cls not in ILSVRC_TGT_CLS:
                print('If being specified, the num of target classes must be in {256, 385}.')
                raise NotImplementedError
        self.num_tgt_cls = num_tgt_cls
        self.mode =  mode
        self.path_dataset = PATH_ILSVRC
        self.path_images = os.path.join(self.path_dataset, 'images', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features_2', self.mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500', self.mode)
        self.path_saliency = os.path.join(self.path_dataset, 'images', self.mode + '_' + prior)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        self.normalize_feature = normalize_feature

        # get list images
        with open(os.path.join(self.path_dataset, '%s_list.txt'%self.mode), 'r') as f:
            tmp = f.readlines()
        list_names = [l.split(' ')[0] for l in tmp] # .JPEG
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        labels = [int(l.split(' ')[1]) for l in tmp]
        if self.num_tgt_cls is not None:
            self.lb_mappings = pickle.load(open(
                os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_train_{}.pickle'.format(num_tgt_cls)), 'rb'))
            # self.lb_mappings = pickle.load(open(
            #     os.path.join(PATH_ILSVRC, 'WordTree', 'lb_mappings_{}_{}.pickle'.format(self.mode, num_tgt_cls)), 'rb'))
            # # tmp = [lb_mappings[l] for l in labels]
            # labels = tmp.copy()
            # assert max(labels)==num_tgt_cls-1

        self.labels = labels
        if N is not None:
            self.list_names = list_names[:N]
            self.labels = labels[:N]

        # self.seq = Sequence([RandomHSV(10, 10, 10),
        #                      RandomHorizontalFlip(),  # p=0.5
        #                      RandomScale(0.05, diff=True),
        #                      RandomTranslate(0.05, diff=True),
        #                      RandomRotate(5),
        #                      RandomShear(0.05)]
        #                     )  # _aug3

        self.seq = RandomRotate(5)  # _aug7

        # if mode=='train':
        #     random.shuffle(self.list_names)

        # embed()
        print("Init ILSVRC full dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.JPEG')
        sal_path = os.path.join(self.path_saliency, self.list_names[index] + '.png')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        saliency = cv2.imread(sal_path, 0)
        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        if boxes.shape[0] == 0:
            img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)
            boxes_ = np.zeros_like(boxes)
            boxes_[:, 0] = boxes[:, 0] * self.img_w
            boxes_[:, 2] = boxes[:, 2] * self.img_w
            boxes_[:, 1] = boxes[:, 1] * self.img_h
            boxes_[:, 3] = boxes[:, 3] * self.img_h

        else:
            boxes[:, 0] = boxes[:, 0] * image.shape[1]
            boxes[:, 2] = boxes[:, 2] * image.shape[1]
            boxes[:, 1] = boxes[:, 1] * image.shape[0]
            boxes[:, 3] = boxes[:, 3] * image.shape[0]

            image_, saliency_, boxes_ = self.seq(image.copy(), saliency.copy(), boxes.copy())

            img_processed, sal_processed = imageProcessing(image_, saliency_, h=self.img_h, w=self.img_w)

            # np.clip(boxes_[:, 0], 0., image.shape[1], out=boxes_[:, 0])
            # np.clip(boxes_[:, 2], 0., image.shape[1], out=boxes_[:, 2])
            # np.clip(boxes_[:, 1], 0., image.shape[0], out=boxes_[:, 1])
            # np.clip(boxes_[:, 3], 0., image.shape[0], out=boxes_[:, 3])

            # boxes_[boxes_[:, 0] < 0., 0] = 0.
            # boxes_[boxes_[:, 0] > image.shape[1], 0] = image.shape[1]
            # boxes_[boxes_[:, 2] < boxes_[:, 0], 2] = boxes_[:, 0]
            # boxes_[boxes_[:, 2] > image.shape[1], 2] = image.shape[1]

            # boxes_[boxes_[:, 1] < 0., 1] = 0.
            # boxes_[boxes_[:, 1] > image.shape[0], 1] = image.shape[0]
            # boxes_[boxes_[:, 3] < boxes_[:, 1], 3] = boxes_[:, 1]
            # boxes_[boxes_[:, 3] > image.shape[0], 3] = image.shape[0]

            np.clip(boxes_[:, 0], 0., image.shape[1], out=boxes_[:, 0])
            np.clip(boxes_[:, 2], 0., image.shape[1], out=boxes_[:, 2])
            boxes_[:, 1] = np.minimum(np.maximum(boxes_[:, 0], boxes_[:, 1]), image.shape[0])
            boxes_[:, 3] = np.minimum(np.maximum(boxes_[:, 2], boxes_[:, 3]), image.shape[0])

            boxes_[:, 0] = boxes_[:, 0] / image.shape[1] * self.img_w
            boxes_[:, 2] = boxes_[:, 2] / image.shape[1] * self.img_w
            boxes_[:, 1] = boxes_[:, 1] / image.shape[0] * self.img_h
            boxes_[:, 3] = boxes_[:, 3] / image.shape[0] * self.img_h


        # load features and transfor to geometric Data
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max == 0.] = 1e-8
        #     box_features = box_features / box_f_max

        if self.num_tgt_cls is not None:
            label = self.lb_mappings[self.labels[index]][0]
        else:
            label = self.labels[index]

        if self.return_path:
            return img_processed, label, boxes_, sal_processed, self.list_names[index]
        else:
            return img_processed, label, boxes_, sal_processed


# ===MIT1003============================
def resize_fixation(img, rows=output_h, cols=output_w):
    out = np.zeros((rows, cols))
    factor_scale_r = rows*1.0 / img.shape[0]
    factor_scale_c = cols*1.0 / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r >= rows:
            r -= rows-1
        if c >= cols:
            c -= cols-1
        out[r, c] = 1

    return out

# preserve aspect ratio
def fixationProcessing(fix_img, h=output_h, w=output_w):
    fix = np.zeros((h, w))

    original_shape = fix_img.shape
    rows_rate = original_shape[0] * 1.0 / h
    cols_rate = original_shape[1] * 1.0 / w

    if rows_rate > cols_rate:
        new_cols = int((original_shape[1] * float(h)) // original_shape[0])
        fix_img = resize_fixation(fix_img, rows=h, cols=new_cols)
        if new_cols > w:
            new_cols = w
        fix[:, ((fix.shape[1] - new_cols) // 2):((fix.shape[1] - new_cols) // 2 + new_cols)] = fix_img
    else:
        new_rows = int((original_shape[0] * float(w)) // original_shape[1])
        fix_img = resize_fixation(fix_img, rows=new_rows, cols=w)
        if new_rows > h:
            new_rows = h
        fix[((fix.shape[0] - new_rows) // 2):((fix.shape[0] - new_rows) // 2 + new_rows), :] = fix_img

    return fix

class MIT1003_full(Dataset):
    def __init__(self, return_path=False, N=None,
                 img_h = det_input_h, img_w = det_input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_MIT1003
        self.path_images = os.path.join(self.path_dataset, 'ALLSTIMULI')
        # self.path_features = os.path.join(self.path_dataset, 'features_2')
        # self.path_features = os.path.join(self.path_dataset, 'features')
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500')
        self.path_saliency = os.path.join(self.path_dataset, 'ALLFIXATIONMAPS')
        self.path_fixation = os.path.join(self.path_dataset, 'ALLFIXATIONS')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        # embed()
        print("Init MIT1003 full dataset.")
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpeg')
        sal_path = os.path.join(self.path_saliency, self.list_names[index]+'_fixMap.jpg')
        fix_path = os.path.join(self.path_fixation, self.list_names[index]+'_fixPts.jpg')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        # image = cv2.imread(rgb_ima)
        saliency = cv2.imread(sal_path, 0)

        fixation = cv2.imread(fix_path, 0)
        fix_processed = fixationProcessing(fixation)

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        # -------------------------------
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max == 0.] = 1e-8
        #     box_features = box_features / box_f_max
        # boxes = np.load(os.path.join(self.path_features, '{}_boxes.npy'.format(self.list_names[index])))
        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.return_path:
            return img_processed, boxes, sal_processed, fix_processed, self.list_names[index]
        else:
            return img_processed, boxes, sal_processed, fix_processed

# ==========PASCAL-S==================
#=== can use collate_fn_mit1003, they share the save output format =====================
class PASCAL_full(Dataset):
    def __init__(self, return_path=False, N=None,
                 img_h = input_h, img_w = input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_PASCAL
        self.path_images = os.path.join(self.path_dataset, 'images')
        # self.path_features = os.path.join(self.path_dataset, 'features_2')
        # self.path_features = os.path.join(self.path_dataset, 'features')
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500')
        self.path_saliency = os.path.join(self.path_dataset, 'maps')
        self.path_fixation = os.path.join(self.path_dataset, 'fixation')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        # embed()
        print("Init PASCAL-S full dataset.")
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        sal_path = os.path.join(self.path_saliency, self.list_names[index]+'.png')
        fix_path = os.path.join(self.path_fixation, self.list_names[index]+'.png')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        # image = cv2.imread(rgb_ima)
        saliency = cv2.imread(sal_path, 0)

        fixation = cv2.imread(fix_path, 0)
        fix_processed = fixationProcessing(fixation)

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        # load features and transfor to geometric Data
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max == 0.] = 1e-8
        #     box_features = box_features / box_f_max
        # boxes = np.load(os.path.join(self.path_features, '{}_boxes.npy'.format(self.list_names[index])))
        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.return_path:
            return img_processed, boxes, sal_processed, fix_processed, self.list_names[index]
        else:
            return img_processed, boxes, sal_processed, fix_processed

# ========MIT300====================
class MIT300_full(Dataset):
    def __init__(self, return_path=False, N=None,
                 img_h = det_input_h, img_w = det_input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_MIT300
        self.path_images = os.path.join(self.path_dataset, 'images')
        # self.path_features = os.path.join(self.path_dataset, 'features_2')
        # self.path_features = os.path.join(self.path_dataset, 'features')
        self.edge_boxes = os.path.join(self.path_dataset, 'eb500')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        # self.normalize_feature = normalize_feature

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        # embed()
        print("Init MIT300 full dataset.")
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # Image and saliency map paths
        rgb_ima = os.path.join(self.path_images, self.list_names[index]+'.jpg')
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '_bboxes.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')

        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA).astype(np.float32)

        img_processed = transform(image / 255.)

        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]
        # print('boxes bf', boxes.max(), boxes.min())
        # load features and transfor to geometric Data
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max == 0.] = 1e-8
        #     box_features = box_features / box_f_max
        # boxes = np.load(os.path.join(self.path_features, '{}_boxes.npy'.format(self.list_names[index])))
        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h
        # print('boxes aft', boxes.max(), boxes.min())

        if self.return_path:
            return img_processed, boxes, self.list_names[index]
        else:
            return img_processed, boxes


# ========== collate_fn =========================
def gen_edge_index(n_node):
    edge_index = list(range(n_node))
    source_idx = np.array(edge_index).repeat(n_node)
    des_idx = np.tile(edge_index, n_node)
    edge_index_tensor = torch.from_numpy(np.concatenate([np.expand_dims(source_idx, 0), np.expand_dims(des_idx, 0)], 0))

    return  edge_index_tensor
# MS_COCO (no sal maps, multi labels)
def collate_fn_coco_rn(batch): # batch: image, label, boxes(, image_name)
    images = list()
    labels = list()
    # box_features = list()

    boxes_nums = np.array([X[2].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))
    # box_features_batch = np.zeros((len(batch), max_boxes_num, batch[0][2].shape[-1]))-float('Inf')

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        labels.append(X[1].unsqueeze(0))

        # x = torch.from_numpy(X[2])
        # y = None
        # edge_index = gen_edge_index(x.size(0))
        #
        # if boxes_nums[i]>0:
        #     box_features.append(Data(x=x, y=y, edge_index=edge_index))

        # box_features_batch[i, :boxes_nums[i], :] = X[2]
        boxes_batch[i, :boxes_nums[i], :] = X[2]

    images_batch = torch.cat(images, dim=0)
    labels_batch = torch.cat(labels, dim=0)
    # if len(box_features) > 0:
    #     box_features_batch = Batch.from_data_list(box_features)
    # else:
    #     box_features_batch = None
    # box_features_batch = torch.FloatTensor(box_features_batch)
    boxes_batch = torch.FloatTensor(boxes_batch)
    boxes_nums_batch = torch.LongTensor(boxes_nums)

    if len(batch[0]) == 3:
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch

    elif len(batch[0]) == 4:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, image_names_batch

# MS_COCO_map (no sal maps, multi labels)
def collate_fn_coco_map_rn(batch): # batch: image, label, boxes(, image_name)
    images = list()
    labels = list()
    rf_maps = list()

    boxes_nums = np.array([X[2].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))
    # box_features_batch = np.zeros((len(batch), max_boxes_num, batch[0][2].shape[-1]))-float('Inf')

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        labels.append(X[1].unsqueeze(0))

        # x = torch.from_numpy(X[2])
        # y = None
        # edge_index = gen_edge_index(x.size(0))
        #
        # if boxes_nums[i]>0:
        #     box_features.append(Data(x=x, y=y, edge_index=edge_index))

        # box_features_batch[i, :boxes_nums[i], :] = X[2]
        boxes_batch[i, :boxes_nums[i], :] = X[2]

        rf_maps.append(X[3].unsqueeze(0))

    images_batch = torch.cat(images, dim=0)
    labels_batch = torch.cat(labels, dim=0)
    # if len(box_features) > 0:
    #     box_features_batch = Batch.from_data_list(box_features)
    # else:
    #     box_features_batch = None
    # box_features_batch = torch.FloatTensor(box_features_batch)
    boxes_batch = torch.FloatTensor(boxes_batch)
    boxes_nums_batch = torch.LongTensor(boxes_nums)
    rf_maps_batch = torch.cat(rf_maps, dim=0)

    if len(batch[0]) == 4:
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, rf_maps_batch

    elif len(batch[0]) == 5:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, rf_maps_batch, image_names_batch

def collate_fn_coco_map_rn_multiscale(batch): # batch: image, label, boxes(, image_name)
    scale = random.choice(scales)

    images = list()
    labels = list()
    rf_maps = list()

    boxes_nums = np.array([X[2].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))
    # box_features_batch = np.zeros((len(batch), max_boxes_num, batch[0][2].shape[-1]))-float('Inf')

    for i, X in enumerate(batch):
        images.append(F.interpolate(X[0].unsqueeze(0), scale_factor=scale))
        labels.append(X[1].unsqueeze(0))

        # x = torch.from_numpy(X[2])
        # y = None
        # edge_index = gen_edge_index(x.size(0))
        #
        # if boxes_nums[i]>0:
        #     box_features.append(Data(x=x, y=y, edge_index=edge_index))

        # box_features_batch[i, :boxes_nums[i], :] = X[2]
        boxes_batch[i, :boxes_nums[i], :] = X[2]*scale

        rf_maps.append(F.interpolate(X[3].unsqueeze(0), scale_factor=scale))

    images_batch = torch.cat(images, dim=0)
    labels_batch = torch.cat(labels, dim=0)
    # if len(box_features) > 0:
    #     box_features_batch = Batch.from_data_list(box_features)
    # else:
    #     box_features_batch = None
    # box_features_batch = torch.FloatTensor(box_features_batch)
    boxes_batch = torch.FloatTensor(boxes_batch)
    boxes_nums_batch = torch.LongTensor(boxes_nums)
    rf_maps_batch = torch.cat(rf_maps, dim=0)

    if len(batch[0]) == 4:
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, rf_maps_batch

    elif len(batch[0]) == 5:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, rf_maps_batch, image_names_batch

# SALICON (with sal maps, multiple labels)
def collate_fn_salicon_rn(batch): # batch: image, label, boxes, sal_map, fix_map(, image_name)
    images = list()
    labels = list()
    # box_features = list()

    boxes_nums = np.array([X[2].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))
    # box_features_batch = np.zeros((len(batch), max_boxes_num, batch[0][2].shape[-1]))

    sal_maps = list()
    fix_maps = list()

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        labels.append(X[1].unsqueeze(0))

        #x = torch.from_numpy(X[2])
        #y = None
        #edge_index = gen_edge_index(x.size(0))
        #if boxes_nums[i] > 0:
        #    box_features.append(Data(x=x, y=y, edge_index=edge_index))

        # box_features_batch[i, :boxes_nums[i], :] = X[2]

        boxes_batch[i, :boxes_nums[i], :] = X[2]

        sal_maps.append(X[3].unsqueeze(0))
        fix_maps.append(torch.from_numpy(X[4]).unsqueeze(0))

    images_batch = torch.cat(images, dim=0)
    labels_batch = torch.cat(labels, dim=0)
    #if len(box_features) > 0:
    #    box_features_batch = Batch.from_data_list(box_features)
    #else:
    #    box_features_batch = None

    # box_features_batch = torch.FloatTensor(box_features_batch)
    boxes_batch = torch.FloatTensor(boxes_batch)
    boxes_nums_batch = torch.LongTensor(boxes_nums)

    sal_maps_batch = torch.cat(sal_maps, dim=0)
    fix_maps_batch = torch.cat(fix_maps, dim=0)

    if len(batch[0]) == 5:
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, sal_maps_batch, fix_maps_batch

    elif len(batch[0]) == 6:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, sal_maps_batch, fix_maps_batch, image_names_batch

# ILSVRC (no sal maps, single label)
def collate_fn_ilsvrc_rn(batch): # batch: image, label, boxes, box_nums(, image_name)
    images = list()
    labels = list()
    # box_features = list()

    boxes_nums = np.array([X[2].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))
    # box_features_batch = np.zeros((len(batch), max_boxes_num, batch[0][2].shape[-1])) - float('Inf')

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        labels.append(X[1])

        # x = torch.from_numpy(X[2])
        # y = None
        # edge_index = gen_edge_index(x.size(0))
        #
        # if boxes_nums[i]>0:
        #     box_features.append(Data(x=x, y=y, edge_index=edge_index))

        # box_features_batch[i, :boxes_nums[i], :] = X[2]
        boxes_batch[i, :boxes_nums[i], :] = X[2]

    images_batch = torch.cat(images, dim=0)
    labels_batch = torch.from_numpy(np.array(labels))
    # if len(box_features) > 0:
    #     box_features_batch = Batch.from_data_list(box_features)
    # else:
    #     box_features_batch = None

    # box_features_batch = torch.FloatTensor(box_features_batch)
    boxes_batch = torch.FloatTensor(boxes_batch)
    boxes_nums_batch = torch.LongTensor(boxes_nums)

    if len(batch[0]) == 3:
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch

    elif len(batch[0]) == 4:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, image_names_batch

def collate_fn_ilsvrc_map_rn(batch): # batch: image, label, boxes, box_nums, rf_maps(, image_name)
    images = list()
    labels = list()
    rf_maps = list()
    # box_features = list()

    boxes_nums = np.array([X[2].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))
    # box_features_batch = np.zeros((len(batch), max_boxes_num, batch[0][2].shape[-1])) - float('Inf')

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        labels.append(X[1])

        # x = torch.from_numpy(X[2])
        # y = None
        # edge_index = gen_edge_index(x.size(0))
        #
        # if boxes_nums[i]>0:
        #     box_features.append(Data(x=x, y=y, edge_index=edge_index))

        # box_features_batch[i, :boxes_nums[i], :] = X[2]
        boxes_batch[i, :boxes_nums[i], :] = X[2]

        rf_maps.append(X[3].unsqueeze(0))

    images_batch = torch.cat(images, dim=0)
    labels_batch = torch.from_numpy(np.array(labels))
    # if len(box_features) > 0:
    #     box_features_batch = Batch.from_data_list(box_features)
    # else:
    #     box_features_batch = None

    # box_features_batch = torch.FloatTensor(box_features_batch)
    boxes_batch = torch.FloatTensor(boxes_batch)
    boxes_nums_batch = torch.LongTensor(boxes_nums)
    rf_maps_batch = torch.cat(rf_maps, dim=0)

    if len(batch[0]) == 4:
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, rf_maps_batch

    elif len(batch[0]) == 5:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, rf_maps_batch, image_names_batch

# MIT1003 (with sal maps, no labels)
def collate_fn_mit1003_rn(batch): # batch: image, boxes, sal_map, fix_map(, image_name)
    images = list()
    # box_features = list()

    boxes_nums = [X[1].shape[0] for X in batch]
    max_boxes_num = np.max(np.array(boxes_nums))
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))
    # box_features_batch = np.zeros((len(batch), max_boxes_num, batch[0][1].shape[-1]))

    sal_maps = list()
    fix_maps = list()

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))

        # x = torch.from_numpy(X[1])
        # y = None
        # edge_index = gen_edge_index(x.size(0))
        # if boxes_nums[i] > 0:
        #     box_features.append(Data(x=x, y=y, edge_index=edge_index))

        # box_features_batch[i, :boxes_nums[i], :] = X[1]

        boxes_batch[i, :boxes_nums[i], :] = X[1]

        sal_maps.append(X[2].unsqueeze(0))
        fix_maps.append(torch.from_numpy(X[3]).unsqueeze(0))

    images_batch = torch.cat(images, dim=0)
    # if len(box_features)>0:
    #     box_features_batch = Batch.from_data_list(box_features)
    # else:
    #     box_features_batch = None

    # box_features_batch = torch.FloatTensor(box_features_batch)
    boxes_batch = torch.FloatTensor(boxes_batch)

    sal_maps_batch = torch.cat(sal_maps, dim=0)
    fix_maps_batch = torch.cat(fix_maps, dim=0)

    if len(batch[0]) == 4:
        return images_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch

    elif len(batch[0]) == 5:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_batch

# MIT300 (no sal maps, no labels)
def collate_fn_mit300_rn(batch): # batch: image, boxes(, image_name)
    images = list()
    # box_features = list()

    boxes_nums = [X[1].shape[0] for X in batch]
    max_boxes_num = np.max(np.array(boxes_nums))
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))

        # x = torch.from_numpy(X[1])
        # y = None
        # edge_index = gen_edge_index(x.size(0))
        # if boxes_nums[i] > 0:
        #     box_features.append(Data(x=x, y=y, edge_index=edge_index))
        # print('X[1]', X[1].max(), X[1].min())
        boxes_batch[i, :boxes_nums[i], :] = X[1]

    images_batch = torch.cat(images, dim=0)
    # if len(box_features)>0:
    #     box_features_batch = Batch.from_data_list(box_features)
    # else:
    #     box_features_batch = None

    boxes_batch = torch.FloatTensor(boxes_batch)

    if len(batch[0]) == 2:
        return images_batch, boxes_batch, boxes_nums

    elif len(batch[0]) == 3:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, boxes_batch, boxes_nums, image_names_batch


# can change box loading to real time edge box calculation, then return to function
def main2():
    # --------------collate_fn_ilsvrc-----------------------
    # ds_train = ILSVRC_full(N=4, return_path=True, mode='val')  # batch: image, label, box_feature, boxes(, image_name)
    # # ds_train = SALICON(mode='test', N=1, return_path=True)
    # # if ilsvrc_num_tgt_classes is not None:
    # #     train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_coco_rn,
    # #                               shuffle=True, num_workers=2)
    # # else:
    # #     train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_ilsvrc_rn,
    # #                               shuffle=True, num_workers=2)
    # train_dataloader = DataLoader(ds_train, batch_size=2, collate_fn=collate_fn_ilsvrc_rn,
    #                               shuffle=True, num_workers=2)
    #
    #
    # for i, X in enumerate(train_dataloader):
    #     print('images', X[0].size())
    #     print('labels', X[1].size())
    #     print('boxes', X[2])
    #     print('boxes_nums', X[3])
    #     print('image_names', X[-1])
    #     print(i)

    ds_train = ILSVRC_map_full_aug(N=4, return_path=True, mode='train')  # image, label, boxes, box_nums, rf_maps(, image_name)
    # ds_train = SALICON(mode='test', N=1, return_path=True)
    # if ilsvrc_num_tgt_classes is not None:
    #     train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_coco_map_rn,
    #                               shuffle=True, num_workers=2)
    # else:
    #     train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_ilsvrc_map_rn,
    #                               shuffle=True, num_workers=2)

    train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_ilsvrc_map_rn,
                                  shuffle=True, num_workers=2)


    for i, X in enumerate(train_dataloader):
        print('images', X[0].size())
        print('labels', X[1].size())
        print('boxes', X[2])
        print('boxes_nums', X[3])
        print('rf_maps', X[4].size())
        print('image_names', X[-1])
        print(i)

    # -----------------collate_fn_salicon--------------------------------
    # ds_train = SALICON_full(N=4, return_path=True, mode='val')  # batch: image, label, box_feature, boxes, sal_map, fix_map(, image_name)
    # # ds_train = SALICON(mode='test', N=1, return_path=True)
    # train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_salicon_rn,
    #                               shuffle=False, num_workers=2)
    # for i, X in enumerate(train_dataloader):
    #     print('images', X[0].size())
    #     print'labels', (X[1].size())
    #     print('num_graphs', X[2].size())
    #     # print('box_features', X[3].size())
    #     print('boxes_nums', X[3])
    #     print('sal_map', X[4].size())
    #     print('fix_map', X[5].size())
    #     print('image_names', X[-1])
    #     print(i)

    # -----------------collate_fn_val--------------------------------
    # ds_train = MS_COCO_map_full_aug(mode='train', N=4, return_path=True, img_h=input_h, img_w=input_w)
    # # # batch: image, label, box_feature, boxes(, image_name)
    # # # ds_train = SALICON(mode='test', N=1, return_path=True)
    # train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_coco_map_rn,
    #                               shuffle=True, num_workers=2)
    # for i, X in enumerate(train_dataloader):
    #     print('images', X[0].size())
    #     print('labels', X[1].size())
    #     # print('num_graphs', X[2].num_graphs)
    #     print('boxes', X[2].int())
    #     print('boxes_nums', X[3])
    #     print('rf_maps', X[4].size())
    #     print('image_names', X[-1])
    #     print(i)

    # # -----------------collate_fn_val--------------------------------
    # ds_train = MS_COCO_full(mode='val', N=4, return_path=False, img_h=input_h, img_w=input_w)
    # # batch: image, label, box_feature, boxes(, image_name)
    # # ds_train = SALICON(mode='test', N=1, return_path=True)
    # train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_coco_rn,
    #                               shuffle=True, num_workers=2)
    # for i, X in enumerate(train_dataloader):
    #     print('images', X[0].size())
    #     print('labels', X[1].size())
    #     # print('box_features', X[2].size())
    #     # print('box_features max ',X[2].max(), 'box_features min ', X[2].min())
    #     # print('num_graphs', X[2].num_graphs)
    #     print('boxes', X[2].size())
    #     print('boxes_nums', X[3])
    #     print('image_names', X[-1])
    #     print(i)
    #
    # # --------------collate_fn_mit1003-----------------------------------
    # ds_train = MIT1003_full(N=4, return_path=True)  # batch: image, box_feature, boxes, sal_map, fix_map(, image_name)
    # train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_mit1003_rn,
    #                               shuffle=True, num_workers=2)
    # for i, X in enumerate(train_dataloader):
    #     print('images', X[0].size())
    #     # print'labels', (X[1].size())
    #     print('num_graphs', X[1].size())
    #     # print('box_features', X[2].size())
    #     print('boxes_nums', X[2])
    #     print('sal_map', X[3].size())
    #     print('fix_map', X[4].size())
    #     print('image_names', X[-1])
    #     print(i)

    # --------------collate_fn_mit300-----------------------------------
    #ds_train = MIT300_full(N=4, return_path=True)  # batch: image, box_feature, boxes(, image_name)
    #train_dataloader = DataLoader(ds_train, batch_size=2,  collate_fn=collate_fn_mit300_rn,
    #                              shuffle=False, num_workers=2)
    #for i, X in enumerate(train_dataloader):
    #    print('images', X[0].size())
    #    # print'labels', (X[1].size())
    #    print('boxes', X[1].int())
    #    # print('box_features', X[2].size())
    #    print('boxes_nums', X[2])
    #    print('image_names', X[-1])
    #    print(i)


if __name__ == '__main__':
    main2()