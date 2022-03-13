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

Proposal_Methods = ['eb500', 'mcg']
prop_idx = 1
prop_m = Proposal_Methods[prop_idx]

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

##### ****
class SALICON_full(Dataset):
    def __init__(self, mode='train', return_path=False, N=None,
                 img_h = input_h, img_w = input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_SALICON
        self.path_images = os.path.join(self.path_dataset, 'images', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features_2', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        if mode=='val':
            self.edge_boxes = os.path.join(self.path_dataset, prop_m, mode)
        else:
            self.edge_boxes = os.path.join(PATH_COCO, 'train2014_%s' % prop_m)

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
        box_path = os.path.join(self.edge_boxes, self.list_names[index]+'.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        img_HEIGHT, img_WIDTH = image.shape[0], image.shape[1]
        # image = cv2.imread(rgb_ima)
        # saliency = cv2.imread(sal_path, 0)
        fixation = scipy.io.loadmat(fix_path)
        fix_processed = fixationProcessing_mat(fixation)
        if PRO_RATIO is None:
            boxes_tmp = scipy.io.loadmat(box_path)['boxes'][:MAX_BNUM, :]
        else:
            boxes_tmp_tmp = scipy.io.loadmat(box_path)['boxes']  # exlude props with area larger than PRO_RATIO
            # y1 x1 y2 x2, not normalized
            boxes_tmp = [box for box in boxes_tmp_tmp
                         if (box[2] - box[0])*1.0/img_HEIGHT * (box[3] - box[1])*1.0/img_WIDTH < PRO_RATIO]
            if len(boxes_tmp) > 0:
                boxes_tmp = np.vstack(boxes_tmp)
                boxes_tmp = boxes_tmp[:MAX_BNUM, :]
            else:
                boxes_tmp = np.zeros((0, boxes_tmp_tmp.shape[1]))
        boxes = np.zeros_like(boxes_tmp).astype(np.float32)

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
        # # x1, y1, x2, y2, normalized
        # boxes[:, 0] = boxes[:, 0] * self.img_w
        # boxes[:, 2] = boxes[:, 2] * self.img_w
        # boxes[:, 1] = boxes[:, 1] * self.img_h
        # boxes[:, 3] = boxes[:, 3] * self.img_h

        # y1, x1, y2, x2 not normalized
        # swap and normalize
        boxes[:, 0] = boxes_tmp[:, 1] * 1.0 / img_WIDTH * self.img_w  # x1
        boxes[:, 2] = boxes_tmp[:, 3] * 1.0 / img_WIDTH * self.img_w  # x2
        boxes[:, 1] = boxes_tmp[:, 0] * 1.0 / img_HEIGHT * self.img_h  # y1
        boxes[:, 3] = boxes_tmp[:, 2] * 1.0 / img_HEIGHT * self.img_h  # y2

        if self.return_path:
            return img_processed, label, boxes, sal_processed, fix_processed, self.list_names[index]
        else:
            return img_processed, label, boxes, sal_processed, fix_processed

#### *****
class MS_COCO_map_full_aug(Dataset):
    def __init__(self, mode='train', return_path=False, N=None, prior = 'nips08',
                 img_h = input_h, img_w = input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_COCO
        self.path_images = os.path.join(self.path_dataset, mode+'2014')
        # self.path_features = os.path.join(self.path_dataset, 'features_2', mode)
        # self.path_features = os.path.join(self.path_dataset, 'features', mode)
        self.edge_boxes = os.path.join(self.path_dataset, mode + '2014_%s' % prop_m)

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

        # self.seq = RandomHorizontalFlip() # _aug6
        self.seq = RandomRotate(5) # _aug7 ## BEST
        # self.seq = RandomRotate(10) # _aug7_2
        # self.seq = RandomScale(0.1, diff=True) # _aug8
        # self.seq = RandomScale(0.01, diff=True) # _aug8_2
        # self.seq = RandomScale(0.05, diff=True) # _aug8_3

        # self.seq = Sequence([ # RandomHSV(10, 10, 10),
        #                     RandomHorizontalFlip(), #p=0.5
        #                     # RandomScale(0.1, diff=True),
        #                     # RandomTranslate(0.1, diff=True),
        #                     RandomRotate(5)]
        #                     #RandomShear(0.1)]
        #                     ) # _aug9

        # self.seq = RandomTranslate(0.1, diff=True) # _aug10

        # self.seq = RandomHSV(5, 5, 5) # _aug11

        # self.seq = RandomShear(0.05) # _aug12


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
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB') # (h,w,c)
        img_HEIGHT, img_WIDTH = image.shape[0], image.shape[1]
        saliency = cv2.imread(sal_path, 0)
        if PRO_RATIO is None:
            boxes_tmp = scipy.io.loadmat(box_path)['boxes'][:MAX_BNUM, :]
        else:
            boxes_tmp_tmp = scipy.io.loadmat(box_path)['boxes']  # exlude props with area larger than PRO_RATIO
            # boxes_tmp = [box for box in boxes_tmp_tmp if (box[2] - box[0]) * (box[3] - box[1]) < PRO_RATIO]
            # y1 x1 y2 x2 not normalized
            boxes_tmp = [box for box in boxes_tmp_tmp if
                         (box[2] - box[0])*1.0/img_HEIGHT * (box[3] - box[1])*1.0/img_WIDTH < PRO_RATIO]
            if len(boxes_tmp) > 0:
                boxes_tmp = np.vstack(boxes_tmp)
                boxes_tmp = boxes_tmp[:MAX_BNUM, :]
            else:
                boxes_tmp = np.zeros((0, boxes_tmp_tmp.shape[1]))

        if boxes_tmp.shape[0]==0:
            img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)
            boxes_ = np.zeros_like(boxes_tmp).astype(np.float32)
            # y1 x1 y2 x2 not normalized
            # swap, normalize
            boxes_[:, 0] = boxes_tmp[:, 1] * 1.0 / img_WIDTH * self.img_w # x1
            boxes_[:, 2] = boxes_tmp[:, 3] * 1.0 / img_WIDTH * self.img_w # x2
            boxes_[:, 1] = boxes_tmp[:, 0] * 1.0 / img_HEIGHT * self.img_h # y1
            boxes_[:, 3] = boxes_tmp[:, 2] * 1.0 / img_HEIGHT * self.img_h # y2
        else:
            boxes = np.zeros_like(boxes_tmp).astype(np.float32)
            # print('load_data, boxes', boxes.dtype)
            # pdb.set_trace()
            # y1 x1 y2 x2 not normalized
            # swap, not normalized
            boxes[:, 0] = boxes_tmp[:, 1] * 1.0 #* image.shape[1] # x1
            boxes[:, 2] = boxes_tmp[:, 3] * 1.0 #* image.shape[1] # x2
            boxes[:, 1] = boxes_tmp[:, 0] * 1.0 #* image.shape[0] # y1
            boxes[:, 3] = boxes_tmp[:, 2] * 1.0 #* image.shape[0] # y2
            # print('load_data, boxes', boxes.dtype)
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
#### ****
class MIT1003_full(Dataset):
    def __init__(self, return_path=False, N=None,
                 img_h = det_input_h, img_w = det_input_w): #'train', 'test', 'val'

        self.path_dataset = PATH_MIT1003
        self.path_images = os.path.join(self.path_dataset, 'ALLSTIMULI')
        # self.path_features = os.path.join(self.path_dataset, 'features_2')
        # self.path_features = os.path.join(self.path_dataset, 'features')
        self.edge_boxes = os.path.join(self.path_dataset, prop_m)
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
            # self.list_names = list_names[-N:]

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
        box_path = os.path.join(self.edge_boxes, self.list_names[index] + '.jpeg.mat')

        image = scipy.misc.imread(rgb_ima, mode='RGB')
        img_HEIGHT, img_WIDTH = image.shape[0], image.shape[1]
        # image = cv2.imread(rgb_ima)
        saliency = cv2.imread(sal_path, 0)

        fixation = cv2.imread(fix_path, 0)
        fix_processed = fixationProcessing(fixation)

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        if PRO_RATIO is None:
            # boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]
            boxes = scipy.io.loadmat(box_path)['proposals'][0][0][0][:MAX_BNUM, :]
        else:
            # boxes_tmp = scipy.io.loadmat(box_path)['bboxes']  # exlude props with area larger than PRO_RATIO
            # boxes = [box for box in boxes_tmp if (box[2] - box[0]) * (box[3] - box[1]) < PRO_RATIO]
            # This is also x1, y1, x2, y2, not normalized
            boxes_tmp = scipy.io.loadmat(box_path)['proposals'][0][0][0]  # exlude props with area larger than PRO_RATIO
            boxes = [box for box in boxes_tmp if
                     (box[2] - box[0])*1.0/img_WIDTH * (box[3] - box[1])*1.0/img_HEIGHT < PRO_RATIO]
            if len(boxes) > 0:
                boxes = np.vstack(boxes)
                boxes = boxes[:MAX_BNUM, :]
            else:
                boxes = np.zeros((0, boxes_tmp.shape[1]))

        # -------------------------------
        # box_features = np.load(os.path.join(self.path_features, '{}_box_features.npy'.format(self.list_names[index])))
        # if self.normalize_feature:
        #     # box_features = box_features / (np.max(box_features, axis=-1, keepdims=True)+1e-8)
        #     box_f_max = np.max(box_features, axis=-1, keepdims=True)
        #     box_f_max[box_f_max == 0.] = 1e-8
        #     box_features = box_features / box_f_max
        # boxes = np.load(os.path.join(self.path_features, '{}_boxes.npy'.format(self.list_names[index])))
        # # x1, y1, x2, y2, normalized
        # boxes[:, 0] = boxes[:, 0] * self.img_w
        # boxes[:, 2] = boxes[:, 2] * self.img_w
        # boxes[:, 1] = boxes[:, 1] * self.img_h
        # boxes[:, 3] = boxes[:, 3] * self.img_h

        # x1, y1, x2, y2, not normalized
        # normalize
        boxes[:, 0] = boxes[:, 0] * 1.0 / img_WIDTH * self.img_w
        boxes[:, 2] = boxes[:, 2] * 1.0 / img_WIDTH * self.img_w
        boxes[:, 1] = boxes[:, 1] * 1.0 / img_HEIGHT * self.img_h
        boxes[:, 3] = boxes[:, 3] * 1.0 / img_HEIGHT * self.img_h

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
        self.edge_boxes = os.path.join(self.path_dataset, prop_m)
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
        self.edge_boxes = os.path.join(self.path_dataset, prop_m)
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


# MIT1003 (with sal maps, no labels)
def collate_fn_mit1003_rn(batch): # batch: image, boxes, sal_map, fix_map(, image_name)
    images = list()
    # box_features = list()

    boxes_nums = np.array([X[1].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
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
    boxes_nums_batch = torch.LongTensor(boxes_nums)

    sal_maps_batch = torch.cat(sal_maps, dim=0)
    fix_maps_batch = torch.cat(fix_maps, dim=0)

    if len(batch[0]) == 4:
        return images_batch, boxes_batch, boxes_nums_batch, sal_maps_batch, fix_maps_batch

    elif len(batch[0]) == 5:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, boxes_batch, boxes_nums_batch, sal_maps_batch, fix_maps_batch, image_names_batch

# MIT300 (no sal maps, no labels)
def collate_fn_mit300_rn(batch): # batch: image, boxes(, image_name)
    images = list()
    # box_features = list()

    boxes_nums = np.array([X[1].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
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
    boxes_nums_batch = torch.LongTensor(boxes_nums)

    if len(batch[0]) == 2:
        return images_batch, boxes_batch, boxes_nums_batch

    elif len(batch[0]) == 3:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, boxes_batch, boxes_nums_batch, image_names_batch

