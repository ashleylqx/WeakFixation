import pickle
import scipy.misc
import scipy.io
from scipy.ndimage import gaussian_filter


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config_new import *

from data_aug.data_aug_map import *
from data_aug.bbox_util import *


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # rgb_mean,
                             std=[0.229, 0.224, 0.225])])

prop_m = 'eb500' # type of bounding boxes

def imageProcessing(image, saliency, h=input_h, w=input_w):
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    saliency = cv2.resize(saliency, (output_w, output_h), interpolation=cv2.INTER_AREA).astype(np.float32)

    return transform(image/255.), torch.FloatTensor(saliency/255.)

def resize_fixation_mat(fix, rows=output_h, cols=output_w):
    out = np.zeros((rows, cols))
    resolution = fix["resolution"][0]

    factor_scale_r = rows*1.0 / resolution[0]
    factor_scale_c = cols*1.0 / resolution[1]

    coords = []
    for gaze in fix["gaze"][0]: # previous (1, N, 3)
        coords.extend(gaze[2])

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
# 'train', 'test', 'val'
class SALICON_full(Dataset):
    def __init__(self, mode='train', return_path=False, N=None,
                 img_h=input_h, img_w=input_w):
        self.path_dataset = PATH_SALICON
        self.path_images = os.path.join(self.path_dataset, 'images', mode)

        if mode=='val':
            self.edge_boxes = os.path.join(self.path_dataset, prop_m, mode)
        else:
            self.edge_boxes = os.path.join(PATH_COCO, 'train2014_%s' % prop_m)

        self.path_saliency = os.path.join(self.path_dataset, 'maps', mode)
        self.path_fixation = os.path.join(self.path_dataset, 'fixations', mode)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        self.imgNsToCat = pickle.load(open(os.path.join(PATH_COCO, 'imgNsToCat_{}.p'.format(mode)), "rb"))

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

        fixation = scipy.io.loadmat(fix_path)
        fix_processed = fixationProcessing_mat(fixation)

        boxes_tmp = scipy.io.loadmat(box_path)['boxes'][:MAX_BNUM, :]
        boxes = np.zeros_like(boxes_tmp).astype(np.float32)

        if os.path.exists(sal_path):
            saliency = cv2.imread(sal_path, 0)
        else:
            saliency = gaussian_filter(fix_processed, sigma=5)

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        label_indices = self.imgNsToCat[self.list_names[index]]

        label = torch.zeros(coco_num_classes)
        if len(label_indices)>0:
            label[label_indices] = 1
        else:
            label[0] = 1

        boxes[:, 0] = boxes_tmp[:, 1] * 1.0 / img_WIDTH * self.img_w  # x1
        boxes[:, 2] = boxes_tmp[:, 3] * 1.0 / img_WIDTH * self.img_w  # x2
        boxes[:, 1] = boxes_tmp[:, 0] * 1.0 / img_HEIGHT * self.img_h  # y1
        boxes[:, 3] = boxes_tmp[:, 2] * 1.0 / img_HEIGHT * self.img_h  # y2

        if self.return_path:
            return img_processed, label, boxes, sal_processed, fix_processed, self.list_names[index]
        else:
            return img_processed, label, boxes, sal_processed, fix_processed

#### *****
# 'train', 'test', 'val'
class MS_COCO_map_full_aug(Dataset):
    def __init__(self, mode='train', return_path=False, N=None, prior='nips08',
                 img_h=input_h, img_w=input_w):
        self.path_dataset = PATH_COCO
        self.path_images = os.path.join(self.path_dataset, mode+'2014')

        self.edge_boxes = os.path.join(self.path_dataset, mode + '2014_%s' % prop_m)

        self.path_saliency = os.path.join(self.path_dataset, mode + '2014_%s' % prior)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])

        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        self.imgNsToCat = pickle.load(open(os.path.join(PATH_COCO, 'imgNsToCat_{}.p'.format(mode)), "rb"))

        self.seq = RandomRotate(5)

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

        boxes_tmp = scipy.io.loadmat(box_path)['boxes'][:MAX_BNUM, :]

        if boxes_tmp.shape[0]==0:
            img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)
            boxes_ = np.zeros_like(boxes_tmp).astype(np.float32)

            boxes_[:, 0] = boxes_tmp[:, 1] * 1.0 / img_WIDTH * self.img_w # x1
            boxes_[:, 2] = boxes_tmp[:, 3] * 1.0 / img_WIDTH * self.img_w # x2
            boxes_[:, 1] = boxes_tmp[:, 0] * 1.0 / img_HEIGHT * self.img_h # y1
            boxes_[:, 3] = boxes_tmp[:, 2] * 1.0 / img_HEIGHT * self.img_h # y2
        else:
            boxes = np.zeros_like(boxes_tmp).astype(np.float32)

            boxes[:, 0] = boxes_tmp[:, 1] * 1.0 #* image.shape[1] # x1
            boxes[:, 2] = boxes_tmp[:, 3] * 1.0 #* image.shape[1] # x2
            boxes[:, 1] = boxes_tmp[:, 0] * 1.0 #* image.shape[0] # y1
            boxes[:, 3] = boxes_tmp[:, 2] * 1.0 #* image.shape[0] # y2
            # print('load_data, boxes', boxes.dtype)
            image_, saliency_, boxes_ = self.seq(image.copy(), saliency.copy(), boxes.copy())

            img_processed, sal_processed = imageProcessing(image_, saliency_, h=self.img_h, w=self.img_w)

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
# 'train', 'test', 'val'
class MIT1003_full(Dataset):
    def __init__(self, return_path=False, N=None, img_h=det_input_h, img_w=det_input_w):
        self.path_dataset = PATH_MIT1003
        self.path_images = os.path.join(self.path_dataset, 'ALLSTIMULI')

        self.edge_boxes = os.path.join(self.path_dataset, prop_m)
        self.path_saliency = os.path.join(self.path_dataset, 'ALLFIXATIONMAPS')
        self.path_fixation = os.path.join(self.path_dataset, 'ALLFIXATIONS')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

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
        img_HEIGHT, img_WIDTH = image.shape[0], image.shape[1]

        saliency = cv2.imread(sal_path, 0)

        fixation = cv2.imread(fix_path, 0)
        fix_processed = fixationProcessing(fixation)

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

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
# 'train', 'test', 'val'
class PASCAL_full(Dataset):
    def __init__(self, return_path=False, N=None, img_h=input_h, img_w=input_w):
        self.path_dataset = PATH_PASCAL
        self.path_images = os.path.join(self.path_dataset, 'images')

        self.edge_boxes = os.path.join(self.path_dataset, prop_m)
        self.path_saliency = os.path.join(self.path_dataset, 'maps')
        self.path_fixation = os.path.join(self.path_dataset, 'fixation')
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

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

        saliency = cv2.imread(sal_path, 0)

        fixation = cv2.imread(fix_path, 0)
        fix_processed = fixationProcessing(fixation)

        img_processed, sal_processed = imageProcessing(image, saliency, h=self.img_h, w=self.img_w)

        boxes = scipy.io.loadmat(box_path)['bboxes'][:MAX_BNUM, :]

        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.return_path:
            return img_processed, boxes, sal_processed, fix_processed, self.list_names[index]
        else:
            return img_processed, boxes, sal_processed, fix_processed


# ========MIT300====================
# 'train', 'test', 'val'
class MIT300_full(Dataset):
    def __init__(self, return_path=False, N=None, img_h=det_input_h, img_w=det_input_w):
        self.path_dataset = PATH_MIT300
        self.path_images = os.path.join(self.path_dataset, 'images')

        self.edge_boxes = os.path.join(self.path_dataset, prop_m)
        self.return_path = return_path

        self.img_h = img_h
        self.img_w = img_w

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

        boxes[:, 0] = boxes[:, 0] * self.img_w
        boxes[:, 2] = boxes[:, 2] * self.img_w
        boxes[:, 1] = boxes[:, 1] * self.img_h
        boxes[:, 3] = boxes[:, 3] * self.img_h

        if self.return_path:
            return img_processed, boxes, self.list_names[index]
        else:
            return img_processed, boxes


# batch: image, label, boxes(, image_name)
def collate_fn_coco_map_rn(batch):
    images = list()
    labels = list()
    rf_maps = list()

    boxes_nums = np.array([X[2].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        labels.append(X[1].unsqueeze(0))
        boxes_batch[i, :boxes_nums[i], :] = X[2]
        rf_maps.append(X[3].unsqueeze(0))

    images_batch = torch.cat(images, dim=0)
    labels_batch = torch.cat(labels, dim=0)

    boxes_batch = torch.FloatTensor(boxes_batch)
    boxes_nums_batch = torch.LongTensor(boxes_nums)
    rf_maps_batch = torch.cat(rf_maps, dim=0)

    if len(batch[0]) == 4:
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, rf_maps_batch

    elif len(batch[0]) == 5:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, labels_batch, boxes_batch, boxes_nums_batch, rf_maps_batch, image_names_batch


# SALICON (with sal maps, multiple labels)
# batch: image, label, boxes, sal_map, fix_map(, image_name)
def collate_fn_salicon_rn(batch):
    images = list()
    labels = list()

    boxes_nums = np.array([X[2].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))

    sal_maps = list()
    fix_maps = list()

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        labels.append(X[1].unsqueeze(0))
        boxes_batch[i, :boxes_nums[i], :] = X[2]
        sal_maps.append(X[3].unsqueeze(0))
        fix_maps.append(torch.from_numpy(X[4]).unsqueeze(0))

    images_batch = torch.cat(images, dim=0)
    labels_batch = torch.cat(labels, dim=0)

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
# batch: image, boxes, sal_map, fix_map(, image_name)
def collate_fn_mit1003_rn(batch):
    images = list()

    boxes_nums = np.array([X[1].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))

    sal_maps = list()
    fix_maps = list()

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        boxes_batch[i, :boxes_nums[i], :] = X[1]
        sal_maps.append(X[2].unsqueeze(0))
        fix_maps.append(torch.from_numpy(X[3]).unsqueeze(0))

    images_batch = torch.cat(images, dim=0)

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
# batch: image, boxes(, image_name)
def collate_fn_mit300_rn(batch):
    images = list()

    boxes_nums = np.array([X[1].shape[0] for X in batch])
    max_boxes_num = np.max(boxes_nums)
    boxes_batch = np.zeros((len(batch), max_boxes_num, 4))

    for i, X in enumerate(batch):
        images.append(X[0].unsqueeze(0))
        boxes_batch[i, :boxes_nums[i], :] = X[1]

    images_batch = torch.cat(images, dim=0)

    boxes_batch = torch.FloatTensor(boxes_batch)
    boxes_nums_batch = torch.LongTensor(boxes_nums)

    if len(batch[0]) == 2:
        return images_batch, boxes_batch, boxes_nums_batch

    elif len(batch[0]) == 3:
        image_names_batch = [X[-1] for X in batch]
        return images_batch, boxes_batch, boxes_nums_batch, image_names_batch

