# base_path = '/home/qx/'
base_path = '/home/hz1/QX/'
# base_path = '/research/dept2/qxlai/'
# base_path = '/raid/QX/'


input_h = 224 #224
input_w = 224 #224
img_area = input_h*input_w*1.0

det_input_h = 800
det_input_w = 800
# image_sizes = [(det_input_h, det_input_w)]
# img_det.size(-2)*img_det.size(-1)*1.
det_img_area = det_input_h*det_input_w*1.0

hm_k = 8 #25% from 32 samples

output_h = 56
output_w = 56

SALGAN_RESIZE = [224, 224]

train_log_interval = 400
eval_log_interval = 100
test_log_interval = 20

tb_log_interval = 50
# PATH_LOG = base_path + 'tbx_log/'
# PATH_LOG = base_path + 'WF/log/'
PATH_LOG = base_path + 'WF/log2/'
# SALICON
PATH_SALICON = base_path + 'DataSets/SALICON/'
bgr_mean = [103.939, 116.779, 123.68]
rgb_mean = [123.68, 116.779, 103.939]

# MIT1003
PATH_MIT1003 = base_path + 'DataSets/MIT1003/'

# MIT300
PATH_MIT300 = base_path + 'DataSets/MIT300/'

# COCO
PATH_COCO = base_path + 'DataSets/MS_COCO/'
coco_classes = ['__background__',  # always index 0
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
# https://blog.csdn.net/u014106566/article/details/95195333
coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

coco_num_classes = 91
# num_classes = len(coco_classes)

# ILSVRC2012
PATH_ILSVRC = base_path + 'DataSets/ILSVRC2012/'

ilsvrc_num_classes = 1000
ilsvrc_num_tgt_classes = 85 # 85 124 169 207
# ILSVRC_TGT_CLS = [49, 91, 122, 131, 149, 333]
ILSVRC_TGT_CLS = [207, 176, 169, 140, 130, 124, 116, 108, 85]

# PASCAL-S
PATH_PASCAL = base_path + 'DataSets/PASCAL-S/'

# tiny-imagenet
PATH_TINY_I = base_path + 'DataSets/tiny-imagenet'


# image mixed with mask
# rf_maps = alpha * aux_map + (1-alpha)*prior_map
ALPHA = 0.95 #0.50

n_gaussian = 16# default 8
n_gaussian_A = 16# default 8

rn_emb_dim=64 # default 64


# graph network
embed_dim = 128

augment = False
augment_ratio = 0.2
augment_range = int(6./augment_ratio)+1

scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

MAX_BNUM = 50 #100 # 50 for all the augs

FEATURE_DIM = 512 #512 #128 #256 #512 # 1024
BOI_SIZE = 7 # default 7
GRID_SIZE = 7 # default 7
RN_GROUP = 1 #2 #4 #16 #8

VIB_n_sample = 6
VIB_beta = 1e-3
VIB_dim = 256

ATT_RES = False

GBVS_R = 0.25

ALT_RATIO = 1.01