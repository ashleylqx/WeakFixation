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
PATH_LOG = base_path + 'WF/log/'
# PATH_LOG = base_path + 'WF/log2/'
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
ALPHA = 0.95

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

FEATURE_DIM = 512 #128 #256 #512 # 1024
