base_path = '/home/qx/'

input_h = 224
input_w = 224

det_input_h = 800
det_input_w = 800

output_h = 56
output_w = 56

# PASCAL-S
PATH_PASCAL = base_path + 'DataSets/PASCAL-S/'

# MIT300
PATH_MIT300 = base_path + 'DataSets/MIT300/'

# SALICON
PATH_SALICON = base_path + 'DataSets/SALICON/'
bgr_mean = [103.939, 116.779, 123.68]
rgb_mean = [123.68, 116.779, 103.939]

# MIT1003
PATH_MIT1003 = base_path + 'DataSets/MIT1003/'

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

# image mixed with mask
# rf_maps = alpha * aux_map + (1-alpha)*prior_map
ALPHA = 0.95 #0.50

# number of gaussian maps
n_gaussian = 16
n_gaussian_A = 16

scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

MAX_BNUM = 50 #100 # 50 for all the augs

# weight of loss term
cps_weight = 1.0
hth_weight = 0.1
hdsup_weight = 0.1
rf_weight = 0.1
