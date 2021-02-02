
import os
import numpy as np
from scipy.misc import imsave, imresize

def sigmoid(x):
    return 1/(1+np.exp(-x))

coco_id_name_map={ 0: 'background', 
                   1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
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

base_path = 'H:/Codes/WF/Preds/SALICON_train/semantics_cw/'
#model = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_2_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00_cw'
#base_path = 'R:/dept2/qxlai/WF/Preds/SALICON_train/%s/'%model
save_path = 'H:/Codes/WF/Preds/SALICON_train/semantic_map'

label_path = 'H:/Codes/WF/Preds/SALICON_train/pred_and_label/'


#img_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image/'
img_path = 'H:/Codes/WF/Preds/SALICON_train/examples_0224/'
files = os.listdir(img_path)
namelist = [f[:-6] for f in files if f[-4:]=='.jpg']

namelist = ['COCO_train2014_000000048595']

'''
namelist = ['COCO_train2014_000000016957',
'COCO_train2014_000000188388',
'COCO_train2014_000000267105',
'COCO_train2014_000000366706',
'COCO_train2014_000000431494',
'COCO_train2014_000000465670',
'COCO_train2014_000000466491'
]
namelist = ['COCO_train2014_000000003713',
'COCO_train2014_000000059776',
'COCO_train2014_000000068623',
'COCO_train2014_000000091123',
'COCO_train2014_000000105532',
'COCO_train2014_000000125091',
'COCO_train2014_000000237385',
'COCO_train2014_000000313923',
'COCO_train2014_000000569520']
namelist = ['COCO_train2014_000000052030',
            'COCO_train2014_000000240112',
            'COCO_train2014_000000388779',
            'COCO_train2014_000000486789']
'''
for i in range(len(namelist)):
    img_name = namelist[i]
    semantics = np.load(base_path+img_name+'.npy')
    semantics = sigmoid(semantics)
    label = np.load(label_path+img_name+'_label.npy')
    pred_score = np.load(label_path+img_name+'.npy')
    #pred_score = sigmoid(pred_score)
    save_fold = os.path.join(save_path, img_name)
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)

    gt_index = np.where(label==1)[0]
    pred_index = np.argsort(-pred_score)
    
    for l in range(len(gt_index)):
        imsave(os.path.join(save_fold,
               '%s_gt_%02d_%s.png'%(namelist[i], gt_index[l], coco_id_name_map[gt_index[l]])), 
               imresize(semantics[gt_index[l],:,:], (56,56)))
        imsave(os.path.join(save_fold,
               '%s_pred_%02d_%s.png'%(namelist[i], pred_index[l], coco_id_name_map[pred_index[l]])), 
               imresize(semantics[pred_index[l],:,:], (56,56)))
    '''    
    imsave(os.path.join(save_fold,
               '%s_%02d_%s.png'%(namelist[i], 0, coco_id_name_map[0])), 
               imresize(semantics[0,:,:], (56,56)))
    '''
