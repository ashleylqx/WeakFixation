import os
import shutil
import numpy as np
from scipy.misc import imread, imsave
import scipy.io
import cv2

def gen_grid_boxes(in_h, in_w, N=14):
    x = np.linspace(0., in_w, N+1)
    y = np.linspace(0., in_h, N+1)
    xv, yv = np.meshgrid(x, y)

    grid_boxes = list()
    for i in range(N):
        for j in range(N):
            grid_boxes.append(np.array([[xv[i, j], yv[i, j], xv[i+1, j+1], yv[i+1, j+1]]]))

    return np.concatenate(grid_boxes, axis=0) # leave dim 0 for batch size repeat


base_path = 'R:/dept2/qxlai/DataSets/MS_COCO/train2014_eb500/'
our_path = 'H:/Codes/WF/Preds/SALICON_train/pred_map/'
attscore_path = 'H:/Codes/WF/Preds/SALICON_train/attention_score/'
save_path = 'H:/Codes/WF/Preds/SALICON_train/proposal_map/'


# namelist = ['COCO_train2014_000000048595']
#namelist = ['COCO_train2014_000000365817']
namelist = ['COCO_train2014_000000467063']

for i in range(len(namelist)):
    save_fold = save_path + namelist[i]
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
    
    image = imread(our_path+namelist[i]+'.png')
    att_scores = np.load(attscore_path+namelist[i]+'.npy')
    
    box_path = os.path.join(base_path, namelist[i]+'_bboxes.mat')
    boxes = scipy.io.loadmat(box_path)['bboxes'][:50, :]
    #boxes[:, 0] = boxes[:, 0] * image.shape[1]
    #boxes[:, 2] = boxes[:, 2] * image.shape[1]
    #boxes[:, 1] = boxes[:, 1] * image.shape[0]
    #boxes[:, 3] = boxes[:, 3] * image.shape[0]
    boxes = boxes * 224

    grids = gen_grid_boxes(224,224,7)
   
    '''
    for box_i in range(boxes.shape[0]):
        box = boxes[box_i].astype('int')
        # att_map = np.zeros((image.shape[0],image.shape[1]))
        att_map = np.zeros((224,224))
        #att_map[box[1]:box[3], box[0]:box[2]] = int(att_scores[box_i]*255)
        att_map[box[1]:box[3], box[0]:box[2]] = max(int(att_scores[box_i+49]/att_scores.max()*255),1)
        att_map = cv2.resize(att_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        imsave(os.path.join(save_fold, '%s_%02d.png'%(namelist[i], box_i+49)), 
               att_map)
    '''
    '''
    print(att_scores[11+49],att_scores[31+49],att_scores[40+49], att_scores[41+49])
    att_scores[11+49] = att_scores[31+49]/10.
    att_scores[31+49] = att_scores[31+49]/10.
    att_scores[40+49] = att_scores[40+49]/10.
    att_scores[41+49] = att_scores[41+49]/10.
    print(att_scores[11+49],att_scores[31+49],att_scores[40+49], att_scores[41+49])
    '''
    #att_scores[17] = att_scores[17]*1e4
    #att_scores[18] = att_scores[18]*1e3
    #att_scores[24] = att_scores[24]*1e5
    #att_scores[25] = att_scores[25]*1e4
    '''
    print(att_scores[17],att_scores[18],att_scores[24], att_scores[25],att_scores[31], att_scores[32])
    index = [17,18,24,25,31,32]
    #ratio = [1.,1.,8.5e4,6e3,4e4,3e5]
    ratio = [1.,1.,5e4,6e3,2e4,2.5e5]
    for idx in range(len(index)):
        att_scores[index[idx]] = att_scores[index[idx]]*ratio[idx]
    print(att_scores[17],att_scores[18],att_scores[24], att_scores[25],att_scores[31], att_scores[32])
    '''
    
    index = [68,65,49]
    #ratio = [1.,1.,8.5e4,6e3,4e4,3e5]
    ratio = [1e-1]*len(index)
    print([att_scores[a] for a in index])
    for idx in range(len(index)):
        att_scores[index[idx]] = att_scores[index[idx]]*ratio[idx]  
    print([att_scores[a] for a in index])
    
    #att_map = np.zeros((boxes.shape[0], image.shape[0], image.shape[1]))
    att_map = np.zeros((grids.shape[0]+boxes.shape[0], 224, 224))
    for box_i in range(grids.shape[0]):
        box = grids[box_i].astype('int')  
        #att_map[box[1]:box[3], box[0]:box[2]] = int(att_scores[box_i]*255)
        #att_map[box_i, box[1]:box[3], box[0]:box[2]] = max(int(att_scores[box_i]/att_scores.max()*255),1)#att_scores[box_i]
        att_map[box_i, box[1]:box[3], box[0]:box[2]] = att_scores[box_i]
    for box_i in range(boxes.shape[0]):
        box = boxes[box_i].astype('int')  
        #att_map[box[1]:box[3], box[0]:box[2]] = int(att_scores[box_i]*255)
        #att_map[box_i, box[1]:box[3], box[0]:box[2]] = max(int(att_scores[box_i]/att_scores.max()*255),1)#att_scores[box_i]
        att_map[box_i, box[1]:box[3], box[0]:box[2]] = att_scores[box_i+49]
    att_map_sum = att_map.sum(axis=0)
    att_map_sum = att_map_sum-att_map_sum.min()
    att_map_sum = (att_map_sum/np.max(att_map_sum) * 255).astype(np.uint8)
    sigma=4
    att_map_sum = cv2.GaussianBlur(att_map_sum, (sigma*6+1, sigma*6+1), sigma)
    att_map_sum = cv2.resize(att_map_sum, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    att_map_post = att_map_sum.astype('float') / np.max(att_map_sum) * 255.                
    imsave(os.path.join(save_fold, '%s_om_2.png'%(namelist[i])), att_map_post)
    #imsave(save_fold+'%s_om_2.png'%(namelist[i]), att_map_post)
    
