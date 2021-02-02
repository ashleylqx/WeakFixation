import os
import shutil
from scipy.misc import imread, imsave
import cv2
import numpy as np

base_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image/'
#base_path = 'H:/Codes/WF/Preds/SALICON_train/framework_examples_0221/'
#base_path = 'H:/Codes/WF/Examples/framework_examples_0221/'
gt_path = 'R:/dept2/qxlai/DataSets/SALICON/maps/train/'
our_path = 'H:/Codes/WF/Preds/SALICON_train/pred_map/'
nips_path = 'R:/dept2/qxlai/DataSets/MS_COCO/train2014_nips08/'
mask_path = 'H:/Codes/WF/Preds/SALICON_train/object_mask/'

files = os.listdir(base_path)

sigma=25
win_size=3*sigma*2+1

for i in range(len(files)):
    if files[i][-4:]=='.jpg':
        img_name = files[i][:-6]
        num = int(files[i][-5:-4])

        # step 1: remove either _1.jpg or _2.jpg
        ''' 
        if num==1:           
            if not os.path.exists(base_path+img_name+'_2.jpg'):
                os.remove(base_path+files[i])
        elif num==2:
            if not os.path.exists(base_path+img_name+'_1.jpg'):
                os.remove(base_path+files[i])
        '''
        # step 2: remove duplicate
        ''' 
        if num==1:
            os.remove(base_path+img_name+'_2.jpg')
           
        '''
        # step 3: delete bad images by hand first
        
        # step 4: copy gt, ours and nips08 to the folder
        #shutil.copyfile(gt_path+img_name+'.png',
        #                base_path+img_name+'_gt.png')
        '''
        gt = cv2.imread(gt_path+img_name+'.png')
        gt_blur = cv2.GaussianBlur(gt, (win_size, win_size), sigma)
        gt_blur = gt_blur.astype('float') / np.max(gt_blur) * 255.
        cv2.imwrite(base_path+img_name+'_gt_blur.png', gt_blur.astype(np.uint8))
        shutil.copyfile(our_path+img_name+'.png',
                        base_path+img_name+'.png')
        shutil.copyfile(nips_path+img_name+'.png',
                        base_path+img_name+'_nips08.png')
        '''
        # step 5: copy object masks to the folder
        shutil.copyfile(mask_path+img_name+'.png',
                        base_path+img_name+'_om.png')
        
             
        #print("\'%s\',\n"%img_name)
        
                    
                      
        
