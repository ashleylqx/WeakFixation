import os
import shutil
from scipy.misc import imread, imsave
import cv2
import numpy as np

results_path = 'R:/dept2/qxlai/WF/Preds/MIT1003/'
postfix = '.jpeg'
# postfix = '.jpg' # SALICON

base_path = 'H:/Codes/WF/Preds/MIT1003/rebuttal_image/'
# base_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image/'
#base_path = 'H:/Codes/WF/Preds/SALICON_train/framework_examples_0221/'
#base_path = 'H:/Codes/WF/Examples/framework_examples_0221/'
gt_path = 'R:/dept2/qxlai/DataSets/MIT1003/ALLFIXATIONMAPS/'
our_path = os.path.join(results_path, 'ours')
mask_path = os.path.join(results_path, 'ours_om')
gaus_path = os.path.join(results_path, 'ours_gs')
sema_path = os.path.join(results_path, 'ours_cw')

files = os.listdir(base_path)

sigma=25
win_size=3*sigma*2+1

step = 5

for i in range(len(files)):
    # if files[i][-4:]=='.jpg':
    if files[i][-len(postfix):]==postfix:
        # img_name = files[i][:-6]
        # num = int(files[i][-5:-4]) # SALICON

        img_name = files[i][:-len(postfix)]
        num = None

        # step 1: remove either _1.jpg or _2.jpg SALICON
        if step ==1:
            if num==1:
                if not os.path.exists(base_path+img_name+'_2%s'%postfix):
                    os.remove(base_path+files[i])
            elif num==2:
                if not os.path.exists(base_path+img_name+'_1%s'%postfix):
                    os.remove(base_path+files[i])

        # step 2: remove duplicate SALICON
        elif step ==2:
            if num==1:
                os.remove(base_path+img_name+'_2%s'%postfix)


        # step 3: delete bad images by hand first
        elif step ==3:
            # img = imread(base_path + img_name+'_1%s'%postfix)
            # if img.shape[0]>img.shape[1]:
            #     os.remove(base_path+img_name+'_1%s'%postfix) # SALICON
            img = imread(base_path + img_name+postfix)
            if img.shape[0]>img.shape[1]:
                os.remove(base_path+img_name+postfix)

        # step 4: copy gt and ours, to the folder MIT1003
        elif step ==4:
            # shutil.copyfile(gt_path+img_name+'_fixMap.jpg',
            #                base_path+img_name+'_fixMap.jpg') # mit1003
            gt = cv2.imread(gt_path + img_name + '_fixMap.jpg')
            gt_blur = cv2.GaussianBlur(gt, (win_size, win_size), sigma)
            gt_blur = gt_blur.astype('float') / np.max(gt_blur) * 255.
            cv2.imwrite(base_path + img_name + '_fixMap_blur.jpg', gt_blur.astype(np.uint8))
            shutil.copyfile(os.path.join(our_path, img_name + '.png'),
                            base_path + img_name + '.png')
        # step 5: copy object masks to the folder
        elif step ==5:
            shutil.copyfile(os.path.join(mask_path, img_name+'.png'),
                            base_path+img_name+'_om.png')
            shutil.copyfile(os.path.join(gaus_path, img_name+'.png'),
                            base_path+img_name+'_gs.png')
            shutil.copyfile(os.path.join(sema_path, img_name+'.png'),
                            base_path+img_name+'_cw.png')

        
             
        #print("\'%s\',\n"%img_name)
        
                    
                      
        
