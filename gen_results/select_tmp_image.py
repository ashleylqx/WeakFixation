import os
import shutil
from scipy.misc import imread
'''
#base_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image/'
img_path = 'H:/Codes/WF/Preds/Example0211-12/examples_img/pascal-s/'
base_path = 'H:/Codes/WF/Preds/Example0211-12/examples_vis/pascal-s/'
gt_path = 'R:/dept2/qxlai/DataSets/MIT1003/ALLFIXATIONMAPS/'
our_path = 'H:/Codes/WF/Preds/Example0211-12/Examples0212/pascal-s/'
nips_path = 'R:/dept2/qxlai/WF/Preds/MIT1003/nips08/'
#mask_path = 'H:/Codes/WF/Preds/SALICON_train/object_mask/'

files = os.listdir(img_path)

for i in range(len(files)):
    img_name = files[i][:-4] #-5 '.jpeg' for mit1003, -4 '.jpg' for pascal-s
    #shutil.copyfile(gt_path+img_name+'_fixMap.jpg',
    #                base_path+img_name+'_fixMap.jpg') #mit1003
    shutil.copyfile(our_path+img_name+'.png',
                    base_path+img_name+'.png')
    #shutil.copyfile(nips_path+img_name+'.png',
    #                base_path+img_name+'_nips08.png')
    #shutil.copyfile(mask_path+img_name+'.png',
    #                base_path+img_name+'_om.png')
    print("\'%s\',"%img_name)
'''

DataSets = ['mit1003','pascal-s']
ds = DataSets[1]
img_path = 'H:/Codes/WF/Examples/examples_0219/%s/'%ds
base_path = img_path
our_path = 'H:/Codes/WF/Preds/Example0211-12/Examples0212/%s/'%ds
if ds=='mit1003':
    nips_path = 'R:/dept2/qxlai/WF/Preds/MIT1003/nips08/'
    postfix='.jpeg'
elif ds=='pascal-s':
    nips_path = 'R:/dept2/qxlai/WF/Preds/PASCAL-S/nips08/'
    postfix='.jpg'
     
files = os.listdir(img_path)
for i in range(len(files)):
    if 'fixMap' in files[i] or '.png' in files[i]:
        continue
    img_name = files[i][:-len(postfix)]
    # img = imread(img_path+img_name)
    # print(img.shape)

    # if img.shape[0]>img.shape[1]:
    #     os.remove(img_path+img_name)

    shutil.copyfile(our_path+img_name+'.png',
                    base_path+img_name+'_ours.png')
    shutil.copyfile(nips_path+img_name+'.png',
                    base_path+img_name+'_nips08.png')
    
        
                      
                      
        
