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
ds = DataSets[0]
# img_path = 'H:/Codes/WF/Examples/examples_0219/%s/'%ds


if ds=='mit1003':
    img_path = 'H:/Codes/WF/Preds/MIT1003/tmp_image/'
    nips_path = 'R:/dept2/qxlai/WF/Preds/MIT1003/nips08/'
    gt_path = 'R:/dept2/qxlai/DataSets/MIT1003/ALLFIXATIONMAPS/'
    postfix='.jpeg'
    methods = ['BMS', 'gbvs', 'SALICON', 'deepnet', 'DVA', 'sam-resnet_mit1003']
elif ds=='pascal-s':
    img_path = 'H:/Codes/WF/Preds/PASCAL-S/tmp_image/'
    nips_path = 'R:/dept2/qxlai/WF/Preds/PASCAL-S/nips08/'
    gt_path = 'R:/dept2/qxlai/DataSets/PASCAL-S/maps/'
    postfix='.jpg'
    methods = ['BMS', 'gbvs', 'SALICON', 'deepnet', 'DVA', 'sam-resnet_pascal']

base_path = img_path
our_path = 'H:/Codes/WF/Preds/Example0211-12/Examples0211/%s/'%ds


# methods = ['BMS', 'gbvs', 'SALICON', 'deepnet', 'DVA', 'sam-resnet_mit1003']
     
files = os.listdir(img_path)
for i in range(len(files)):
    #if 'fixMap' in files[i] or '.png' in files[i]:
    #    continue
    if postfix not in files[i]:
        continue
   
    img_name = files[i][:-len(postfix)]
    
    '''
    img = imread(img_path+img_name+postfix)
    print(img.shape)

    if img.shape[0]>img.shape[1]:
        os.remove(img_path+img_name+postfix)
    '''
    
    shutil.copyfile(our_path+img_name+'.png',
                    base_path+img_name+'_ours.png')
    #shutil.copyfile(nips_path+img_name+'.png',
    #                base_path+img_name+'_nips08.png')
    shutil.copyfile(gt_path+img_name+'_fixMap.jpg',
                    base_path+img_name+'_fixMap.jpg') #mit1003
    #shutil.copyfile(gt_path+img_name+'.png',
    #                base_path+img_name+'.png') #pascal-s
    '''
    for md in methods:
        des_path = 'R:/dept2/qxlai/WF/Preds/MIT1003/%s/'%md
        #des_path = 'R:/dept2/qxlai/WF/Preds/PASCAL-S/%s/'%md
        
        shutil.copyfile(des_path+img_name+'.png',
                    base_path+img_name+'_%s.png'%md[:10])
    '''                  
                      
        
