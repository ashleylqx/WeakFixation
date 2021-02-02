import os
from scipy.misc import imread, imsave
'''
base_path = 'H:/Codes/WF/Preds/MIT300'

folders = os.listdir(base_path)

for f in range(len(folders)):
    fold = folders[f]

    img_path = os.path.join(base_path, fold)
    imgs = os.listdir(img_path)

    for i in range(len(imgs)):
        img_name = imgs[i]
        img = imread(os.path.join(img_path, img_name))

        new_name = img_name[:-4]+'.jpg'

        imsave(os.path.join(img_path, new_name), img)
'''

# base_path = 'R:/dept2/qxlai/DataSets/toronto/maps'
base_path = 'R:/dept2/qxlai/WF/Preds/TORONTO/sam-resnet'

imgs = os.listdir(base_path)

for i in range(len(imgs)):
    img_name = imgs[i]
    new_name = img_name[:-4]+'.png'
    
    if img_name[-4:]!='.jpg' or os.path.exists(os.path.join(base_path, new_name)):
        continue
    
    img = imread(os.path.join(base_path, img_name))  
    imsave(os.path.join(base_path, new_name), img)
