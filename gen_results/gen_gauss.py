
import os
import numpy as np
from scipy.misc import imsave, imresize

base_path = 'H:/Codes/WF/Preds/SALICON_train/gauss_prior'
save_path = 'H:/Codes/WF/Preds/SALICON_train/gauss_map'
#img_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image/'
img_path = 'H:/Codes/WF/Preds/SALICON_train/examples_0223/'
files = os.listdir(img_path)
namelist = [f[:-6] for f in files if f[-4:]=='.jpg'] 

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
    gauss = np.load(os.path.join(base_path, namelist[i]+'.npy'))
    save_fold = os.path.join(save_path, namelist[i])
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
        
    for g in range(gauss.shape[0]):
        imsave(os.path.join(save_fold, '%s_%02d.png'%(namelist[i], g)), 
               imresize(gauss[g,:,:], (56,56)))
