import os
import numpy as np
import pickle
from collections import OrderedDict

from config import PATH_TINY_I, PATH_ILSVRC

with open(os.path.join(PATH_TINY_I, 'wnids.txt'),'r') as f:
    tmp=f.readlines()
wnids = [w[:-1] for w in tmp]


# mode = 'train'
#
# # lines = list()
# other_lines = list()
# if mode == 'train':
#     folders = os.listdir(os.path.join(PATH_TINY_I, mode))
#
#     for f in folders:
#         wnid = wnids.index(f)
#         imgs = os.listdir(os.path.join(PATH_TINY_I, mode, f, 'images'))
#         # f_lines = ['{}/images/{} {}\n'.format(f, img, wnid) for img in imgs]
#         # lines.extend(f_lines)
#
#         o_lines = ['{}/{} {}\n'.format(f, img[:-5], wnid) for img in imgs]
#         other_lines.extend(o_lines)
#
# elif mode=='val':
#     with open(os.path.join(PATH_TINY_I, mode, 'val_annotations.txt'), 'r') as f:
#         vals = f.readlines()
#
#     imgs = [l.split('\t')[0] for l in vals]
#     folders = [l.split('\t')[1] for l in vals]
#     # lines = ['images/{} {}\n'.format(imgs[i], wnids.index(folders[i])) for i in range(len(imgs))]
#
#     other_lines = ['{} {}\n'.format(imgs[i][:-5], wnids.index(folders[i])) for i in range(len(imgs))]
#
#
# # with open(os.path.join(PATH_TINY_I, '{}_list.txt'.format(mode)), 'w') as f:
# #     f.writelines(lines)
#
# with open(os.path.join(PATH_TINY_I, '{}_list_other.txt'.format(mode)), 'w') as f:
#     f.writelines(other_lines)

#==========================================
mode = 'val'

with open(os.path.join(PATH_ILSVRC, '{}_list.txt'.format(mode)),'r') as f:
    ils=f.readlines()

paths_ils = [e.split(' ')[0] for e in ils]

if mode=='train':
    names_ils = [e.split(' ')[0].split('/')[0] for e in ils]
    # names_ils2 = list(OrderedDict.fromkeys(names_ils))

elif mode=='val':
    wnid2f = tmp = pickle.load(open(os.path.join(PATH_ILSVRC, 'WordTree/ils_wnid_f.pickle'),'rb'))
    val_lb = [int(e.split(' ')[1]) for e in ils]
    names_ils = [wnid2f[lb] for lb in val_lb]

new_ils_lines = ['{} {}\n'.format(paths_ils[i], wnids.index(names_ils[i]))
                 for i in range(len(paths_ils)) if names_ils[i] in wnids]

with open(os.path.join(PATH_ILSVRC, '{}_list_200.txt'.format(mode)), 'w') as f:
    f.writelines(new_ils_lines)
