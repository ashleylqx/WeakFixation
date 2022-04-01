import os
import sys
import argparse
import time
import datetime
import random

import numpy as np
import math
import scipy.misc
import pdb

import torch
# torch.cuda.set_device(0)
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchnet as tnt
import sal_metrics
from logger import Logger

from load_data import SALICON_full, MIT1003_full, MS_COCO_map_full_aug
from load_data import collate_fn_salicon_rn, collate_fn_mit1003_rn, collate_fn_coco_map_rn

from models import WeakFixation

from custom_loss import HLoss
from config import *
from utils import *


# https://github.com/rdroste/unisal/blob/192abd2affbb1824895118a914806663bb897aa1/unisal/train.py#L710
# https://github.com/rdroste/unisal/blob/192abd2affbb1824895118a914806663bb897aa1/unisal/train.py#L607
# SIM, AUC-J, s-AUC, https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py
# KLD, NSS, CC, https://github.com/rdroste/unisal/blob/master/unisal/utils.py
def save_predictions(model, folder_name, best_model_file, dataloader, args, tgt_sizes,
                                                 metrics=('kld', 'nss', 'cc', 'sim', 'aucj')):
    model.eval()

    out_folder = os.path.join(folder_name, best_model_file)
    if len(tgt_sizes)>1:
        out_folder = out_folder+'_multiscale'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        # MIT1003 & PASCAL-S image, boxes, sal_map, fix_map(, image_name)
        ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X

        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()
            boxes_nums = boxes_nums.cuda()

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)

        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size * tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size * tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size * tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size * tgt_s

            output = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            pred_maps = output[1]

            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        pred_final = (pred_maps_all / len(tgt_sizes))
        pred_final_np = pred_final.detach().cpu().numpy()

        for b_i in range(ori_inputs.size(0)):
            ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[b_i] + '.jpeg'))  # height, width, channel

            scipy.misc.imsave(os.path.join(out_folder, img_name[b_i]+'.png'),
                              postprocess_prediction(pred_final_np[b_i][0], size=[ori_img.shape[0], ori_img.shape[1]]))

    if len(metrics) > 0:
        results = evaluate(args, out_folder, metrics)
        return results

def checkBounds(dim, data):
    pts = np.round(data)
    valid = np.sum((pts < np.tile(dim, [pts.shape[0], 1])), 1) # pts < image dimensions
    valid = valid + np.sum((pts >= 0), 1)  #pts > 0
    data = data[valid == 4, :]
    return data

def makeFixationMap(dim,pts):
    pts = np.round(pts)
    map = np.zeros(dim)
    pts = checkBounds(dim, pts)
    pts = pts.astype('int')
    map[(pts[:,0], pts[:,1])] += 1

    return map

def evaluate(args, out_folder, metrics):
    assert len(metrics) > 0

    results = {metric: tnt.meter.AverageValueMeter() for metric in metrics}
    for metric in metrics:
        results[metric].reset()
    path_saliency = os.path.join(PATH_MIT1003, 'ALLFIXATIONMAPS')
    path_fixation = os.path.join(PATH_MIT1003, 'ALLFIXATIONS')
    pred_files = os.listdir(out_folder)

    for f_i in range(len(pred_files)):
        file_name = pred_files[f_i]
        sal_path = os.path.join(path_saliency, file_name[:-4] + '_fixMap.jpg')
        fix_path = os.path.join(path_fixation, file_name[:-4] + '_fixPts.jpg')

        sal_map_np = cv2.imread(sal_path, 0)
        fix_map_np = cv2.imread(fix_path, 0)
        fix_map_np = fix_map_np > 0
        sal_map_np = sal_map_np[np.newaxis, :].astype('float')
        fix_map_np = fix_map_np[np.newaxis, :].astype('uint8')
        sal_map = torch.tensor(sal_map_np, dtype=torch.float)
        fix_map = torch.tensor(fix_map_np, dtype=torch.uint8)

        pred_final_np = cv2.imread(os.path.join(out_folder, file_name), 0)
        pred_final_np = pred_final_np[np.newaxis, :].astype('float')
        pred_final = torch.tensor(pred_final_np, dtype=torch.float)

        for this_metric in metrics:
            if this_metric == 'sim':
                # sim_val = sal_metrics.similarity(pred_final, sal_map)
                sim_val = sal_metrics.similarity(pred_final_np, sal_map_np) # ok!
                # results[this_metric].add(sim_val.mean(), sim_val.shape[0])
                results[this_metric].add(sim_val.mean())
            elif this_metric == 'aucj':
                aucj_val = sal_metrics.auc_judd(pred_final_np, fix_map_np) # ok!
                # results[this_metric].add(aucj_val.mean(), aucj_val.shape[0])
                results[this_metric].add(aucj_val.mean())
            elif this_metric == 'kld':
                kld_val = sal_metrics.kld_loss(pred_final, sal_map)
                results[this_metric].add(kld_val.mean().item(), kld_val.size(0))
            elif this_metric == 'nss':
                nss_val = sal_metrics.nss(pred_final, fix_map) # do not need .exp() for our case; ok!
                results[this_metric].add(nss_val.mean().item(), nss_val.size(0))
            elif this_metric == 'cc':
                cc_val = sal_metrics.corr_coeff(pred_final, sal_map) # do not need .exp() for our case; ok!
                results[this_metric].add(cc_val.mean().item(), cc_val.size(0))

    print_content = ''
    for metric in metrics:
        print_content += '%s:%.4f\t' % (metric, results[metric].mean)

    print(print_content)
    return results


def main(args):
    path_models = os.path.join(args.path_out, 'Models', args.model_name)
    if not os.path.exists(path_models):
        os.makedirs(path_models)

    phase = args.phase

    kmax = 1
    kmin = None
    alpha = 0.7
    num_maps = 4
    fix_feature = False


    # train soft attention model
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor  # computation in GPU
        args.use_gpu = True
        print('Using GPU.')
    else:
        dtype = torch.FloatTensor
        print('Using CPU.')

    # generate saliency maps
    if phase == 'test':
        model = WeakFixation(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                             fix_feature=fix_feature, use_grid=True)

        if args.use_gpu:
            model.cuda()

        model_name = args.model_name
        print(model_name)

        best_model = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
        saved_state_dict = best_model['state_dict']
        new_params = model.state_dict().copy()
        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                if k[7:] in new_params.keys():
                    new_params[k[7:]] = y
        else:
            for k, y in saved_state_dict.items():
                if k in new_params.keys():
                    new_params[k] = y
        model.load_state_dict(new_params)

        folder_name = os.path.join(args.path_out, 'Preds/MIT1003')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        eval_metrics = ('nss',)

        results = save_predictions(model, folder_name, model_name, test_dataloader, args,
                                                               tgt_sizes=tgt_sizes, metrics=eval_metrics)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_out", default='WF/',
                        type=str,
                        help="""set output path for the trained model""")
    parser.add_argument("--batch-size", default=4, type=int, help="""Set batch size""")
    parser.add_argument("--train-batch", default=36, type=int, help="""Set batch size""")
    parser.add_argument("--test-batch", default=20, type=int, help="""Set batch size""")
    parser.add_argument("--n_epochs", default=200, type=int,
                        help="""Set total number of epochs""")
    parser.add_argument("--lr", type=float, default=1e-4, help="""Learning rate for training""")
    parser.add_argument('--schedule', type=int, nargs='+', default=[60,120,180],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  # converge faster
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument("--patience", type=int, default=200,
                        help="""Patience for learning rate scheduler (default 3)""")
    parser.add_argument("--use_gpu", type=bool, default=False,
                        help="""Whether use GPU (default False)""")
    parser.add_argument('--resume', action='store_true',
                        help='whether to resume from folder')
    parser.add_argument('--phase', default='test', type=str, help='running phase')
    parser.add_argument('--ckptname', default='checkpoint.pt', type=str, help='filename of model')
    parser.add_argument('--bestname', default='model_best.pt', type=str, help='filename of best model')
    parser.add_argument('--model_name', default='210426_sgd', type=str, help='folder of model')
    parser.add_argument('--init_model', default='210426_sgd', type=str, help='folder of init model')

    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, schedule):
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma


if __name__ == '__main__':
    args = parse_arguments()

    main(args)
