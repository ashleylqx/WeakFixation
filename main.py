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

from models import WeakFixation_base, WeakFixation_base_comp, WeakFixation

from custom_loss import HLoss
from config import *
from utils import *

def train_Wildcat_WK_hd_compf_map_cw(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        optimizer.zero_grad()

        inputs, gt_labels, boxes, boxes_nums, rf_maps = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        cps_logits, pred_maps = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        cps_losses = logits_loss(cps_logits, gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))

        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)

        cps_losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        bar.set_description("Train [{}] | cps_loss:{:.4f}({:.4f}) | "
                  "h_loss:{:.4f}({:.4f}) | rf_loss:{:.4f}({:.4f})".format(
                epoch,
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),np.mean(np.array(total_map_loss))


def train_Wildcat_WK_hd_compf_map_cw_alt_alpha(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, path_models):
    model.train()

    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    if os.path.exists(os.path.join(path_models, args.bestname)):
        checkpoint = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
        saved_state_dict = checkpoint['state_dict'] # cpu version
        new_params = model.state_dict().copy()

        if list(new_params.keys())[0][:7] == 'module.': # saved_state_dict have 'module.' for parallel training
            for k, y in saved_state_dict.items():
                if k in new_params.keys():
                    new_params[k] = y
        else:   # if current trianing is on single gpu
            for k, y in saved_state_dict.items():
                if k[:7] in new_params.keys():
                    new_params[k[:7]] = y

        model_aux.load_state_dict(new_params)

    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        optimizer.zero_grad()

        inputs, gt_labels, boxes, boxes_nums, prior_maps = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            prior_maps = prior_maps.cuda()

        cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        _, aux_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        aux_maps = aux_maps - torch.min(torch.min(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        prior_maps = prior_maps - torch.min(torch.min(prior_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values

        rf_maps = ALPHA * aux_maps + (1 - ALPHA) * (prior_maps.unsqueeze(1))
        cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))

        rf_losses = rf_weight * torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0),
                                                  torch.clamp(rf_maps.detach().squeeze(), min=0.0, max=1.0))

        cps_losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        bar.set_description("Train [{}] | cps_loss:{:.4f}({:.4f})"
                  " | h_loss:{:.4f}({:.4f}) | rf_loss:{:.4f}({:.4f})".format(
                epoch,
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))


def eval_Wildcat_WK_hd_compf_salicon_cw(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()

    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        cps_losses = logits_loss(cps_logits, gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)

        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        bar.set_description("Eval [{}] | "
                  "cps_loss:{:.4f}({:.4f}) | h_loss:{:.4f}({:.4f}) | map_loss:{:.4f}({:.4f})".format(
                epoch,
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)) , np.mean(np.array(total_map_loss))


def train_Wildcat_WK_hd_compf_map_cw_sa(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        optimizer.zero_grad()

        inputs, gt_labels, boxes, boxes_nums, rf_maps = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        cps_logits, pred_maps, att_scores = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values

        cps_losses = logits_loss(cps_logits, gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))

        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)

        cps_losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        bar.set_description("Train [{}] | cps:{:.4f}({:.4f})"
                  " | h:{:.4f}({:.4f}) | rf:{:.4f}({:.4f})"
                  " | att_scores:{:.4f}/{:.4f}/{:.4f}".format(
                epoch,
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss)),
                att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))


def eval_Wildcat_WK_hd_compf_salicon_cw_sa(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()

    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    bar = tqdm(dataloader)
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        cps_logits, pred_maps, att_scores = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        cps_losses = logits_loss(cps_logits, gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)

        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        bar.set_description("Eval [{}]"
                  " | cps:{:.4f}({:.4f}) | h:{:.4f}({:.4f}) | map:{:.4f}({:.4f})"
                  " | att_scores:{:.4f}/{:.4f}/{:.4f}".format(
                epoch,
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss)),
                att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))


def train_Wildcat_WK_hd_compf_map_cw_sa_sp(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        optimizer.zero_grad()

        inputs, gt_labels, boxes, boxes_nums, rf_maps = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        cps_logits, pred_maps, _, att_scores = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values

        cps_losses = logits_loss(cps_logits, gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))

        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)

        cps_losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        bar.set_description("Train [{}] | cps_loss:{:.4f}({:.4f})"
              " | h_loss:{:.4f}({:.4f}) | rf_loss:{:.4f}({:.4f})"
              " | att_scores:{:.4f}/{:.4f}/{:.4f}".format(
            epoch,
            cps_losses.item(), np.mean(np.array(total_cps_loss)),
            h_losses.item(), np.mean(np.array(total_h_loss)),
            rf_losses.item(), np.mean(np.array(total_map_loss)),
            att_scores.max().item(), att_scores.min().item(), torch.argmax(att_scores, dim=0).item()
            ))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))


def eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    total_obj_map_loss = list()

    # for i, X in enumerate(dataloader):
    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.to(inputs.device)

            boxes = boxes.to(inputs.device)
            sal_maps = sal_maps.to(inputs.device)
        inputs, gt_labels, boxes, sal_maps = \
            torch.autograd.Variable(inputs), torch.autograd.Variable(gt_labels), \
            torch.autograd.Variable(boxes), torch.autograd.Variable(sal_maps)

        cps_logits, pred_maps, obj_att_maps, att_scores = model(img=inputs,
                                                   boxes=boxes, boxes_nums=boxes_nums)

        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = loss_HM(pred_logits, gt_labels)  # use bce loss with sigmoid
        # losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # losses = torch.nn.BCEWithLogitsLoss()(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        # cps_losses = cps_weight * loss_HM(cps_logits, gt_labels)
        cps_losses = logits_loss(cps_logits, gt_labels)
        # cps_losses = cps_weight*torch.nn.BCEWithLogitsLoss()(cps_logits, gt_labels)
        # cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((sal_maps.size(-2), sal_maps.size(-1)))(pred_maps).squeeze(),
        #                                             min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        obj_map_losses = torch.nn.BCELoss()(torch.clamp(obj_att_maps.squeeze(), min=0.0, max=1.0), sal_maps)

        # total_loss.append(losses.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())
        total_obj_map_loss.append(obj_map_losses.item())

        # if i%train_log_interval == 0:
        #     # print("Eval [{}][{}/{}]"
        #     #       "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})"
        #     #       "\tmap_loss:{:.4f}({:.4f})\tobj_map_loss:{:.4f}({:.4f})"
        #     #       "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
        #     #     epoch, i, int(N),
        #     #     cps_losses.item(), np.mean(np.array(total_cps_loss)),
        #     #     h_losses.item(), np.mean(np.array(total_h_loss)),
        #     #     map_losses.item(), np.mean(np.array(total_map_loss)),
        #     #     obj_map_losses.item(), np.mean(np.array(total_obj_map_loss)),
        #     #     att_scores.max().item(), att_scores.min().item(), torch.argmax(att_scores, dim=0).item()
        #     #     ))
        bar.set_description("Eval [{}]"
              " | cps:{:.4f}({:.4f}) | h:{:.4f}({:.4f})"
              " | map:{:.4f}({:.4f}) | obj_map:{:.4f}({:.4f})"
              " | att_scores:{:.4f}/{:.4f}/{:.4f}".format(
            epoch, #i, int(N),
            cps_losses.item(), np.mean(np.array(total_cps_loss)),
            h_losses.item(), np.mean(np.array(total_h_loss)),
            map_losses.item(), np.mean(np.array(total_map_loss)),
            obj_map_losses.item(), np.mean(np.array(total_obj_map_loss)),
            att_scores.max().item(), att_scores.min().item(), torch.argmax(att_scores, dim=0).item()
            ))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))

# https://github.com/rdroste/unisal/blob/192abd2affbb1824895118a914806663bb897aa1/unisal/train.py#L710
# https://github.com/rdroste/unisal/blob/192abd2affbb1824895118a914806663bb897aa1/unisal/train.py#L607
# SIM, AUC-J, s-AUC, https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py
# KLD, NSS, CC, https://github.com/rdroste/unisal/blob/master/unisal/utils.py
def save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, dataloader, args, tgt_sizes,
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

        results = save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, model_name, test_dataloader, args,
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
