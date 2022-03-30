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

from load_data_new import SALICON_full, MIT1003_full, MS_COCO_map_full_aug
from load_data_new import collate_fn_salicon_rn, collate_fn_mit1003_rn, collate_fn_coco_map_rn

from models_new import Wildcat_WK_hd_gs_compf_cls_att_A4_cw, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp

from custom_loss_new import HLoss
from config_new import *
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

    out_folder = os.path.join(args.path_out, folder_name, best_model_file)
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
        results = evaluate(args, folder_name, out_folder, metrics)
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

def evaluate(args, folder_name, best_model_file, metrics):
    assert len(metrics) > 0

    results = {metric: tnt.meter.AverageValueMeter() for metric in metrics}
    for metric in metrics:
        results[metric].reset()
    path_saliency = os.path.join(PATH_MIT1003, 'ALLFIXATIONMAPS')
    path_fixation = os.path.join(PATH_MIT1003, 'ALLFIXATIONS')
    out_folder = os.path.join(args.path_out, folder_name, best_model_file)
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

    # *** 210427 ***
    if phase == 'train_cw_aug':
        print('lr %.4f'%args.lr)

        prior = 'nips08'

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, use_grid=True) #################

        '''optimizer'''
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) ############

        s_epoch = 0
        nss_value = 0
        model_name = args.model_name
        print(model_name)
        title = 'MIT1003-' + model_name
        if args.resume:
            # checkpoint = torch.load(os.path.join(args.path_out, 'Models', model_name + '.pt'))  # checkpoint is a dict, containing much info
            checkpoint = torch.load(os.path.join(path_models, args.ckptname), map_location='cuda:0')  # checkpoint is a dict, containing much info
            # model.load_state_dict(checkpoint['state_dict'])
            saved_state_dict = checkpoint['state_dict']
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

            # opt_state_dict = checkpoint['optimizer']
            # pdb.set_trace()
            # for key in opt_state_dict['state'].keys():
            #     opt_state_dict['state'][key] = opt_state_dict['state'][key].cpu()
            # optimizer.load_state_dict(opt_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            s_epoch = checkpoint['epoch'] + 1

            best_model = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
            nss_value = best_model['nss']

            logger = Logger(os.path.join(path_models, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(path_models, 'log.txt'), title=title)
            # logger.set_names(['Epoch', 'LR', 'T_cps', 'V_cps', 'T_h', 'V_h', 'T_map', 'V_map', 'Nss'])
            logger.set_names(['Epoch', 'LR', 'Train_cps', 'Val_cps', 'Train_h', 'Val_h', 'Train_map', 'Val_map', 'Nss'])

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()

        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) # ********

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w) #, N=24

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch*gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch*gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        folder_name = 'Preds/MIT1003'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size * gpu_number, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)] # batch_size=4
        tgt_sizes = [224]  # batch_size=16
        eval_metrics = ('nss',)

        logits_loss = torch.nn.BCEWithLogitsLoss()
        h_loss = HLoss()

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        cnt = 0
        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            adjust_learning_rate(optimizer, i_epoch, args.schedule) # for SGD
            is_best = False

            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)
            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)


            results = save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, model_name, test_dataloader, args, tgt_sizes=tgt_sizes, metrics=eval_metrics)

            tmp_nss_value = results['nss'].mean
            if tmp_nss_value > nss_value:
                eval_loss = val_cps
                # eval_loss = tmp_eval_loss
                nss_value = tmp_nss_value
                is_best = True
                cnt = 0
                print('Saving model with nss %.4f ...' % nss_value)

            save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=args.ckptname, results=results,
                       is_best=is_best, best_name=args.bestname)

            cnt += 1

            logger.append([i_epoch, optimizer.param_groups[0]['lr'], train_cps, val_cps, train_h, val_h,
                           train_map, val_map, tmp_nss_value])
            if cnt >= args.patience:
                break

        logger.close()
        print('Best model nss: %.4f' % (nss_value))

    # **** 210502 ***
    elif phase == 'train_cw_alt_alpha':
        print('lr %.4f' % args.lr)

        prior = 'nips08'
        #########################################
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, use_grid=True)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                         num_maps=num_maps,
                                                         fix_feature=fix_feature, use_grid=True)

        model_name = args.model_name
        print(model_name)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################

        print('relation lr factor: 1.0')

        s_epoch = 0
        model_name = args.model_name
        print(model_name)
        title = 'MIT1003-' + model_name
        if args.resume:
            checkpoint = torch.load(os.path.join(path_models, args.ckptname), map_location='cuda:0')
            saved_state_dict = checkpoint['state_dict']
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

            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            s_epoch = checkpoint['epoch'] + 1

            # load model_aux from previous best basemodel_alt
            if os.path.exists(os.path.join(path_models, args.bestname)):
                checkpoint_aux = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
            else:
                model_aux_path = os.path.join(args.path_out, 'Models', args.init_model, args.bestname)
                checkpoint_aux = torch.load(model_aux_path, map_location='cuda:0')

            nss_value = checkpoint_aux['nss']
            saved_state_dict = checkpoint_aux['state_dict']
            if list(saved_state_dict.keys())[0][:7] == 'module.':
                new_params = model_aux.state_dict().copy()
                for k, y in saved_state_dict.items():
                    new_params[k[7:]] = y
            else:
                new_params = saved_state_dict.copy()
            model_aux.load_state_dict(new_params)

            logger = Logger(os.path.join(path_models, 'log.txt'), title=title, resume=True)
        else:
            # load model_aux from basemodel_sgd
            model_aux_path = os.path.join(args.path_out, 'Models', args.init_model, args.bestname)
            checkpoint_aux = torch.load(model_aux_path)
            nss_value = checkpoint_aux['nss']

            saved_state_dict = checkpoint_aux['state_dict']
            if list(saved_state_dict.keys())[0][:7] == 'module.':
                new_params = model_aux.state_dict().copy()
                for k, y in saved_state_dict.items():
                    new_params[k[7:]] = y
            else:
                new_params = saved_state_dict.copy()
            model_aux.load_state_dict(new_params)

            logger = Logger(os.path.join(path_models, 'log.txt'), title=title)
            logger.set_names(['Epoch', 'LR', 'Train_cps', 'Val_cps', 'Train_h', 'Val_h', 'Train_map', 'Val_map', 'Nss'])

        for param in model_aux.parameters():
            param.requires_grad = False


        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            model_aux = torch.nn.DataParallel(model_aux).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()

        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)  # , N=48 # ***

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)  # , N=24

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch * gpu_number,
                                      collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch * gpu_number,
                                     collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        folder_name = 'Preds/MIT1003'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size * gpu_number, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)] # batch_size=4
        tgt_sizes = [224]  # batch_size=16
        eval_metrics = ('nss',)

        '''loss function'''
        logits_loss = torch.nn.BCEWithLogitsLoss()

        h_loss = HLoss()

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()

        eval_loss = np.inf

        cnt = 0
        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            # adjust_learning_rate(optimizer, i_epoch, args.schedule)
            is_best = False
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw_alt_alpha(i_epoch, model, model_aux,
                                                                                       optimizer, logits_loss, h_loss,
                                                                                       train_dataloader, args,
                                                                                       path_models)

            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss,
                                                                          eval_dataloader, args)

            results = save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, model_name, test_dataloader,
                                                                   args, tgt_sizes=tgt_sizes, metrics=eval_metrics)

            tmp_nss_value = results['nss'].mean
            if tmp_nss_value > nss_value:
                eval_loss = val_cps
                nss_value = tmp_nss_value
                is_best = True
                cnt = 0
                print('Saving model with nss %.4f ...' % nss_value)


            save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=args.ckptname, results=results,
                       is_best=is_best, best_name=args.bestname)

            cnt += 1

            logger.append([i_epoch, optimizer.param_groups[0]['lr'], train_cps, val_cps, train_h, val_h,
                           train_map, val_map, tmp_nss_value])

            if cnt >= args.patience:
                break

        logger.close()
        print('Best model nss: %.4f' % (nss_value))

    # *** 210511 ***
    elif phase == 'train_cw_aug_sa_art':
        print('lr %.4f'%args.lr)

        prior = 'nips08'

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, use_grid=True) #################


        '''init model'''
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', args.init_model, args.bestname))
        saved_state_dict = checkpoint['state_dict']
        new_params = model.state_dict().copy()

        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                new_params[k[7:]] = y

        else:
            for k, y in saved_state_dict.items():
                new_params[k] = y

        model.load_state_dict(new_params)

        s_epoch = 0
        nss_value = 1.5
        model_name = args.model_name
        print(model_name)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################

        print('real learning rate %f.' % args.lr)

        title = 'MIT1003-' + model_name
        if args.resume:
            checkpoint = torch.load(os.path.join(path_models, args.ckptname), map_location='cuda:0')
            saved_state_dict = checkpoint['state_dict']
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

            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            s_epoch = checkpoint['epoch'] + 1

            best_model = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
            nss_value = best_model['nss']

            logger = Logger(os.path.join(path_models, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(path_models, 'log.txt'), title=title)
            logger.set_names(['Epoch', 'LR', 'Train_cps', 'Val_cps', 'Train_h', 'Val_h', 'Train_map', 'Val_map', 'Nss'])

        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, 'relation_net'):
            model.relation_net.self_attention.weight.requires_grad = True
            model.relation_net.self_attention.bias.requires_grad = True
        else:
            model.self_attention.weight.requires_grad = True
            model.self_attention.bias.requires_grad = True

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()

        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) # *******

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch * gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch * gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        folder_name = 'Preds/MIT1003'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size * gpu_number, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)] # batch_size=4
        tgt_sizes = [224]  # batch_size=16
        eval_metrics = ('nss',)

        '''loss function'''
        logits_loss = torch.nn.BCEWithLogitsLoss()
        h_loss = HLoss()

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()

        eval_loss = np.inf
        cnt = 0
        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            # adjust_learning_rate(optimizer, i_epoch, args.schedule)
            is_best = False
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw_sa(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw_sa(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)

            results = save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, model_name, test_dataloader, args, tgt_sizes=tgt_sizes, metrics=eval_metrics)

            tmp_nss_value = results['nss'].mean
            if tmp_nss_value > nss_value:
                eval_loss = val_cps
                nss_value = tmp_nss_value
                is_best = True
                cnt = 0
                print('Saving model with nss %.4f ...' % nss_value)

            save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=args.ckptname, results=results,
                       is_best=is_best, best_name=args.bestname)

            cnt += 1

            logger.append([i_epoch, optimizer.param_groups[0]['lr'], train_cps, val_cps, train_h, val_h,
                           train_map, val_map, tmp_nss_value])

            if cnt >= args.patience:
                break

        logger.close()
        print('Best model nss: %.4f' % (nss_value))

    # *** 210512 ***
    elif phase == 'train_cw_aug_sa_sp_fixf':
        print('lr %.4f'%args.lr)

        prior = 'nips08'

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, use_grid=True)

        '''checkpoint (init)'''
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', args.init_model, args.bestname))
        saved_state_dict = checkpoint['state_dict']
        new_params = model.state_dict().copy()

        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                new_params[k[7:]] = y

        else:
            for k, y in saved_state_dict.items():
                new_params[k] = y

        model.load_state_dict(new_params)

        s_epoch = 0
        nss_value = 1.5
        model_name = args.model_name
        print(model_name)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ######################

        print('real learning rate %f.' % args.lr)

        title = 'MIT1003-' + model_name
        if args.resume:
            checkpoint = torch.load(os.path.join(path_models, args.ckptname), map_location='cuda:0')
            saved_state_dict = checkpoint['state_dict']
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

            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            s_epoch = checkpoint['epoch'] + 1

            best_model = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
            nss_value = best_model['nss']

            logger = Logger(os.path.join(path_models, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(path_models, 'log.txt'), title=title)
            logger.set_names(['Epoch', 'LR', 'Train_cps', 'Val_cps', 'Train_h', 'Val_h', 'Train_map', 'Val_map', 'Nss'])


        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, 'relation_net'):
            model.relation_net.self_attention.weight.requires_grad = True
            model.relation_net.self_attention.bias.requires_grad = True
        else:
            model.self_attention.weight.requires_grad = True
            model.self_attention.bias.requires_grad = True


        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()

        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) # ********

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch * gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch * gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        folder_name = 'Preds/MIT1003'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size * gpu_number, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)] # batch_size=4
        tgt_sizes = [224]  # batch_size=16
        eval_metrics = ('nss',)

        '''loss function'''
        logits_loss = torch.nn.BCEWithLogitsLoss()
        h_loss = HLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################

        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        cnt = 0
        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            # adjust_learning_rate(optimizer, i_epoch, args.schedule)
            is_best = False
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw_sa_sp(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)

            results = save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, model_name, test_dataloader,
                                                                   args, tgt_sizes=tgt_sizes, metrics=eval_metrics)

            tmp_nss_value = results['nss'].mean
            if tmp_nss_value > nss_value:
                eval_loss = val_cps
                # eval_loss = tmp_eval_loss
                nss_value = tmp_nss_value
                is_best = True
                cnt = 0
                print('Saving model with nss %.4f ...' % nss_value)

            save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=args.ckptname, results=results,
                       is_best=is_best, best_name=args.bestname)

            cnt += 1

            logger.append([i_epoch, optimizer.param_groups[0]['lr'], train_cps, val_cps, train_h, val_h,
                           train_map, val_map, tmp_nss_value])

            if cnt >= args.patience:
                break

        logger.close()
        print('Best model nss: %.4f' % (nss_value))


    # *** 210423 ***
    elif phase == 'train_cw_aug_sa_sp':
        print('lr %.4f'%args.lr)

        prior = 'nips08'

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, use_grid=True) #################

        '''init model'''
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', args.init_model, args.bestname))
        saved_state_dict = checkpoint['state_dict']
        new_params = model.state_dict().copy()

        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                new_params[k[7:]] = y

        else:
            for k, y in saved_state_dict.items():
                new_params[k] = y

        model.load_state_dict(new_params)

        s_epoch = 0
        nss_value = 0
        model_name = args.model_name
        print(model_name)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################

        print('real learning rate %f.' % args.lr)

        title = 'MIT1003-' + model_name
        if args.resume:
            checkpoint = torch.load(os.path.join(path_models, args.ckptname), map_location='cuda:0')
            saved_state_dict = checkpoint['state_dict']
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

            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            s_epoch = checkpoint['epoch']+1

            best_model = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
            nss_value = best_model['nss']

            logger = Logger(os.path.join(path_models, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(path_models, 'log.txt'), title=title)
            logger.set_names(['Epoch', 'LR', 'Train_cps', 'Val_cps', 'Train_h', 'Val_h', 'Train_map', 'Val_map', 'Nss'])


        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()

        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) #, N=48 ******

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w) # , N=32

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch*gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch*gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        folder_name = 'Preds/MIT1003'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size*gpu_number, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)] # batch_size=4
        tgt_sizes = [224] # batch_size=16
        eval_metrics = ('nss',)

        logits_loss = torch.nn.BCEWithLogitsLoss()
        h_loss = HLoss()

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()

        eval_loss = np.inf
        cnt = 0

        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            # adjust_learning_rate(optimizer, i_epoch, args.schedule)
            is_best = False
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw_sa_sp(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)

            results = save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, model_name, test_dataloader,
                                                                   args, tgt_sizes=tgt_sizes, metrics=eval_metrics)
            
            tmp_nss_value = results['nss'].mean
            if tmp_nss_value > nss_value:
                eval_loss = val_cps
                # eval_loss = tmp_eval_loss
                nss_value = tmp_nss_value
                is_best = True
                cnt = 0
                print('Saving model with nss %.4f ...' % nss_value)

            save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=args.ckptname, results=results,
                      is_best=is_best, best_name=args.bestname)

            cnt += 1

            logger.append([i_epoch, optimizer.param_groups[0]['lr'], train_cps, val_cps, train_h, val_h,
                           train_map, val_map, tmp_nss_value])
            if cnt >= args.patience:
                break

        logger.close()
        print('Best model nss: %.4f' % (nss_value))

    # generate saliency maps
    elif phase == 'test_cw_sa_sp_multiscale_210822':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
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

        folder_name = 'Preds/MIT1003'
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
    parser.add_argument("--path_out", default=base_path + 'WF/',
                        type=str,
                        help="""set output path for the trained model""")
    parser.add_argument("--batch-size", default=24, type=int, help="""Set batch size""")
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
    parser.add_argument("--clip", type=float, default=1e-2,
                        help="""Glip gradient norm of relation net""")
    parser.add_argument('--resume', action='store_true',
                        help='whether to resume from folder')
    parser.add_argument('--phase', default='train_cw_aug_sa_sp', type=str, help='running phase')
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
