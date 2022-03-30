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

# >>> torch.__version__ \\
# '1.7.1'
# import horovod.torch as hvd

from load_data_new import SALICON_full, MIT1003_full, MS_COCO_map_full_aug
from load_data_new import collate_fn_salicon_rn, collate_fn_mit1003_rn, collate_fn_coco_map_rn

from models_new import Wildcat_WK_hd_gs_compf_cls_att_A4_cw, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp

from custom_loss_new import HLoss
from config import *
from utils import *

from tensorboardX import SummaryWriter

cps_weight = 1.0
hth_weight = 0.1
hdsup_weight = 0.1
rf_weight = 0.1

def train_Wildcat_WK_hd_compf_map_cw(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    # total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
    # for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums, rf_maps = X


        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        cps_logits, pred_maps = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        # rf_maps = rf_maps - rf_maps.min() # do not have this previously
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        # rf_maps = 0.1 * rf_maps # for gbvs
        # rf_maps = torch.relu(rf_maps - torch.mean(rf_maps.view(rf_maps.size(0), -1), dim=-1, keepdim=True).unsqueeze(2)) # for gbvs_thm

        # losses = loss_HM(pred_logits, gt_labels) # use bce loss with sigmoid
        # losses = logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(cps_logits, (torch.sigmoid(pred_logits)>0.5).float())
        # cps_losses = cps_weight*loss_HM(cps_logits, gt_labels)
        cps_losses = logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)

        # if torch.isnan(pred_maps).any():
        #     pdb.set_trace()

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # h_losses = hth_weight * (0.9**epoch) * info_loss(pred_maps.squeeze(1))
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # print('rf_maps', rf_maps.size(), rf_maps.max(), rf_maps.min())

        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*(0.9**epoch)*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((rf_maps.size(-2), rf_maps.size(-1)))(pred_maps).squeeze(),
        #                                                      min=0.0, max=1.0), rf_maps)

        # losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        # total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        # if i%train_log_interval == 0:
        #     print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
        #           "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
        #         epoch, i, int(N),
        #         cps_losses.item(), np.mean(np.array(total_cps_loss)),
        #         h_losses.item(), np.mean(np.array(total_h_loss)),
        #         rf_losses.item(), np.mean(np.array(total_map_loss))))
        bar.set_description("Train [{}] | cps_loss:{:.4f}({:.4f}) | "
                  "h_loss:{:.4f}({:.4f}) | rf_loss:{:.4f}({:.4f})".format(
                epoch, #i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),np.mean(np.array(total_map_loss))

def train_Wildcat_WK_hd_compf_map_cw_alt_alpha(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, path_models):
    model.train()

    N = len(dataloader)
    # total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    # if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
    #     checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
    #     model_aux.load_state_dict(checkpoint['state_dict'])
    if os.path.exists(os.path.join(path_models, args.bestname)):
        checkpoint = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')  # checkpoint is a dict, containing much info
        # model.load_state_dict(checkpoint['state_dict'])
        saved_state_dict = checkpoint['state_dict'] # cpu version
        new_params = model.state_dict().copy()
        # if list(new_params.keys())[0][:7] == 'module.':
        #     for k, y in saved_state_dict.items():
        #         if k in new_params.keys():
        #             new_params['module.'+k] = y
        # else:
        #     for k, y in saved_state_dict.items():
        #         if k in new_params.keys():
        #             new_params[k] = y

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

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        # inputs, gt_labels, box_features, boxes, boxes_nums, rf_maps = X
        inputs, gt_labels, boxes, boxes_nums, prior_maps = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            prior_maps = prior_maps.cuda()

        cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # if epoch > 0:
        _, aux_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # pdb.set_trace()
        # aux_maps = aux_maps - aux_maps.min()
        # aux_maps = torch.div(aux_maps - torch.min(torch.min(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values,
        #             torch.max(torch.max(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values+1e-9)
        aux_maps = aux_maps - torch.min(torch.min(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        prior_maps = prior_maps - torch.min(torch.min(prior_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        # # prior_maps = torch.relu(prior_maps - torch.mean(prior_maps.view(prior_maps.size(0), -1), dim=-1, keepdim=True).unsqueeze(2))  # for bms_thm
        # # prior_maps = GBVS_R * prior_maps # for gbvs

        rf_maps = ALPHA * aux_maps + (1 - ALPHA) * (prior_maps.unsqueeze(1))
        # rf_maps = prior_maps.unsqueeze(1) # can improve; so the aux_maps range is not good actually?
        # rf_maps = rf_maps - rf_maps.min()
        # losses = 0*logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(cps_logits, (torch.sigmoid(pred_logits)>0.5).float())
        cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))

        # if epoch > 0:
        #     rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0),
        #                                          torch.clamp(rf_maps.detach().squeeze(), min=0.0, max=1.0))
        # else:
        #     rf_losses = 0. * info_loss(pred_maps.squeeze(1))

        rf_losses = rf_weight * torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0),
                                                  torch.clamp(rf_maps.detach().squeeze(), min=0.0, max=1.0))

        # losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        # total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        # if i%train_log_interval == 0:
        #     print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
        #           "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
        #         epoch, i, int(N),
        #         cps_losses.item(), np.mean(np.array(total_cps_loss)),
        #         h_losses.item(), np.mean(np.array(total_h_loss)),
        #         rf_losses.item(), np.mean(np.array(total_map_loss))))

        bar.set_description("Train [{}] | cps_loss:{:.4f}({:.4f})"
                  " | h_loss:{:.4f}({:.4f}) | rf_loss:{:.4f}({:.4f})".format(
                epoch, #i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))

def eval_Wildcat_WK_hd_compf_salicon_cw(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
    # for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        cps_logits, pred_maps = model(img=inputs,
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

        # total_loss.append(losses.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        # if i%train_log_interval == 0:
        #     print("Eval [{}][{}/{}]"
        #           "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
        #         epoch, i, int(N),
        #         cps_losses.item(), np.mean(np.array(total_cps_loss)),
        #         h_losses.item(), np.mean(np.array(total_h_loss)),
        #         map_losses.item(), np.mean(np.array(total_map_loss))))

        bar.set_description("Eval [{}] | "
                  "cps_loss:{:.4f}({:.4f}) | h_loss:{:.4f}({:.4f}) | map_loss:{:.4f}({:.4f})".format(
                epoch, #i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))


    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)) , np.mean(np.array(total_map_loss)) # uncomment for hth_2_x


def train_Wildcat_WK_hd_compf_map_cw_sa(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    # total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums, rf_maps = X


        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        cps_logits, pred_maps, att_scores = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        # rf_maps = rf_maps - rf_maps.min() # do not have this previously
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        # rf_maps = torch.relu(rf_maps - torch.mean(rf_maps.view(rf_maps.size(0), -1), dim=-1, keepdim=True).unsqueeze(2))  # for bms_thm
        # rf_maps = GBVS_R * rf_maps # for gbvs

        # losses = loss_HM(pred_logits, gt_labels) # use bce loss with sigmoid
        # losses = logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(cps_logits, (torch.sigmoid(pred_logits)>0.5).float())
        # cps_losses = cps_weight*loss_HM(cps_logits, gt_labels)
        cps_losses = logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)

        # if torch.isnan(pred_maps).any():
        #     pdb.set_trace()

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # h_losses = hth_weight * (0.9**epoch) * info_loss(pred_maps.squeeze(1))
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # print('rf_maps', rf_maps.size(), rf_maps.max(), rf_maps.min())

        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*(0.9**epoch)*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((rf_maps.size(-2), rf_maps.size(-1)))(pred_maps).squeeze(),
        #                                                      min=0.0, max=1.0), rf_maps)

        # losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        # total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        # if i%train_log_interval == 0:
        #     print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
        #           "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})"
        #           "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
        #         epoch, i, int(N),
        #         cps_losses.item(), np.mean(np.array(total_cps_loss)),
        #         h_losses.item(), np.mean(np.array(total_h_loss)),
        #         rf_losses.item(), np.mean(np.array(total_map_loss)),
        #         att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))

        bar.set_description("Train [{}] | cps:{:.4f}({:.4f})"
                  " | h:{:.4f}({:.4f}) | rf:{:.4f}({:.4f})"
                  " | att_scores:{:.4f}/{:.4f}/{:.4f}".format(
                epoch, # i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss)),
                att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))


    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))

def eval_Wildcat_WK_hd_compf_salicon_cw_sa(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
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

        cps_logits, pred_maps, att_scores = model(img=inputs,
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

        # total_loss.append(losses.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        # if i%train_log_interval == 0:
        #     print("Eval [{}][{}/{}]"
        #           "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})"
        #           "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
        #         epoch, i, int(N),
        #         cps_losses.item(), np.mean(np.array(total_cps_loss)),
        #         h_losses.item(), np.mean(np.array(total_h_loss)),
        #         map_losses.item(), np.mean(np.array(total_map_loss)),
        #         att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))

        bar.set_description("Eval [{}]"
                  " | cps:{:.4f}({:.4f}) | h:{:.4f}({:.4f}) | map:{:.4f}({:.4f})"
                  " | att_scores:{:.4f}/{:.4f}/{:.4f}".format(
                epoch, # i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss)),
                att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))


    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))


def train_Wildcat_WK_hd_compf_map_cw_sa_sp(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    # total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    # total_obj_map_loss = list()
    # for i, X in enumerate(dataloader):
    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums, rf_maps = X


        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        # cps_logits, pred_maps, obj_att_maps = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        cps_logits, pred_maps, _, att_scores = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        # rf_maps = rf_maps - rf_maps.min() # do not have this previously
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        # rf_maps = torch.relu(rf_maps - torch.mean(rf_maps.view(rf_maps.size(0), -1), dim=-1, keepdim=True).unsqueeze(2))  # for bms_thm, gbvs_thm
        # # rf_maps = 0.2 * rf_maps # for gbvs
        # # rf_maps = GBVS_R * rf_maps # for gbvs

        # losses = loss_HM(pred_logits, gt_labels) # use bce loss with sigmoid
        # losses = logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(cps_logits, (torch.sigmoid(pred_logits)>0.5).float())
        # cps_losses = cps_weight*loss_HM(cps_logits, gt_labels)
        cps_losses = logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)

        # if torch.isnan(pred_maps).any():
        #     pdb.set_trace()

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # h_losses = hth_weight * (0.9**epoch) * info_loss(pred_maps.squeeze(1))
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # print('rf_maps', rf_maps.size(), rf_maps.max(), rf_maps.min())

        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*(0.9**epoch)*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((rf_maps.size(-2), rf_maps.size(-1)))(pred_maps).squeeze(),
        #                                                      min=0.0, max=1.0), rf_maps)

        # rf_obj_losses =

        # losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        # total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        # if i%train_log_interval == 0:
        #     # print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
        #     #       "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})"
        #     #       "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
        #     #     epoch, i, int(N),
        #     #     cps_losses.item(), np.mean(np.array(total_cps_loss)),
        #     #     h_losses.item(), np.mean(np.array(total_h_loss)),
        #     #     rf_losses.item(), np.mean(np.array(total_map_loss)),
        #     #     att_scores.max().item(), att_scores.min().item(), torch.argmax(att_scores, dim=0).item()
        #     #     ))
        bar.set_description("Train [{}] | cps_loss:{:.4f}({:.4f})"
              " | h_loss:{:.4f}({:.4f}) | rf_loss:{:.4f}({:.4f})"
              " | att_scores:{:.4f}/{:.4f}/{:.4f}".format(
            epoch, #i, int(N),
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
# from load_data import fixationProcessing
def save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, dataloader, args, tgt_sizes,
                                                 metrics=('kld', 'nss', 'cc', 'sim', 'aucj')):
    # if load_weight: #
    #     checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'))  # checkpoint is a dict, containing much info
    #     # model.load_state_dict(checkpoint['state_dict'])
    #     saved_state_dict = checkpoint['state_dict']
    #     new_params = model.state_dict().copy()
    #     if list(saved_state_dict.keys())[0][:7] == 'module.':
    #         for k, y in saved_state_dict.items():
    #             if k[7:] in new_params.keys():
    #                 new_params[k[7:]] = y
    #     else:
    #         for k, y in saved_state_dict.items():
    #             if k in new_params.keys():
    #                 new_params[k] = y
    #     model.load_state_dict(new_params)
    #
    # # pdb.set_trace()
    # if args.use_gpu:
    #     model.cuda()
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    model.eval()

    # out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_multiscale')
    out_folder = os.path.join(args.path_out, folder_name, best_model_file)
    if len(tgt_sizes)>1:
        out_folder = out_folder+'_multiscale'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # N = len(dataloader) // args.batch_size
    # for i, X in enumerate(dataloader):
    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        # MIT1003 & PASCAL-S image, boxes, sal_map, fix_map(, image_name)
        ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X
        # ori_inputs, ori_boxes, boxes_nums, sal_map, fix_map, img_name = X

        # SALICON image, label, boxes, sal_map, fix_map(, image_name)
        # # ori_inputs, _, ori_boxes, boxes_nums, _, _, img_name = X
        # ori_inputs, _, ori_boxes, boxes_nums, sal_map, fix_map, img_name = X

        # MIT300 & SALICON test image, boxes(, image_name)
        # ori_inputs, ori_boxes, boxes_nums, img_name = X

        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()
            # sal_map = sal_map.cuda()
            # fix_map = fix_map.type(torch.uint8).cuda()
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
            # pdb.set_trace()

            output = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # _, pred_maps, _, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            pred_maps = output[1]

            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)
            pred_final = (pred_maps_all / len(tgt_sizes))
            pred_final_np = pred_final.detach().cpu().numpy()

        # pdb.set_trace()
        for b_i in range(ori_inputs.size(0)):
            ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[b_i] + '.jpeg'))  # height, width, channel
            # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
            # ori_img = scipy.misc.imread(os.path.join(PATH_MIT300, 'images', img_name[0]+'.jpg')) # height, width, channel
            # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'val', img_name[0]+'.jpg')) # height, width, channel
            # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'test', img_name[0]+'.jpg')) # height, width, channel

            # pdb.set_trace()

            scipy.misc.imsave(os.path.join(out_folder, img_name[b_i]+'.png'),
                              postprocess_prediction(pred_final_np[b_i][0], size=[ori_img.shape[0], ori_img.shape[1]]))
            # # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
            # #                   postprocess_prediction_salgan((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
            # #                                             size=[ori_img.shape[0], ori_img.shape[1]])) # the ratio is not right..

    # pdb.set_trace()
    # evaluate
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
    # pdb.set_trace()
    pts = np.round(pts)
    map = np.zeros(dim)
    pts = checkBounds(dim, pts)
    # pdb.set_trace()
    pts = pts.astype('int')
    map[(pts[:,0], pts[:,1])] += 1

    return map

def other_maps(results_size, path_fixation, pred_files, n_aucs_maps=10):
    """Sample reference maps for s-AUC"""
    # while True:
        # this_map = np.zeros(results_size[-2:])
    ids = random.sample(range(len(pred_files)), min(len(pred_files), n_aucs_maps))
    # pdb.set_trace()
    for k in range(len(ids)):
        fix_path = os.path.join(path_fixation, pred_files[ids[k]][:-4] + '_fixPts.jpg')
        fix_map_np = cv2.imread(fix_path, 0)
        fix_map_np = (fix_map_np > 0).astype('float')
        training_resolution = fix_map_np.shape
        # pdb.set_trace()
        rescale = np.array(results_size)/np.array(training_resolution)
        rows, cols = np.where(fix_map_np)
        pts = np.vstack([rows, cols]).transpose()

        if 'fixation_point' not in locals():
            fixation_point = pts.copy()*np.tile(rescale, [pts.shape[0], 1])
        else:
            fixation_point = np.vstack([fixation_point, pts*np.tile(rescale, [pts.shape[0], 1])])

    other_map = makeFixationMap(results_size, fixation_point)
    pdb.set_trace()
    return other_map
        # yield other_map

def evaluate(args, folder_name, best_model_file, metrics):
    assert len(metrics)>0

    results = {metric: tnt.meter.AverageValueMeter() for metric in metrics}
    for metric in metrics:
        results[metric].reset()
    path_saliency = os.path.join(PATH_MIT1003, 'ALLFIXATIONMAPS')
    path_fixation = os.path.join(PATH_MIT1003, 'ALLFIXATIONS')
    # path_fixpts = path_fixation.replace('/ALLSTIMULI', '/ALLFIXATIONS/');
    out_folder = os.path.join(args.path_out, folder_name, best_model_file)
    pred_files = os.listdir(out_folder)
    # pdb.set_trace()
    # bar = tqdm(range(len(pred_files)))
    # for f_i in bar:
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
            elif this_metric == 'aucs':
                other_map = other_maps(pred_final_np.shape[-2:], path_fixation, pred_files, n_aucs_maps=10)
                # pdb.set_trace()
                aucs_val = sal_metrics.auc_shuff_acl(pred_final_np, fix_map_np, other_map) # 0.715 not equal to 0.74
                results[this_metric].add(aucs_val)
            elif this_metric == 'kld':
                kld_val = sal_metrics.kld_loss(pred_final, sal_map)
                results[this_metric].add(kld_val.mean().item(), kld_val.size(0))
            elif this_metric == 'nss':
                nss_val = sal_metrics.nss(pred_final, fix_map) # do not need .exp() for our case; ok!
                results[this_metric].add(nss_val.mean().item(), nss_val.size(0))
            elif this_metric == 'cc':
                cc_val = sal_metrics.corr_coeff(pred_final, sal_map) # do not need .exp() for our case; ok!
                results[this_metric].add(cc_val.mean().item(), cc_val.size(0))
        # pdb.set_trace()

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


        # prior='gbvs'
        prior='nips08'

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, use_grid=True) #################


        '''optimizer'''
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) ############

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        # print('relation lr factor: 1.0')

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

        # if args.use_gpu:
        #     model.cuda()
        # if torch.cuda.device_count()>1:
        #     model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        # ds_train = MS_COCO_ALL_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) #, N=48 ******
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) # ********
        # # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_ALL_map_full_aug(mode='all', img_h=input_h, img_w=input_w, prior=prior) # *******

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w) #, N=24

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch*gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch*gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        # folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'

        # ds_test = SALICON_full(return_path=True, img_h=input_h, img_w=input_w, mode='train')  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
        #                          shuffle=False, num_workers=2)
        # ds_test = SALICON_test(return_path=True, img_h=input_h, img_w=input_w, mode='test')  # N=4,
        # # # ds_test = MIT300_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size * gpu_number, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)] # batch_size=4
        tgt_sizes = [224]  # batch_size=16
        eval_metrics = ('nss',)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        h_loss = HLoss()

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        # pdb.set_trace()
        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            adjust_learning_rate(optimizer, i_epoch, args.schedule) # for SGD
            is_best = False

            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            # tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_loss, map_loss = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)


            # # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)
            #
            # # scheduler.step()
            #
            # # if tmp_eval_loss < eval_loss:
            # #     cnt = 0
            # #     eval_loss = tmp_eval_loss
            # #     print('Saving model ...')
            # #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            # if map_loss<=0.1674:
            #     cnt = 0
            #     # eval_loss = tmp_eval_loss
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            # else:
            #     cnt += 1

            results = save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, model_name, test_dataloader,
                                                                   args, tgt_sizes=tgt_sizes, metrics=eval_metrics)

            tmp_nss_value = results['nss'].mean
            # pdb.set_trace()
            if tmp_nss_value > nss_value:
                eval_loss = val_cps
                # eval_loss = tmp_eval_loss
                nss_value = tmp_nss_value
                is_best = True
                cnt = 0
                print('Saving model with nss %.4f ...' % nss_value)

            # save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name, results=results,
            #           is_best=is_best)
            save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=args.ckptname, results=results,
                       is_best=is_best, best_name=args.bestname)

            cnt += 1

            # if tmp_eval_loss < eval_loss:
            #     cnt = 0
            #     eval_loss = tmp_eval_loss
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name, results=results)
            # elif tmp_eval_loss < 0.830 and map_loss < 0.1630:
            #     cnt = 0
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name, results=results)
            # else:
            #     cnt += 1

            logger.append([i_epoch, optimizer.param_groups[0]['lr'], train_cps, val_cps, train_h, val_h,
                           train_map, val_map, tmp_nss_value])
            if cnt >= args.patience:
                break

        logger.close()
        print('Best model nss: %.4f' % (nss_value))

    # **** 210502 ***
    elif phase == 'train_cw_alt_alpha':
        print('lr %.4f' % args.lr)
        # prior = 'gbvs'
        # prior = 'bms'
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
        # print('alt learning rate: 1e-5')

        s_epoch = 0
        # nss_value = 0
        model_name = args.model_name
        print(model_name)
        title = 'MIT1003-' + model_name
        if args.resume:
            # checkpoint = torch.load(os.path.join(args.path_out, 'Models', model_name + '.pt'))  # checkpoint is a dict, containing much info
            checkpoint = torch.load(os.path.join(path_models, args.ckptname),
                                    map_location='cuda:0')  # checkpoint is a dict, containing much info
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
            # logger.set_names(['Epoch', 'LR', 'T_cps', 'V_cps', 'T_h', 'V_h', 'T_map', 'V_map', 'Nss'])
            logger.set_names(['Epoch', 'LR', 'Train_cps', 'Val_cps', 'Train_h', 'Val_h', 'Train_map', 'Val_map', 'Nss'])

        for param in model_aux.parameters():
            param.requires_grad = False


        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            model_aux = torch.nn.DataParallel(model_aux).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()
        # # ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)  # , N=48 # ***
        # # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_ALL_map_full_aug(mode='all', img_h=input_h, img_w=input_w, prior=prior)  # *******

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)  # , N=24

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch * gpu_number,
                                      collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch * gpu_number,
                                     collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,

        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()

        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        # folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'

        # ds_test = SALICON_full(return_path=True, img_h=input_h, img_w=input_w, mode='train')  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
        #                          shuffle=False, num_workers=2)
        # ds_test = SALICON_test(return_path=True, img_h=input_h, img_w=input_w, mode='test')  # N=4,
        # # # ds_test = MIT300_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
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

    # *** 210511 ***
    elif phase == 'train_cw_aug_sa_art':
        print('lr %.4f'%args.lr)

        prior='nips08'
        # prior='bms'
        # prior='gbvs'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, use_grid=True) #################


        '''init model'''
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', args.init_model, args.bestname))
        # pdb.set_trace()
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
        #nss_value = checkpoint['nss']
        nss_value = 1.5
        model_name = args.model_name
        print(model_name)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  ###for finetuning, best 1e-6
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('real learning rate %f.' % args.lr)

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




        # # fine tune final model ====================================
        # if ATT_RES:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                      '_hth0.1_ms4_fdim512_34_cw_sa_new_fixf_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # saved_state_dict = checkpoint['state_dict']
        # new_params = model.state_dict().copy()
        #
        # if list(saved_state_dict.keys())[0][:7] == 'module.':
        #     for k, y in saved_state_dict.items():
        #         if 'feature_refine' not in k:
        #             new_params[k[7:]] = y
        #
        # else:
        #     for k, y in saved_state_dict.items():
        #         if 'feature_refine' not in k:
        #             new_params[k] = y
        #
        # model.load_state_dict(new_params)


        # # init with final model ===========================================
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_alt2_2_0.95_compf_cls_att_gd_nf4_normTrue_hb_50_aug7' +
        #                                      '_nips08_rf0.1_hth0.1_ms4_fdim512_34_lstm_cw_1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # saved_state_dict = checkpoint['state_dict']
        # new_params = model.state_dict().copy()
        #
        # if list(saved_state_dict.keys())[0][:7] == 'module.':
        #     for k, y in saved_state_dict.items():
        #         if 'feature_refine' not in k:
        #             new_params[k[7:]] = y
        #
        # else:
        #     for k, y in saved_state_dict.items():
        #         if 'feature_refine' not in k:
        #             new_params[k] = y
        #
        # model.load_state_dict(new_params)
        # #
        # fix ============================================
        # for param in model.parameters():
        #     if 'self_attention' not in param.name:
        #         param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

        # if torch.cuda.device_count()>1:
        #     if hasattr(model.module, 'relation_net'):
        #         model.module.relation_net.self_attention.weight.requires_grad = True
        #         model.module.relation_net.self_attention.bias.requires_grad = True
        #     else:
        #         model.module.self_attention.weight.requires_grad = True
        #         model.module.self_attention.bias.requires_grad = True
        #
        # else:
        if hasattr(model, 'relation_net'):
            model.relation_net.self_attention.weight.requires_grad = True
            model.relation_net.self_attention.bias.requires_grad = True
        else:
            model.self_attention.weight.requires_grad = True
            model.self_attention.bias.requires_grad = True

        # ----- for norn --------------
        # if torch.cuda.device_count()>1:
        #     model.module.self_attention.weight.requires_grad = True
        #     model.module.self_attention.bias.requires_grad = True
        # else:
        #     model.self_attention.weight.requires_grad = True
        #     model.self_attention.bias.requires_grad = True



        # -------------------------------------------------


        # if args.use_gpu:
        #     model.cuda()
        # if torch.cuda.device_count()>1:
        #     model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()

        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) # *******
        # # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_ALL_map_full_aug(mode='all', img_h=input_h, img_w=input_w, prior=prior)  # ********

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch * gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch * gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        # folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'

        # ds_test = SALICON_full(return_path=True, img_h=input_h, img_w=input_w, mode='train')  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
        #                          shuffle=False, num_workers=2)
        # ds_test = SALICON_test(return_path=True, img_h=input_h, img_w=input_w, mode='test')  # N=4,
        # # # ds_test = MIT300_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
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
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            # adjust_learning_rate(optimizer, i_epoch, args.schedule)
            is_best = False
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw_sa(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            # tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_loss, map_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw_sa(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            # if tmp_eval_loss < eval_loss:
            #     cnt = 0
            #     eval_loss = tmp_eval_loss
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            # elif tmp_eval_loss < 0.0940 and map_loss<0.1670:
            # # elif tmp_eval_loss < 0.0817 and map_loss<0.1640:
            #     cnt = 0
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            # else:
            #     cnt += 1

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

            # save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name, results=results,
            #           is_best=is_best)
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


        # prior='bms'
        # prior='gbvs'
        prior='nips08'

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, use_grid=True)

        '''checkpoint (init)'''
        # # # init
        # # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08'+
        # #                                      '_rf0.1_hth0.1_ms4_fdim512_34_cw_3_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch06.pt'),
        # #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # # saved_state_dict = checkpoint['state_dict']
        # # new_params = model.state_dict().copy()
        # #
        # # if list(saved_state_dict.keys())[0][:7] == 'module.':
        # #     for k, y in saved_state_dict.items():
        # #         new_params[k[7:]] = y
        # #
        # # else:
        # #     for k, y in saved_state_dict.items():
        # #         new_params[k] = y
        # #
        # # model.load_state_dict(new_params)
        # #
        # # fine tune final model ====================================
        # # if ATT_RES:
        # #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        # #                                      '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01.pt'),
        # #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # # else:
        # #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        # #                                      '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
        # #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # #
        # # saved_state_dict = checkpoint['state_dict']
        # # new_params = model.state_dict().copy()
        # #
        # # if list(saved_state_dict.keys())[0][:7] == 'module.':
        # #     for k, y in saved_state_dict.items():
        # #         if 'feature_refine' not in k:
        # #             if 'gen_g_feature.weight' in k:
        # #                 #pdb.set_trace()
        # #                 new_params[k[7:]] = torch.cat([y, torch.zeros_like(y[:,-1:,:,:])], dim=1)
        # #             else:
        # #                 new_params[k[7:]] = y
        # # else:
        # #     for k, y in saved_state_dict.items():
        # #         if 'feature_refine' not in k:
        # #             if 'gen_g_feature.weight' in k:
        # #                 #0pdb.set_trace()
        # #                 new_params[k] = torch.cat([y, torch.zeros_like(y[:,-1:,:,:])], dim=1)
        # #             else:
        # #                 new_params[k] = y
        # #
        # # model.load_state_dict(new_params)
        #
        # # # init with sa_art_fixf models ====================================
        # if ATT_RES:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                             'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                             '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #
        # else:
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                         'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #     #                         '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                         'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_bms_thm_rf0.1'+
        #     #                         '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                         'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_gbvs_0.5_thm_rf0.1_hth0.1'+
        #     #                         '_ms4_fdim512_34_cw_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                         'resnet50_wildcat_wk_hd_cbA16_alt_2_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_gbvs_0.5_thm_rf0.1'+
        #     #                                  '_hth0.1_2_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch11.pt'), map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                         'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.0'+
        #     #                         '_ms4_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                         'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.0_hth0.1'+
        #     #                         '_ms4_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                         'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1'+
        #     #                         '_ms4_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                         'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_nopsal_rf0.1_hth0.1'+
        #     #                         '_ms4_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                             'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_noGrid_rf0.1_hth0.1'+
        #                             '_ms4_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                      'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_norn_2_rf0.1_hth0.1'+
        #     #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04.pt'), map_location='cuda:0')
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                      'resnet50_wildcat_wk_hd_cbG16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1'+
        #     #                                      '_ms4_sa_art_fixf_2_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch08.pt'), map_location='cuda:0')
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.0_hth0.0'+
        #     #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04.pt'), map_location='cuda:0')
        #
        # saved_state_dict = checkpoint['state_dict']
        # new_params = model.state_dict().copy()
        #
        # if list(saved_state_dict.keys())[0][:7] == 'module.':
        #     for k, y in saved_state_dict.items():
        #         if 'feature_refine' not in k:
        #             new_params[k[7:]] = y
        #
        # else:
        #     for k, y in saved_state_dict.items():
        #         if 'feature_refine' not in k:
        #             new_params[k] = y
        #
        # model.load_state_dict(new_params)

        # init with final model ===========================================
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_alt2_2_0.95_compf_cls_att_gd_nf4_normTrue_hb_50_aug7' +
        #                                      '_nips08_rf0.1_hth0.1_ms4_fdim512_34_lstm_cw_1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # saved_state_dict = checkpoint['state_dict']
        # new_params = model.state_dict().copy()
        #
        # if list(saved_state_dict.keys())[0][:7] == 'module.':
        #     for k, y in saved_state_dict.items():
        #         if 'feature_refine' not in k:
        #             new_params[k[7:]] = y
        #
        # else:
        #     for k, y in saved_state_dict.items():
        #         if 'feature_refine' not in k:
        #             new_params[k] = y
        #
        # model.load_state_dict(new_params)
        #
        # # fix ============================================
        # # for param in model.parameters():
        # #     if 'self_attention' not in param.name:
        # #         param.requires_grad = False
        # for param in model.parameters():
        #     param.requires_grad = False
        #
        # if torch.cuda.device_count() > 1:
        #     if hasattr(model.module, 'relation_net'):
        #         model.module.relation_net.self_attention.weight.requires_grad = True
        #         model.module.relation_net.self_attention.bias.requires_grad = True
        #     else:
        #         model.module.self_attention.weight.requires_grad = True
        #         model.module.self_attention.bias.requires_grad = True
        # else:
        #     if hasattr(model, 'relation_net'):
        #         model.relation_net.self_attention.weight.requires_grad = True
        #         model.relation_net.self_attention.bias.requires_grad = True
        #     else:
        #         model.self_attention.weight.requires_grad = True
        #         model.self_attention.bias.requires_grad = True

        checkpoint = torch.load(os.path.join(args.path_out, 'Models', args.init_model, args.bestname))
        # pdb.set_trace()
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
        #nss_value = checkpoint['nss']
        nss_value = 1.5
        model_name = args.model_name
        print(model_name)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  ###for finetuning, best 1e-6
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('real learning rate %f.' % args.lr)

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



        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) # ********
        # # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_ALL_map_full_aug(mode='all', img_h=input_h, img_w=input_w, prior=prior) # ********

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch * gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch * gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        # folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'

        # ds_test = SALICON_full(return_path=True, img_h=input_h, img_w=input_w, mode='train')  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
        #                          shuffle=False, num_workers=2)
        # ds_test = SALICON_test(return_path=True, img_h=input_h, img_w=input_w, mode='test')  # N=4,
        # # # ds_test = MIT300_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
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
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        # print('finetune lr rate: 1e-5')
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            # adjust_learning_rate(optimizer, i_epoch, args.schedule)
            is_best = False
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw_sa_sp(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            # if tmp_eval_loss < eval_loss:
            #     cnt = 0
            #     eval_loss = tmp_eval_loss
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            # elif tmp_eval_loss < 0.830 and map_loss < 0.1650:
            #     cnt = 0
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            # else:
            #     cnt += 1

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

            # save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name, results=results,
            #           is_best=is_best)
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


        # prior='gbvs'
        # prior='bms'
        prior='nips08'

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, use_grid=True) #################

        '''init model'''
        # # # init
        # # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08'+
        # #                                      '_rf0.1_hth0.1_ms4_fdim512_34_cw_3_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch06.pt'),
        # #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # # saved_state_dict = checkpoint['state_dict']
        # # new_params = model.state_dict().copy()
        # #
        # # if list(saved_state_dict.keys())[0][:7] == 'module.':
        # #     for k, y in saved_state_dict.items():
        # #         new_params[k[7:]] = y
        # #
        # # else:
        # #     for k, y in saved_state_dict.items():
        # #         new_params[k] = y
        # #
        # # model.load_state_dict(new_params)
        # #
        # # fine tune final model ====================================
        # if ATT_RES:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                      '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # else:
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #     #                                  '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_bms_thm_rf0.1_hth0.1'+
        #     #                                  '_ms4_fdim512_34_cw_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_gbvs_0.5_thm_rf0.1_hth0.1'+
        #     #                                  '_ms4_fdim512_34_cw_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #             'resnet50_wildcat_wk_hd_cbA16_alt_2_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_gbvs_0.5_thm_rf0.1'+
        #     #             '_hth0.1_2_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch11.pt'), map_location='cuda:0')
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.0'+
        #     #                                  '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.0_hth0.1'+
        #     #                                  '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_nopsal_rf0.1_hth0.1'+
        #     #                                  '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # # previous weight **********
        #     pass
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.0_hth0.0'+
        #     #                                  '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_noGrid_rf0.1_hth0.1'+
        #     #                                  '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1'+
        #     #                                  '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                          'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_norn_rf0.1_hth0.1' +
        #     #                          '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #     #             map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                          'resnet50_wildcat_wk_hd_cbG16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1'+
        #     #                          '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
        #     #             map_location='cuda:0')  # checkpoint is a dict, containing much info

        checkpoint = torch.load(os.path.join(args.path_out, 'Models', args.init_model, args.bestname))
        # pdb.set_trace()
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
        # nss_value = checkpoint['nss']
        nss_value = 0
        model_name = args.model_name
        print(model_name)

        # # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) ######################
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay) ###### if train using large lr
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) ###########
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        # print('finetune lr rate: 1e-5')
        # print('relation lr factor: 1.0')

        print('real learning rate %f.' % args.lr)

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

            s_epoch = checkpoint['epoch']+1

            best_model = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
            nss_value = best_model['nss']

            logger = Logger(os.path.join(path_models, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(path_models, 'log.txt'), title=title)
            # logger.set_names(['Epoch', 'LR', 'T_cps', 'V_cps', 'T_h', 'V_h', 'T_map', 'V_map', 'Nss'])
            logger.set_names(['Epoch', 'LR', 'Train_cps', 'Val_cps', 'Train_h', 'Val_h', 'Train_map', 'Val_map', 'Nss'])



        # if args.use_gpu:
        #     model.cuda()
        # if torch.cuda.device_count()>1:
        #     model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior) #, N=48 ******
        # # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_ALL_map_full_aug(mode='all', img_h=input_h, img_w=input_w, prior=prior) # ********

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w) # , N=32

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch*gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch*gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # ==== for save_results
        # folder_name = 'Preds/PASCAL-S'
        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        # folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'

        # ds_test = SALICON_full(return_path=True, img_h=input_h, img_w=input_w, mode='train')  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
        #                          shuffle=False, num_workers=2)
        # ds_test = SALICON_test(return_path=True, img_h=input_h, img_w=input_w, mode='test')  # N=4,
        # # # ds_test = MIT300_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
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
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0

        print('Initial nss value: %.4f' % nss_value)
        # args.n_epochs = 5
        for i_epoch in range(s_epoch, args.n_epochs):
            # adjust_learning_rate(optimizer, i_epoch, args.schedule)
            is_best = False
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw_sa_sp(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_loss, map_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

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

            # save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name, results=results,
            #           is_best=is_best)
            save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=args.ckptname, results=results,
                      is_best=is_best, best_name=args.bestname)

            cnt += 1

            # if tmp_eval_loss < eval_loss:
            #     cnt = 0
            #     eval_loss = tmp_eval_loss
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name, results=results)
            # elif tmp_eval_loss < 0.830 and map_loss < 0.1630:
            #     cnt = 0
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name, results=results)
            # else:
            #     cnt += 1

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

        # ***** resume from previous ******
        # checkpoint = torch.load(os.path.join(path_models, args.ckptname), map_location='cuda:0')  # checkpoint is a dict, containing much info
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

        c_epoch = best_model['epoch']

        # best_model = torch.load(os.path.join(path_models, args.bestname), map_location='cuda:0')
        # nss_value = best_model['nss']

        # folder_name = 'Preds/PASCAL-S'
        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        # folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'
        # E_NUM = [1]
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'
        #prior = 'bms'
        # prior = 'gbvs'
        # args.batch_size = 1

        # ds_test = SALICON_full(return_path=True, img_h=input_h, img_w=input_w, mode='train')  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
        #                          shuffle=False, num_workers=2)
        # ds_test = SALICON_test(return_path=True, img_h=input_h, img_w=input_w, mode='test')  # N=4,
        # # # ds_test = MIT300_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
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
    parser.add_argument("--batch-size", default=24,  #cw 72(26xxx) or larger #56(512) can be larger #52 (1024) # 16 5000M, can up to 32 or 64 for larger dataset
                        type=int, # cw512 *80* ; cw1024 *64*; cw512 one5 *32*; cw512 one0 *32(15553),48*; CW512 448input *24*; cw512_101 *42*
                        help="""Set batch size""") # cw512 msl *64*
    parser.add_argument("--train-batch", default=36,  #cw 72(26xxx) or larger #56(512) can be larger #52 (1024) # 16 5000M, can up to 32 or 64 for larger dataset
                        type=int, # cw512 *80* ; cw1024 *64*; cw512 one5 *32*; cw512 one0 *32(15553),48*; CW512 448input *24*; cw512_101 *42*
                        help="""Set batch size""") # cw512 msl *64*
    parser.add_argument("--test-batch", default=20,  #cw 72(26xxx) or larger #56(512) can be larger #52 (1024) # 16 5000M, can up to 32 or 64 for larger dataset
                        type=int, # cw512 *80* ; cw1024 *64*; cw512 one5 *32*; cw512 one0 *32(15553),48*; CW512 448input *24*; cw512_101 *42*
                        help="""Set batch size""") # cw512 msl *64*
    parser.add_argument("--n_epochs", default=200, type=int,
                        help="""Set total number of epochs""")
    parser.add_argument("--lr", type=float, default=1e-4, # 1e-2, # 5e-3,
                        help="""Learning rate for training""")
    parser.add_argument('--schedule', type=int, nargs='+',  # default=[25,50,75,100,125,150,175],
                        default=[60,120,180],
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
