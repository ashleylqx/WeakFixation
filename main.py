import os
import sys
import argparse
import time
import datetime

import numpy as np
import math
import scipy.misc
import pdb

import torch
# torch.cuda.set_device(0)
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

# import horovod.torch as hvd

from load_data import MS_COCO_full, SALICON_full, MIT300_full, MIT1003_full, MS_COCO_map_full, PASCAL_full,\
    MS_COCO_map_full_aug, MS_COCO_map_full_aug_sf, ILSVRC_full, ILSVRC_map_full, ILSVRC_map_full_aug
from load_data import collate_fn_coco_rn, collate_fn_salicon_rn, collate_fn_mit1003_rn, \
                        collate_fn_coco_map_rn, collate_fn_coco_map_rn_multiscale, \
                        collate_fn_ilsvrc_rn, collate_fn_ilsvrc_map_rn, collate_fn_mit300_rn

from models import Wildcat_WK_hd_gs_compf_cls_att_A, Wildcat_WK_hd_gs_compf_cls_att_A_multiscale, \
                Wildcat_WK_hd_gs_compf_cls_att_A_sm, Wildcat_WK_hd_gs_compf_cls_att_A2,\
                Wildcat_WK_hd_gs_compf_cls_att_A2_sm12, Wildcat_WK_hd_gs_compf_cls_att_A_sm12,\
                Wildcat_WK_hd_gs_compf_cls_att_A3, Wildcat_WK_hd_gs_compf_cls_att_A3_sm12,\
                Wildcat_WK_hd_gs_compf_cls_att_A4, Wildcat_WK_hd_gs_compf_cls_att_A5,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_multiscale, Wildcat_WK_hd_gs_compf_cls_att_A6,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw, Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_x,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_try,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_multiscale, Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw_multiscale,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_vib_cwmaps, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_vib_m_cwmaps, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nomlp, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new

from custom_loss import HLoss_th, loss_HM, HLoss_th_3, HLoss_th_2
from config import *
from utils import *

from tensorboardX import SummaryWriter

cps_weight = 1.0
hth_weight = 0.1 #0.1 #1.0 #
hdsup_weight = 0.1  # 0.1, 0.1
rf_weight = 0.1 #0.1 #1.0 #

# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_gbvs_rf{}_hth{}_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_pll_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_fdim{}_2_a'.format(n_gaussian, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_sup2_gd_nf4_normT_eb_sm_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normF_eb_sm_aug2_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normF_eb_{}_aug5_0.2_2_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normFF_eb_{}_aug7_a_A5_fdim{}'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_one5'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_{}_2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, GBVS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_{}_hth_3_2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, GBVS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_lstm_x'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_try_g3g1'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_2_34_cw_try_GS{}'.format(n_gaussian, MAX_BNUM, FEATURE_DIM,GRID_SIZE) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_try_ly34'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_try_g1g1'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_448'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_hth_3_4'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_hth_2_8'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_res101_A4_fdim{}_34_cw_hth_2_one2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_lstm_cw_1'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_sup2_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_lstm_cw_1'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_alt3_2_{}_msl_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_lstm_cw_1'.format(n_gaussian, ALPHA, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_alt3_3_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_lstm_cw_1'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_lstm_cw'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_vib_m_cwmaps_sig_N{}_B{}_D{}'.format(n_gaussian,
#                                                                     MAX_BNUM, FEATURE_DIM,VIB_n_sample,VIB_beta,VIB_dim) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_dcr'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_rng{}_sgd_8'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, RN_GROUP) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_fix'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# if '_sa' in run and ATT_RES:
#     run = run + '_rres'
run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_new_ft'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
if '_sa' in run and ATT_RES:
    run = run + '_rres'

# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_nomlp_sigmoid'.format(n_gaussian, MAX_BNUM, FEATURE_DIM)
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_nomlp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM)

# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug9_3_a_A4_fdim{}_34_cw'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, hth_weight) # 1.0
# run = 'hd_gs_A{}_alt3_2_{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34'.format(n_gaussian, ALPHA, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_{}_gd_nf4_normT_eb_{}_aug7_a_A6_fdim{}'.format(n_gaussian, ALPHA, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normFF_eb_{}_aug7_a_A5_fdim{}_2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_boi{}'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BOI_SIZE) # 1.0
# run = 'hd_gs_A{}_alt2_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_boi{}'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BOI_SIZE) # 1.0
# run = 'hd_gs_A{}_sup2_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_boi{}'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BOI_SIZE) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normF_eb_{}_aug7_2_a_one5'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normF_eb_{}_aug7_sf_3_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_hm_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_RB_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_sup2_gd_nf4_normF_eb_{}_aug7_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_bst_{}_gd_nf4_normF_eb_{}_aug7_a'.format(n_gaussian, ALPHA, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_alt2_{}_gd_nf4_normBR_eb_{}_aug7_a'.format(n_gaussian, ALPHA, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_sup2_{}_gd_nf4_normF_eb_{}_aug7_a'.format(n_gaussian, ALPHA, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normNd_eb_sm_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normNd_eb_sm1_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_smb_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_smb_aug7_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_smb_a_one5'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_aug3_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_a_one0'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_sup2_msl_gd_nf4_normT_eb_a_one5'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_sup2_gd_nf4_normT_eb_a_one5'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_sup2_gd_nf4_normT_eb_smb_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb2_2_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normNd_eb_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_sup2_{}_gd_nf4_normT_eb_a'.format(n_gaussian, ALPHA) # 1.0
# run = 'hd_gs_A{}_sup2_msl_gd_nf4_normT_eb_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_sup3_gd_nf4_normT_eb_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_alt_gd_nf4_normF_eb_a'.format(n_gaussian) # 1.0

# run = 'hd_cbA{}_M4_tgt{}_hth{}_rf{}_ils_eb_aug3_pll'.format(n_gaussian, ilsvrc_num_tgt_classes, hth_weight, rf_weight)
# run = 'hd_cbA{}_M4_tgt{}_hth{}_rf{}_ils_eb_pll'.format(n_gaussian, ilsvrc_num_tgt_classes, hth_weight, rf_weight)
# run = 'hd_cbA{}_M4_tgt{}_hth{}_rf{}_ils_eb_pll_one5'.format(n_gaussian, ilsvrc_num_tgt_classes, hth_weight, rf_weight)

# run = 'hd_gs_A{}_M4_tgt{}_hth{}_rf{}_ils_eb_{}_aug7_a'.format(n_gaussian,
#                         ilsvrc_num_tgt_classes, hth_weight, rf_weight, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_M4_tgt{}_hth{}_rf{}_ils_eb_{}_aug7_a_A4_fdim{}_34'.format(n_gaussian,
#                         ilsvrc_num_tgt_classes, hth_weight, rf_weight, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_M4_tgt{}_hth{}_rf{}_ils_eb_{}_aug7_a_A4_fdim{}_34_cw_4'.format(n_gaussian,
#                         ilsvrc_num_tgt_classes, hth_weight, rf_weight, MAX_BNUM, FEATURE_DIM) # 1.0

# run = 'hd_cbA{}_M2_hth{}_rf{}_ils_eb_pll'.format(n_gaussian, hth_weight, rf_weight)

# run = 'hd_gs_A{}_gd_nf4_normT_eb_nres2_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_norms0.2_a'.format(n_gaussian) # 1.0


print('log dir: {}'.format(os.path.join(PATH_LOG, run)))

writer = SummaryWriter(os.path.join(PATH_LOG, run))
#  tensorboard --logdir=/raid/QX/WF/log --port=6000 # will display all the subfolders within it
#  ssh -NfL or -L localhost:8898:localhost:6000 hz1@172.31.20.57 # at local pc
#  https://127.0.0.1:8898 # local explorer

# no logit loss
def train_Wildcat_WK_hd_compf_map_cw(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    # total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
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

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
            # if True:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.centerbias, 'fc1'):
                    if model.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model, 'box_head'):
                    if hasattr(model.box_head, 'fc6'):
                        writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)

            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module.centerbias, 'fc1'):
                    if model.module.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.module.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model.module, 'box_head'):
                    if hasattr(model.module.box_head, 'fc6'):
                        writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch,
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_cw_gbvs(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    # total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
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
        rf_maps = GBVS_R * rf_maps # for gbvs
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

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
            # if True:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.centerbias, 'fc1'):
                    if model.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model, 'box_head'):
                    if hasattr(model.box_head, 'fc6'):
                        writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)

            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module.centerbias, 'fc1'):
                    if model.module.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.module.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model.module, 'box_head'):
                    if hasattr(model.module.box_head, 'fc6'):
                        writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch,
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def eval_Wildcat_WK_hd_compf_salicon_cw(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
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

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}\tAverage map_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss)) , np.mean(np.array(total_map_loss)) # uncomment for hth_2_x

def eval_Wildcat_WK_hd_compf_cw(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    # total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        # inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            # sal_maps = sal_maps.cuda()

        cps_logits, pred_maps = model(img=inputs,
                                                   boxes=boxes, boxes_nums=boxes_nums)
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        cps_losses = logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        # total_loss.append(losses.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        # total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss))

def eval_Wildcat_WK_hd_compf_map_cw(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    # total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        # cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)


        # total_loss.append(losses.item())
        # total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage h_loss:{:.4f}"
          "\tAverage map_loss:{:.4f}".format(epoch, np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_map_loss))

def test_Wildcat_WK_hd_compf_cw(model, folder_name, best_model_file, dataloader, args):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        saved_state_dict = checkpoint['state_dict']
        if list(saved_state_dict.keys())[0][:7]=='module.':
            new_params = model.state_dict().copy()
            for k,y in saved_state_dict.items():
                new_params[k[7:]] = y
        else:
            new_params = saved_state_dict.copy()
        model.load_state_dict(new_params)

    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        inputs, boxes, boxes_nums, _, _, img_name = X

        if args.use_gpu:
            inputs = inputs.cuda()

            boxes = boxes.cuda()

        ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0]+'.jpeg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        _, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # pred_maps = torch.nn.Sigmoid()(pred_maps)
        print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())

        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction_salgan(pred_maps.squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction(pred_maps.squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..

def test_Wildcat_WK_hd_compf_multiscale_cw(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'))  # checkpoint is a dict, containing much info
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


    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_multiscale')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name) # PASCAL-S
        ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X
        # MIT300 image, boxes(, image_name)
        # ori_inputs, ori_boxes, boxes_nums, img_name = X
        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()

        # ori_img = scipy.misc.imread(
            # os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_MIT300, 'images', img_name[0]+'.jpg')) # height, width, channel

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

            _, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        # print(pred_maps_all.squeeze().size())
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..

def test_Wildcat_WK_hd_compf_multiscale_cw_rank(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'))  # checkpoint is a dict, containing much info
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


    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_multiscale')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    evals = list()
    img_names = list()
    start_time = time.time()
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name) # PASCAL-S
        ori_inputs, ori_boxes, boxes_nums, sal_map, _, img_name = X
        # MIT300 image, boxes(, image_name)
        # ori_inputs, ori_boxes, boxes_nums, img_name = X
        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()
            sal_map = sal_map.cuda()

        ori_img = scipy.misc.imread(
            os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_MIT300, 'images', img_name[0]+'.jpg')) # height, width, channel

        img_names.append(img_name[0])

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

            _, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        # print(pred_maps_all.squeeze().size())
        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        rf_loss = torch.nn.BCELoss()(torch.clamp(pred_maps_all, min=0.0, max=1.0), sal_map)
        evals.append(rf_loss.item())

        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..
    run_time = time.time()-start_time
    print('%.3f sec/image'%(run_time/len(dataloader)))
    inds = np.argsort(np.array(evals))
    img_names = np.array(img_names)
    img_names_sorted = img_names[inds]
    print(img_names_sorted[:100])


# no logit loss; for monitoring box self attention scores
def train_Wildcat_WK_hd_compf_map_cw_sa(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    # total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
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

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})"
                  "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss)),
                att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
            # if True:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                if model.classifier[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    if model.relation_net.pair_pos_fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
                    if hasattr(model.relation_net, 'self_attention') and model.relation_net.self_attention.weight.grad is not None:
                        writer.add_scalar('Grad_hd/sa', model.relation_net.self_attention.weight.grad.abs().mean().item(), niter)
                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.centerbias, 'fc1'):
                    if model.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model, 'box_head'):
                    if model.box_head.fc6.weight.grad is not None:
                        writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)

                #if hasattr(model, 'self_attention'):
                #    writer.add_scalar('Grad_hd/sa', model.self_attention.weight.grad.abs().mean().item(), niter)


            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                if model.module.classifier[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    if model.module.relation_net.pair_pos_fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
                    if hasattr(model.module.relation_net, 'self_attention') and model.module.relation_net.self_attention.weight.grad is not None:
                        writer.add_scalar('Grad_hd/sa', model.module.relation_net.self_attention.weight.grad.abs().mean().item(), niter)
                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module.centerbias, 'fc1'):
                    if model.module.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.module.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model.module, 'box_head'):
                    if model.module.box_head.fc6.weight.grad is not None:
                        writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)

                #if hasattr(model.module, 'self_attention'):
                #    writer.add_scalar('Grad_hd/sa', model.module.self_attention.weight.grad.abs().mean().item(), niter)

    print("Train [{}]\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch,
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def eval_Wildcat_WK_hd_compf_salicon_cw_sa(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
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

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})"
                  "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss)),
                att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}\tAverage map_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))

def eval_Wildcat_WK_hd_compf_cw_sa(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    # total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        # inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            # sal_maps = sal_maps.cuda()

        cps_logits, pred_maps, att_scores = model(img=inputs,
                                                   boxes=boxes, boxes_nums=boxes_nums)
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        cps_losses = logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        # total_loss.append(losses.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        # total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})"
                  "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss))

def eval_Wildcat_WK_hd_compf_map_cw_sa(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    # total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        cps_logits, pred_maps, att_scores = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        # cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)


        # total_loss.append(losses.item())
        # total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})"
                  "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
                epoch, i, int(N),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss)),
                att_scores[:,0].mean().item(), att_scores[:,1].mean().item(), att_scores[:,2].mean().item()))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage h_loss:{:.4f}"
          "\tAverage map_loss:{:.4f}".format(epoch, np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_map_loss))

def test_Wildcat_WK_hd_compf_cw_sa(model, folder_name, best_model_file, dataloader, args):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        saved_state_dict = checkpoint['state_dict']
        if list(saved_state_dict.keys())[0][:7]=='module.':
            new_params = model.state_dict().copy()
            for k,y in saved_state_dict.items():
                new_params[k[7:]] = y
        else:
            new_params = saved_state_dict.copy()
        model.load_state_dict(new_params)

    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        inputs, boxes, boxes_nums, _, _, img_name = X

        if args.use_gpu:
            inputs = inputs.cuda()

            boxes = boxes.cuda()

        ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0]+'.jpeg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        _, pred_maps, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # pred_maps = torch.nn.Sigmoid()(pred_maps)
        print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())

        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction_salgan(pred_maps.squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction(pred_maps.squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..

def test_Wildcat_WK_hd_compf_multiscale_cw_sa(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'))  # checkpoint is a dict, containing much info
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


    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_multiscale')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X
        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()

        ori_img = scipy.misc.imread(
            os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

            _, pred_maps, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        # print(pred_maps_all.squeeze().size())
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..


# no logit loss; for monitoring box self attention scores
def train_Wildcat_WK_hd_compf_map_cw_sa_sp(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    # total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums, rf_maps = X


        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        cps_logits, pred_maps, att_maps = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        # rf_maps = rf_maps - rf_maps.min() # do not have this previously
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values

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

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss)),
                ))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
            # if True:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                if model.classifier[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    if model.relation_net.pair_pos_fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
                    if hasattr(model.relation_net, 'self_attention') and model.relation_net.self_attention.weight.grad is not None:
                        writer.add_scalar('Grad_hd/sa', model.relation_net.self_attention.weight.grad.abs().mean().item(), niter)
                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.centerbias, 'fc1'):
                    if model.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model, 'box_head'):
                    if model.box_head.fc6.weight.grad is not None:
                        writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)

                #if hasattr(model, 'self_attention'):
                #    writer.add_scalar('Grad_hd/sa', model.self_attention.weight.grad.abs().mean().item(), niter)


            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                if model.module.classifier[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    if model.module.relation_net.pair_pos_fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
                    if hasattr(model.module.relation_net, 'self_attention') and model.module.relation_net.self_attention.weight.grad is not None:
                        writer.add_scalar('Grad_hd/sa', model.module.relation_net.self_attention.weight.grad.abs().mean().item(), niter)
                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module.centerbias, 'fc1'):
                    if model.module.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.module.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model.module, 'box_head'):
                    if model.module.box_head.fc6.weight.grad is not None:
                        writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)

                #if hasattr(model.module, 'self_attention'):
                #    writer.add_scalar('Grad_hd/sa', model.module.self_attention.weight.grad.abs().mean().item(), niter)

    print("Train [{}]\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch,
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        cps_logits, pred_maps, att_maps = model(img=inputs,
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

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss)),
                ))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}\tAverage map_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))

def eval_Wildcat_WK_hd_compf_cw_sa_sp(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    # total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        # inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            # sal_maps = sal_maps.cuda()

        cps_logits, pred_maps, att_maps = model(img=inputs,
                                                   boxes=boxes, boxes_nums=boxes_nums)
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        cps_losses = logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        # total_loss.append(losses.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        # total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                ))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss))

def eval_Wildcat_WK_hd_compf_map_cw_sa_sp(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    # total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        cps_logits, pred_maps, att_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        # cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)


        # total_loss.append(losses.item())
        # total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss)),
                ))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage h_loss:{:.4f}"
          "\tAverage map_loss:{:.4f}".format(epoch, np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_map_loss))

def test_Wildcat_WK_hd_compf_cw_sa_sp(model, folder_name, best_model_file, dataloader, args):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        saved_state_dict = checkpoint['state_dict']
        if list(saved_state_dict.keys())[0][:7]=='module.':
            new_params = model.state_dict().copy()
            for k,y in saved_state_dict.items():
                new_params[k[7:]] = y
        else:
            new_params = saved_state_dict.copy()
        model.load_state_dict(new_params)

    if args.use_gpu:
        model.cuda()
    model.eval()

    postfix = '_att'
    out_folder = os.path.join(args.path_out, folder_name, best_model_file+postfix)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        inputs, boxes, boxes_nums, _, _, img_name = X

        if args.use_gpu:
            inputs = inputs.cuda()

            boxes = boxes.cuda()

        ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0]+'.jpeg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        _, pred_maps, att_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # pred_maps = torch.nn.Sigmoid()(pred_maps)
        print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())

        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction_salgan(pred_maps.squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction(pred_maps.squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+postfix+'.png'),
                          postprocess_prediction_my(att_maps.squeeze().detach().cpu().numpy(),
                                                    shape_r=ori_img.shape[0],
                                                    shape_c=ori_img.shape[1])) # the ratio is not right..

def test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'))  # checkpoint is a dict, containing much info
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


    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_multiscale')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X
        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()

        ori_img = scipy.misc.imread(
            os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

            _, pred_maps, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        # print(pred_maps_all.squeeze().size())
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..


# for vib_m_cw_maps
def train_Wildcat_WK_hd_compf_map_cw_vib_logits(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    total_vib_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    total_izy_bound = list()
    total_izx_bound = list()
    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums, rf_maps = X


        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        recon_logits, mu, std, pred_maps = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        # rf_maps = rf_maps - rf_maps.min() # do not have this previously
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values

        class_loss = logits_loss(recon_logits, gt_labels).div(math.log(2))
        # class_loss = F.cross_entropy(recon_logits, gt_labels).div(math.log(2))
        inf_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        vib_loss = class_loss + VIB_beta * inf_loss

        izy_bound = math.log(10, 2) - class_loss
        izx_bound = inf_loss

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # print('rf_maps', rf_maps.size(), rf_maps.max(), rf_maps.min())
        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((rf_maps.size(-2), rf_maps.size(-1)))(pred_maps).squeeze(),
        #                                                      min=0.0, max=1.0), rf_maps)

        # losses.backward(retain_graph=True)
        vib_loss.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_vib_loss.append(vib_loss.item())
        total_cps_loss.append(class_loss.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(rf_losses.item())
        total_izy_bound.append(izy_bound.item())
        total_izx_bound.append(izx_bound.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\tvib_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})"
                  "\tIZY:{:.2f}({:.4f})\tIZX:{:.2f}({:.4f})".format(
                epoch, i, int(N),
                class_loss.item(), np.mean(np.array(total_cps_loss)),
                vib_loss.item(), np.mean(np.array(total_vib_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss)),
                izy_bound.item(), np.mean(np.array(total_izy_bound)),
                izx_bound.item(), np.mean(np.array(total_izx_bound))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', class_loss.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
            # if True:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.centerbias, 'fc1'):
                    if model.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'vib_logits'):
                    writer.add_scalar('Grad_hd/vib_encode0', model.vib_logits.encode[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/vib_decode0', model.vib_logits.decode[0].weight.grad.abs().mean().item(), niter)


            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module.centerbias, 'fc1'):
                    if model.module.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.module.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)

                if hasattr(model.mudule, 'vib_logits'):
                    writer.add_scalar('Grad_hd/vib_encode0', model.mudule.vib_logits.encode[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/vib_decode0', model.mudule.vib_logits.decode[0].weight.grad.abs().mean().item(), niter)


    print("Train [{}]\tAverage cps_loss:{:.4f}\tAverage vib_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}"
          "\tAverage izy_bound:{:.4f}\tAverage izx_bound:{:.4f}".format(epoch,
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_vib_loss)),
                                                 np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss)),
                                                 np.mean(np.array(total_izy_bound)), np.mean(np.array(total_izx_bound))))

def eval_Wildcat_WK_hd_compf_salicon_cw_vib_logits(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    total_vib_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        recon_logits, mu, std, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        class_loss = logits_loss(recon_logits, gt_labels).div(math.log(2))
        # class_loss = F.cross_entropy(recon_logits, gt_labels).div(math.log(2))
        inf_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        vib_loss = class_loss + VIB_beta * inf_loss

        izy_bound = math.log(10, 2) - class_loss
        izx_bound = inf_loss

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((sal_maps.size(-2), sal_maps.size(-1)))(pred_maps).squeeze(),
        #                                             min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        total_cps_loss.append(class_loss.item())
        total_vib_loss.append(vib_loss.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\tvib_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                class_loss.item(), np.mean(np.array(total_cps_loss)),
                vib_loss.item(), np.mean(np.array(total_vib_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', class_loss.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage vib_loss:{:.4f}"
          "\\tAverage h_loss:{:.4f}tAverage map_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_vib_loss)),
              np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss))

def eval_Wildcat_WK_hd_compf_cw_vib_logits(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    total_vib_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    # total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        # inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            # sal_maps = sal_maps.cuda()

        recon_logits, mu, std, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        class_loss = logits_loss(recon_logits, gt_labels).div(math.log(2))
        # class_loss = F.cross_entropy(recon_logits, gt_labels).div(math.log(2))
        inf_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        vib_loss = class_loss + VIB_beta * inf_loss

        izy_bound = math.log(10, 2) - class_loss
        izx_bound = inf_loss

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        total_vib_loss.append(vib_loss.item())
        total_cps_loss.append(class_loss.item())
        total_h_loss.append(h_losses.item())
        # total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\tvib_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                class_loss.item(), np.mean(np.array(total_cps_loss)),
                vib_loss.item(), np.mean(np.array(total_vib_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', class_loss.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage vib_loss:{:.4f}\tAverage h_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_vib_loss)), np.mean(np.array(total_h_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss))

def eval_Wildcat_WK_hd_compf_map_cw_vib_logits(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    # total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        _, _, _, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)


        # total_loss.append(losses.item())
        # total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage h_loss:{:.4f}"
          "\tAverage map_loss:{:.4f}".format(epoch, np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_map_loss))

def test_Wildcat_WK_hd_compf_cw_vib_logits(model, folder_name, best_model_file, dataloader, args):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        saved_state_dict = checkpoint['state_dict']
        if list(saved_state_dict.keys())[0][:7]=='module.':
            new_params = model.state_dict().copy()
            for k,y in saved_state_dict.items():
                new_params[k[7:]] = y
        else:
            new_params = saved_state_dict.copy()
        model.load_state_dict(new_params)

    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        inputs, boxes, boxes_nums, _, _, img_name = X

        if args.use_gpu:
            inputs = inputs.cuda()

            boxes = boxes.cuda()

        ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0]+'.jpeg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        _,_,_, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # pred_maps = torch.nn.Sigmoid()(pred_maps)
        print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())

        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction_salgan(pred_maps.squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction(pred_maps.squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..

def test_Wildcat_WK_hd_compf_multiscale_cw_vib_logits(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'))  # checkpoint is a dict, containing much info
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


    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_multiscale')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X
        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()

        ori_img = scipy.misc.imread(
            os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

            _,_,_, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        # print(pred_maps_all.squeeze().size())
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..


# for vib_cw_maps, two cls losses
def train_Wildcat_WK_hd_compf_map_cw_vib_cw_maps(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    total_vib_loss = list()
    total_h_loss = list()
    total_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    total_izy_bound = list()
    total_izx_bound = list()
    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums, rf_maps = X


        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        recon_logits, cls_logits, mu, std, pred_maps = model(img=inputs,boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        # rf_maps = rf_maps - rf_maps.min() # do not have this previously
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=2, keepdim=True).values, dim=1, keepdim=True).values

        cps_losses = cps_weight*logits_loss(cls_logits, gt_labels)

        class_loss = logits_loss(recon_logits, gt_labels).div(math.log(2))
        # class_loss = F.cross_entropy(recon_logits, gt_labels).div(math.log(2))
        inf_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        vib_loss = class_loss + VIB_beta * inf_loss

        izy_bound = math.log(10, 2) - class_loss
        izx_bound = inf_loss

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # print('rf_maps', rf_maps.size(), rf_maps.max(), rf_maps.min())
        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((rf_maps.size(-2), rf_maps.size(-1)))(pred_maps).squeeze(),
        #                                                      min=0.0, max=1.0), rf_maps)

        cps_losses.backward(retain_graph=True)
        vib_loss.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_vib_loss.append(vib_loss.item())
        total_loss.append(class_loss.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(rf_losses.item())
        total_izy_bound.append(izy_bound.item())
        total_izx_bound.append(izx_bound.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\tvib_cls_loss:{:.4f}({:.4f})\tvib_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})"
                  "\tIZY:{:.2f}({:.4f})\tIZX:{:.2f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                class_loss.item(), np.mean(np.array(total_loss)),
                vib_loss.item(), np.mean(np.array(total_vib_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss)),
                izy_bound.item(), np.mean(np.array(total_izy_bound)),
                izx_bound.item(), np.mean(np.array(total_izx_bound))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
            # if True:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.centerbias, 'fc1'):
                    if model.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'vib_logits'):
                    writer.add_scalar('Grad_hd/vib_encode0', model.vib_logits.encode[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/vib_decode0', model.vib_logits.decode[0].weight.grad.abs().mean().item(), niter)


            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module.centerbias, 'fc1'):
                    if model.module.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.module.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)

                if hasattr(model.mudule, 'vib_logits'):
                    writer.add_scalar('Grad_hd/vib_encode0', model.mudule.vib_logits.encode[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/vib_decode0', model.mudule.vib_logits.decode[0].weight.grad.abs().mean().item(), niter)


    print("Train [{}]\tAverage cps_loss:{:.4f}\tAverage vib_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}"
          "\tAverage izy_bound:{:.4f}\tAverage izx_bound:{:.4f}".format(epoch,
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_vib_loss)),
                                                 np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss)),
                                                 np.mean(np.array(total_izy_bound)), np.mean(np.array(total_izx_bound))))

def eval_Wildcat_WK_hd_compf_salicon_cw_vib_cw_maps(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    total_vib_loss = list()
    total_cps_loss = list()
    total_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        recon_logits, cls_logits, mu, std, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        cps_losses = logits_loss(cls_logits, gt_labels)
        class_loss = logits_loss(recon_logits, gt_labels).div(math.log(2))
        # class_loss = F.cross_entropy(recon_logits, gt_labels).div(math.log(2))
        inf_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        vib_loss = class_loss + VIB_beta * inf_loss

        izy_bound = math.log(10, 2) - class_loss
        izx_bound = inf_loss

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((sal_maps.size(-2), sal_maps.size(-1)))(pred_maps).squeeze(),
        #                                             min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        total_cps_loss.append(cps_losses.item())
        total_loss.append(class_loss.item())
        total_vib_loss.append(vib_loss.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]"
                  "\tcps_loss:{:.4f}({:.4f})\tvib_cls_loss:{:.4f}({:.4f})\tvib_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                class_loss.item(), np.mean(np.array(total_loss)),
                vib_loss.item(), np.mean(np.array(total_vib_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage vib_cls_loss:{:.4f}\tAverage vib_loss:{:.4f}"
          "\\tAverage h_loss:{:.4f}tAverage map_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_cps_loss)), nnp.mean(np.array(total_loss)), np.mean(np.array(total_vib_loss)),
              np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss))


# logit loss and cps loss
def train_Wildcat_WK_hd_compf_map(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums, rf_maps = X


        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            rf_maps = rf_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=inputs,
                                                 boxes=boxes, boxes_nums=boxes_nums)
        if torch.isnan(pred_maps).any():
            print('pred_maps contains nan')

        # losses = loss_HM(pred_logits, gt_labels) # use bce loss with sigmoid
        losses = logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(cps_logits, (torch.sigmoid(pred_logits)>0.5).float())
        # cps_losses = cps_weight*loss_HM(cps_logits, gt_labels)
        cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)

        # if torch.isnan(pred_maps).any():
        #     pdb.set_trace()

        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # print('rf_maps', rf_maps.size(), rf_maps.max(), rf_maps.min())
        rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), rf_maps)
        # rf_losses = rf_weight*torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((rf_maps.size(-2), rf_maps.size(-1)))(pred_maps).squeeze(),
        #                                                      min=0.0, max=1.0), rf_maps)

        losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tloss:{:.4f}({:.4f})\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
            # if True:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.centerbias, 'fc1'):
                    if model.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'feature_refine'):
                    if model.feature_refine.cell_list[0].conv.weight.grad is not None:
                        writer.add_scalar('Grad_hd/lstm_c0', model.feature_refine.cell_list[0].conv.weight.grad.abs().mean().item(), niter)

            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module.centerbias, 'fc1'):
                    if model.module.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.module.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'feature_refine'):
                    if model.module.feature_refine.cell_list[0].conv.weight.grad is not None:
                        writer.add_scalar('Grad_hd/lstm_c0', model.module.feature_refine.cell_list[0].conv.weight.grad.abs().mean().item(), niter)

    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_alt(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, name_model):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
        checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
        model_aux.load_state_dict(checkpoint['state_dict'])

    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        # inputs, gt_labels, box_features, boxes, boxes_nums, rf_maps = X
        inputs, gt_labels, boxes, boxes_nums, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            # rf_maps = rf_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # if epoch > 0:
        _, _, rf_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        # rf_maps = rf_maps - rf_maps.min()
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values

        losses = logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
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

        losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tloss:{:.4f}({:.4f})\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.module.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.module.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_alt_alpha(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, name_model):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
        checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
        model_aux.load_state_dict(checkpoint['state_dict'])

    for i, X in enumerate(dataloader):
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

        pred_logits, cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # if epoch > 0:
        _, _, aux_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        # aux_maps = aux_maps - aux_maps.min()
        aux_maps = aux_maps - torch.min(torch.min(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        rf_maps = ALPHA * aux_maps + (1 - ALPHA) * (prior_maps.unsqueeze(1))
        # rf_maps = rf_maps - rf_maps.min()

        losses = 0*logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
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

        losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tloss:{:.4f}({:.4f})\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module, 'centerbias'):
                    if hasattr(model.module.centerbias, 'fc1'):
                        if model.module.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.module.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_alt_alpha_msl(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, name_model, tgt_sizes):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
        checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
        model_aux.load_state_dict(checkpoint['state_dict'])

    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        # inputs, gt_labels, box_features, boxes, boxes_nums, rf_maps = X
        ori_inputs, gt_labels, ori_boxes, boxes_nums, prior_maps = X

        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            gt_labels = gt_labels.cuda()
            ori_boxes = ori_boxes.cuda()
            prior_maps = prior_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=F.interpolate(ori_inputs, size=(input_h, input_w)),
                                                   boxes=ori_boxes/2,
                                                   boxes_nums=boxes_nums)
        # pred_logits, cps_logits, pred_maps = model(img=ori_inputs, boxes=ori_boxes, boxes_nums=boxes_nums)
        # if epoch > 0:
        # _, _, aux_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        aux_maps = torch.zeros(pred_maps.size(0), 1, output_h, output_w).to(pred_maps.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / input_w * tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / input_h * tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / input_w * tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / input_h * tgt_s

            _, _, tmp_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

            aux_maps += F.interpolate(tmp_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        aux_maps = aux_maps / len(tgt_sizes)

        # aux_maps = aux_maps - aux_maps.min()
        aux_maps = aux_maps - torch.min(torch.min(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        rf_maps = ALPHA * aux_maps + (1 - ALPHA) * (prior_maps.unsqueeze(1))
        # rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values # not good use this

        losses = 0*logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
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

        losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tloss:{:.4f}({:.4f})\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.module.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.module.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_sup(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    #if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
    #    checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
    #    model_aux.load_state_dict(checkpoint['state_dict'])

    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        # inputs, gt_labels, box_features, boxes, boxes_nums, rf_maps = X
        inputs, gt_labels, boxes, boxes_nums, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            # rf_maps = rf_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # if epoch > 0:
        _, _, rf_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        # rf_maps = rf_maps - rf_maps.min()
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        losses = logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
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

        losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tloss:{:.4f}({:.4f})\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module, 'centerbias'):
                    if hasattr(model.module.centerbias, 'fc1'):
                        if model.module.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.module.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_sup_alpha(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    #if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
    #    checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
    #    model_aux.load_state_dict(checkpoint['state_dict'])

    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        # inputs, gt_labels, box_features, boxes, boxes_nums, rf_maps = X
        # inputs, gt_labels, boxes, boxes_nums, _ = X
        inputs, gt_labels, boxes, boxes_nums, prior_maps = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            prior_maps = prior_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # if epoch > 0:
        _, _, aux_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # aux_maps = aux_maps - aux_maps.min()
        aux_maps = aux_maps - torch.min(torch.min(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values

        # print('aux_maps', aux_maps.size(), 'prior_maps', prior_maps.size())
        rf_maps = ALPHA*aux_maps.detach().squeeze() + (1-ALPHA)*prior_maps
        # rf_maps = rf_maps - rf_maps.min()

        losses = 0*logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
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
                                                  torch.clamp(rf_maps, min=0.0, max=1.0))

        losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tloss:{:.4f}({:.4f})\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module, 'centerbias'):
                    if hasattr(model.module.centerbias, 'fc1'):
                        if model.module.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.module.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_bst(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, name_model):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    if epoch>0 and os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
       checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
       model_aux.load_state_dict(checkpoint['state_dict'])

    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        # inputs, gt_labels, box_features, boxes, boxes_nums, rf_maps = X
        # inputs, gt_labels, boxes, boxes_nums, _ = X
        inputs, gt_labels, boxes, boxes_nums, prior_maps = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            prior_maps = prior_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        if epoch > 0:
            _, _, aux_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # print('aux_maps', aux_maps.size(), 'prior_maps', prior_maps.size())
            rf_maps = ALPHA*aux_maps.detach().squeeze() + (1-ALPHA)*prior_maps
        else:
            rf_maps = prior_maps

        losses = logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
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
                                                  torch.clamp(rf_maps, min=0.0, max=1.0))

        losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tloss:{:.4f}({:.4f})\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module, 'centerbias'):
                    if hasattr(model.module.centerbias, 'fc1'):
                        if model.module.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.module.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_sup_msl(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, tgt_sizes):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    #if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
    #    checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
    #    model_aux.load_state_dict(checkpoint['state_dict'])

    for i, X in enumerate(dataloader):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        # inputs, gt_labels, box_features, boxes, boxes_nums, rf_maps = X
        ori_inputs, gt_labels, ori_boxes, boxes_nums, _ = X

        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            gt_labels = gt_labels.cuda()
            ori_boxes = ori_boxes.cuda()
            # rf_maps = rf_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=ori_inputs, boxes=ori_boxes, boxes_nums=boxes_nums)
        # if epoch > 0:

        rf_maps = torch.zeros(pred_maps.size(0), 1, output_h, output_w).to(pred_maps.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / input_w * tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / input_h * tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / input_w * tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / input_h * tgt_s

            _, _, tmp_maps = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

            rf_maps += F.interpolate(tmp_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        rf_maps = rf_maps/len(tgt_sizes)
        # rf_maps = rf_maps - rf_maps.min()
        rf_maps = rf_maps - torch.min(torch.min(rf_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values

        losses = logits_loss(pred_logits, gt_labels) # use bce loss with sigmoid
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

        losses.backward(retain_graph=True)
        cps_losses.backward(retain_graph=True)
        # losses.backward(retain_graph=True)
        h_losses.backward(retain_graph=True)
        rf_losses.backward()

        optimizer.step()
        total_loss.append(losses.item())
        total_h_loss.append(h_losses.item())
        total_cps_loss.append(cps_losses.item())
        total_map_loss.append(rf_losses.item())

        if i%train_log_interval == 0:
            print("Train [{}][{}/{}]\tloss:{:.4f}({:.4f})\tcps_loss:{:.4f}({:.4f})"
                  "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                rf_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model, 'centerbias'):
                    if hasattr(model.centerbias, 'fc1'):
                        if model.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)

            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]', model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    writer.add_scalar('Grad_hd/pair_pos_fc1', model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out', model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)

                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module, 'centerbias'):
                    if hasattr(model.module.centerbias, 'fc1'):
                        if model.module.centerbias.fc1.weight.grad is not None:
                            writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(), niter)
                            writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature',
                            #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                    else:
                        if model.module.centerbias.params.grad is not None:
                            writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(), niter)
                            # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                            #                   niter)

                if hasattr(model.module, 'box_head'):
                    writer.add_scalar('Grad_hd/box_head_fc6', model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7', model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)

    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

def eval_Wildcat_WK_hd_compf_salicon(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()

            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=inputs,
                                                   boxes=boxes, boxes_nums=boxes_nums)

        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = loss_HM(pred_logits, gt_labels)  # use bce loss with sigmoid
        losses = 0*logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # losses = torch.nn.BCEWithLogitsLoss()(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        # cps_losses = cps_weight * loss_HM(cps_logits, gt_labels)
        cps_losses = cps_weight * logits_loss(cps_logits, gt_labels)
        # cps_losses = cps_weight*torch.nn.BCEWithLogitsLoss()(cps_logits, gt_labels)
        # cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(torch.clamp(torch.nn.Upsample((sal_maps.size(-2), sal_maps.size(-1)))(pred_maps).squeeze(),
        #                                             min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        total_loss.append(losses.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]\tloss:{:.4f}({:.4f})"
                  "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Loss', losses.item(), niter)
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}"
          "\tAverage map_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_loss))+np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss))

def eval_Wildcat_WK_hd_compf(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    total_loss = list()
    total_cps_loss = list()
    total_h_loss = list()
    # total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        # inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X
        # ILSVRC  image, label, boxes(, image_name)
        inputs, gt_labels, boxes, boxes_nums = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            # sal_maps = sal_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=inputs,
                                                   boxes=boxes, boxes_nums=boxes_nums)
        # print('pred_maps', pred_maps.size(), pred_maps.max(), pred_maps.min())
        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)

        total_loss.append(losses.item())
        total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        # total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]\tloss:{:.4f}({:.4f})"
                  "\tcps_loss:{:.4f}({:.4f})\th_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N), losses.item(), np.mean(np.array(total_loss)),
                cps_losses.item(), np.mean(np.array(total_cps_loss)),
                h_losses.item(), np.mean(np.array(total_h_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Loss', losses.item(), niter)
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)

    print("Eval [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}".format(epoch,
              np.mean(np.array(total_loss)), np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss))))

    return np.mean(np.array(total_loss))+np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss))

def eval_Wildcat_WK_hd_compf_map(epoch, model, logits_loss, info_loss, dataloader, args):
    model.eval()
    N = len(dataloader)
    # total_loss = list()
    # total_cps_loss = list()
    total_h_loss = list()
    total_map_loss = list()
    for i, X in enumerate(dataloader):
        # SALICON images_batch, labels_batch, boxes_batch, boxes_nums, sal_maps_batch, fix_maps_batch, image_names_
        inputs, gt_labels, boxes, boxes_nums, sal_maps, _ = X

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.cuda()
            boxes = boxes.cuda()
            sal_maps = sal_maps.cuda()

        pred_logits, cps_logits, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

        # losses = logits_loss(pred_logits, ori_logits)
        # losses = logits_loss(pred_logits, torch.argmax(ori_logits, 1))

        # losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
        # cps_losses = cps_weight * logits_loss(cps_logits, (torch.sigmoid(pred_logits) > 0.5).float())
        # cps_losses = cps_weight*logits_loss(cps_logits, gt_labels)
        # losses = logits_loss(torch.sigmoid(pred_logits), gt_labels) # use bce loss with sigmoid
        # cps_losses = cps_weight*logits_loss(torch.sigmoid(cps_logits), gt_labels)
        h_losses = hth_weight * info_loss(pred_maps.squeeze(1))
        map_losses = torch.nn.BCELoss()(torch.clamp(pred_maps.squeeze(), min=0.0, max=1.0), sal_maps)
        # map_losses = torch.nn.BCELoss()(pred_maps.squeeze(), sal_maps)


        # total_loss.append(losses.item())
        # total_cps_loss.append(cps_losses.item())
        total_h_loss.append(h_losses.item())
        total_map_loss.append(map_losses.item())

        if i%train_log_interval == 0:
            print("Eval [{}][{}/{}]\th_loss:{:.4f}({:.4f})\tmap_loss:{:.4f}({:.4f})".format(
                epoch, i, int(N),
                h_losses.item(), np.mean(np.array(total_h_loss)),
                map_losses.item(), np.mean(np.array(total_map_loss))))

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    print("Eval [{}]\tAverage h_loss:{:.4f}"
          "\tAverage map_loss:{:.4f}".format(epoch, np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_map_loss))

def test_Wildcat_WK_hd_compf(model, folder_name, best_model_file, dataloader, args):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'),map_location='cuda:0')  # checkpoint is a dict, containing much info
        saved_state_dict = checkpoint['state_dict']
        if list(saved_state_dict.keys())[0][:7]=='module.':
            new_params = model.state_dict().copy()
            for k,y in saved_state_dict.items():
                new_params[k[7:]] = y
        else:
            new_params = saved_state_dict.copy()
        model.load_state_dict(new_params)

    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        inputs, boxes, boxes_nums, _, _, img_name = X

        if args.use_gpu:
            inputs = inputs.cuda()

            boxes = boxes.cuda()

        ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0]+'.jpeg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        _, _, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # pred_maps = torch.nn.Sigmoid()(pred_maps)
        print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())

        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction_salgan(pred_maps.squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction(pred_maps.squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..

def test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'))  # checkpoint is a dict, containing much info
        model.load_state_dict(checkpoint['state_dict'])

    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_multiscale')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X
        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()

        ori_img = scipy.misc.imread(
            os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

            _, _, pred_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        # print(pred_maps_all.squeeze().size())
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..

def test_Wildcat_WK_hd_compf_gs(model, folder_name, best_model_file, dataloader, args):
    if best_model_file != 'no_training':
        checkpoint = torch.load(os.path.join(args.path_out, 'Models', best_model_file+'.pt'))  # checkpoint is a dict, containing much info
        model.load_state_dict(checkpoint['state_dict'])

    if args.use_gpu:
        model.cuda()
    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_gs')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
        # MIT1003 image, boxes, sal_map, fix_map(, image_name)
        inputs, boxes, boxes_nums, _, _, img_name = X

        if args.use_gpu:
            inputs = inputs.cuda()

            boxes = boxes.cuda()

        ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0]+'.jpeg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        _, _, pred_maps, gaussians, gs_map = model(img=inputs,
                                                   boxes=boxes, boxes_nums=boxes_nums)
        # pred_maps = torch.nn.Sigmoid()(pred_maps)
        # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
        #
        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction(pred_maps.squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder, img_name+'_my.png'),
        #                   postprocess_prediction_my(pred_maps.detach().cpu().numpy(),
        #                                             shape_r=ori_img.shape[0],
        #                                             shape_c=ori_img.shape[1])) # the ratio is not right..
        scipy.misc.imsave(os.path.join(out_folder, img_name[0] + '_gs.png'),
                          gs_map.squeeze().detach().cpu().numpy())
        for i in range(gaussians.size(1)):
            scipy.misc.imsave(os.path.join(out_folder, img_name[0] + '_%02d.png'%i),
                              gaussians[0,i,:,:].detach().cpu().numpy())


def main_Wildcat_WK_hd_compf_map(args):
    path_models = os.path.join(args.path_out, 'Models')
    if not os.path.exists(path_models):
        os.makedirs(path_models)

    # phase = 'test_cw_multiscale'
    # phase = 'test'
    # phase = 'test_cw'
    # phase = 'test_cw_sa'
    phase = 'test_cw_sa_sp'
    # phase = 'test_cw_ils_tgt'

    # phase = 'train_cw_aug'
    # phase = 'train_cw_aug_gbvs'
    # phase = 'train_cw_aug_sa_new'
    # phase = 'train_cw_aug_sa_sp'
    # phase = 'train_cw_aug_sa'
    # phase = 'train_cw_vib_aug'
    # phase = 'train_sup_alpha'
    # phase = 'train_alt_msl_alpha'
    # phase = 'train_alt_alpha'
    # phase = 'train_aug'
    # phase = 'train_ils_tgt_aug'
    # phase = 'train_cw_ils_tgt_aug'

    kmax = 1
    kmin = None
    alpha = 0.7
    num_maps = 4
    fix_feature = False
    dilate = True

    normf = True #'Ndiv'
    # normf = 'Ndiv'
    # normf = 'BR'
    # normf = 'RB'


    # train soft attention model
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor  # computation in GPU
        args.use_gpu = True
        print('Using GPU.')
    else:
        dtype = torch.FloatTensor
        print('Using CPU.')

    if phase == 'train':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)
        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_fdim{}_2_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
                                        n_gaussian, normf, FEATURE_DIM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one0_224'.format(
        #                                 n_gaussian, normf, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        # model = Wildcat_WK_hd_gs_compf_cls_att_A_sm12(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                 fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm12_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model = Wildcat_WK_hd_gs_compf_cls_att_A_sm12(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                 fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_smb_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all




        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)

        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=True, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train2':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)
        model = Wildcat_WK_hd_gs_compf_cls_att_A2(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_norms0.2_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb2_2_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
                                        n_gaussian, normf, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        # model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                 fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all




        print(model_name)

        if args.use_gpu:
            model.cuda()

        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=True, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_pll':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)
        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_pll_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
                                        n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        # model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                 fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all




        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)


        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    # if phase == 'train_hvd':
    #     # cuda version error
    #     # Initialize Horovod
    #     hvd.init()
    #
    #     # Pin GPU to be used to process local rank (one GPU per process)
    #     torch.cuda.set_device(hvd.local_rank())
    #
    #
    #     print('lr %.4f'%args.lr)
    #
    #
    #     prior='nips08'
    #     # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
    #     # #                    fix_feature=fix_feature, dilate=dilate)
    #     model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
    #                      fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
    #     #
    #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_pll_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
    #                                     n_gaussian, normf, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
    #
    #     # prior = 'nips08'
    #     # model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
    #     #                 fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
    #     #
    #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
    #     #                                n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
    #
    #
    #
    #
    #     print(model_name)
    #
    #     if args.use_gpu:
    #         model.cuda()
    #     if torch.cuda.device_count()>1:
    #         model = torch.nn.DataParallel(model)
    #
    #     #folder_name = 'Preds/MIT1003'
    #     #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
    #     ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
    #     #rf_path = os.path.join(args.path_out, folder_name, rf_folder)
    #
    #     # assert os.path.exists(rf_path)
    #     ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w, prior=prior)
    #     # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
    #     # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
    #     # Partition dataset among workers using DistributedSampler
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         ds_train, num_replicas=hvd.size(), rank=hvd.rank())
    #
    #     # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
    #     ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(
    #         ds_validate, num_replicas=hvd.size(), rank=hvd.rank())
    #
    #     # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
    #     #                               shuffle=True, num_workers=2)
    #
    #     train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
    #                                   num_workers=0, sampler=train_sampler) # shuffle=True,
    #
    #     eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
    #                                  num_workers=0, sampler=val_sampler) # shuffle=False,
    #
    #     # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
    #     #                              shuffle=False, num_workers=2)
    #
    #
    #
    #     # logits_loss = torch.nn.CrossEntropyLoss()
    #     logits_loss = torch.nn.BCEWithLogitsLoss()
    #     # logits_loss = torch.nn.BCELoss()
    #     h_loss = HLoss_th()
    #     # h_loss = HLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################3
    #     # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
    #     print('relation lr factor: 1.0')
    #     # Add Horovod Distributed Optimizer
    #     optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    #
    #     # Broadcast parameters from rank 0 to all other processes.
    #     hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    #
    #     if args.use_gpu:
    #         logits_loss = logits_loss.cuda()
    #         h_loss = h_loss.cuda()
    #         # optimizer = optimizer.cuda()
    #
    #
    #     eval_loss = np.inf
    #     # eval_salicon_loss = np.inf
    #     cnt = 0
    #     # args.n_epochs = 5
    #     for i_epoch in range(args.n_epochs):
    #         train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)
    #
    #         tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
    #         # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)
    #
    #         if tmp_eval_loss < eval_loss:
    #             cnt = 0
    #             eval_loss = tmp_eval_loss
    #             print('Saving model ...')
    #             save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
    #         else:
    #             cnt += 1
    #
    #
    #         if cnt >= args.patience:
    #             break

    elif phase == 'train_aug':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        # # #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_snd_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_1_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # A4 fdim512 layer34 0.95
        # checkpoint = torch.load(os.path.join(path_models,
        #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(
        #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info
        #
        # saved_state_dict = checkpoint['state_dict']
        # new_params = model.state_dict().copy()
        # if list(saved_state_dict.keys())[0][:7] == 'module.':
        #     for k, y in saved_state_dict.items():
        #         if k[7:] in new_params.keys():
        #             new_params[k[7:]] = y
        # else:
        #     for k, y in saved_state_dict.items():
        #         if k in new_params.keys():
        #             new_params[k] = y
        #
        # model.load_state_dict(new_params)

        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_cw_aug':
        print('lr %.4f'%args.lr)


        # prior='gbvs'
        prior='nips08'

        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nomlp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # # #
        model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_0.1_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_0.1_rf{}_hth{}_2_8_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # revisit other augmentations
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_3_{}_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all


        # no mlp as box head; use avg pool
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_nomlp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_nomlp_sigmoid_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        # h_loss = HLoss_th() #####
        h_loss = HLoss_th_2()
        # h_loss = HLoss_th_3()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_cw(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            # tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            tmp_eval_loss, map_loss = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            elif map_loss<=0.1670:
                cnt = 0
                # eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_cw_aug_gbvs':
        print('lr %.4f'%args.lr)


        prior='gbvs'
        # prior='nips08'

        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nomlp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # # #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_2_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, GBVS_R, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_rf{}_hth{}_3_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, GBVS_R, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # revisit other augmentations
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_3_{}_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all


        # no mlp as box head; use avg pool
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_nomlp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_nomlp_sigmoid_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th_3()
        # h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_cw_gbvs(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_cw_aug_sa':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # # #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # revisit other augmentations
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_3_{}_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        if ATT_RES:
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_fix_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        else:
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_fix_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        print(model_name)

        # init
        checkpoint = torch.load(os.path.join(args.path_out, 'Models',
                                             'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08'+
                                             '_rf0.1_hth0.1_ms4_fdim512_34_cw_3_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch06.pt'),
                                map_location='cuda:0')  # checkpoint is a dict, containing much info
        saved_state_dict = checkpoint['state_dict']
        new_params = model.state_dict().copy()

        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                new_params[k[7:]] = y

        else:
            for k, y in saved_state_dict.items():
                new_params[k] = y

        model.load_state_dict(new_params)

        # fix
        # for param in model.parameters():
        #     if 'self_attention' not in param.name:
        #         param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

        if torch.cuda.device_count()>1:
            model.module.relation_net.self_attention.weight.requires_grad = True
            model.module.relation_net.self_attention.bias.requires_grad = True
        else:
            model.relation_net.self_attention.weight.requires_grad = True
            model.relation_net.self_attention.bias.requires_grad = True

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_cw_sa(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_cw_aug_sa_new':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # # #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # revisit other augmentations
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_3_{}_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        if ATT_RES:
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_ft_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        else:
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_ft_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        print(model_name)

        # finetuen init ---------------------------------------------
        if ATT_RES:
            checkpoint = torch.load(os.path.join(args.path_out, 'Models',
                                             'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
                                             '_hth0.1_ms4_fdim512_34_cw_sa_new_fix_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
                                map_location='cuda:0')  # checkpoint is a dict, containing much info
        else:
            checkpoint = torch.load(os.path.join(args.path_out, 'Models',
                                                 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
                                                 '_hth0.1_ms4_fdim512_34_cw_sa_new_fix_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
                                    map_location='cuda:0')  # checkpoint is a dict, containing much info

        saved_state_dict = checkpoint['state_dict']
        new_params = model.state_dict().copy()

        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                new_params[k[7:]] = y

        else:
            for k, y in saved_state_dict.items():
                new_params[k] = y

        model.load_state_dict(new_params)

        # # init ---------------------------------------------
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08'+
        #                                      '_rf0.1_hth0.1_ms4_fdim512_34_cw_3_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch06.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # saved_state_dict = checkpoint['state_dict']
        # new_params = model.state_dict().copy()
        #
        # if list(saved_state_dict.keys())[0][:7] == 'module.':
        #     for k, y in saved_state_dict.items():
        #         new_params[k[7:]] = y
        #
        # else:
        #     for k, y in saved_state_dict.items():
        #         new_params[k] = y
        #
        # model.load_state_dict(new_params)
        #
        # # fix -------------------------------------------------------
        # # for param in model.parameters():
        # #     if 'self_attention' not in param.name:
        # #         param.requires_grad = False
        # for param in model.parameters():
        #     param.requires_grad = False
        #
        # if torch.cuda.device_count()>1:
        #     model.module.relation_net.self_attention.weight.requires_grad = True
        #     model.module.relation_net.self_attention.bias.requires_grad = True
        # else:
        #     model.relation_net.self_attention.weight.requires_grad = True
        #     model.relation_net.self_attention.bias.requires_grad = True

        # -------------------------------------------------

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        # h_loss = HLoss_th()
        h_loss = HLoss_th_2()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) ######################
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('real learning rate 1e-5.')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_cw_sa(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            # tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            tmp_eval_loss, map_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            elif tmp_eval_loss < 0.940 and map_loss<0.1670:
                cnt = 0
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_cw_aug_sa_sp':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # # #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # revisit other augmentations
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_3_{}_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        if ATT_RES:
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        else:
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # previsou one5 is actually one2 ... sad ...
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        print(model_name)

        # init
        checkpoint = torch.load(os.path.join(args.path_out, 'Models',
                                             'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08'+
                                             '_rf0.1_hth0.1_ms4_fdim512_34_cw_3_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch06.pt'),
                                map_location='cuda:0')  # checkpoint is a dict, containing much info
        saved_state_dict = checkpoint['state_dict']
        new_params = model.state_dict().copy()

        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                new_params[k[7:]] = y

        else:
            for k, y in saved_state_dict.items():
                new_params[k] = y

        model.load_state_dict(new_params)

        # fix
        # for param in model.parameters():
        #     if 'self_attention' not in param.name:
        #         param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

        if torch.cuda.device_count()>1:
            model.module.relation_net.self_attention.weight.requires_grad = True
            model.module.relation_net.self_attention.bias.requires_grad = True
        else:
            model.relation_net.self_attention.weight.requires_grad = True
            model.relation_net.self_attention.bias.requires_grad = True

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_cw_sa(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_cw_vib_aug':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_vib_m_cwmaps(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        # # #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_vib_m_cwmaps_sig' \
                     '_vibN{}_vib{}_vibD{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,VIB_n_sample,VIB_beta,VIB_dim,
                                        kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # prior = 'nips08'
        #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_cw_vib_logits(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon_cw_vib_logits(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_aug_sf':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        # # #
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_sf_3_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # prior = 'nips08'
        #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all




        print(model_name)

        if args.use_gpu:
            model.cuda()
        # if torch.cuda.device_count()>1:
        #     model = torch.nn.DataParallel(model)
        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug_sf(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_idx':
        print('lr %.4f'%args.lr)

        ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        for idx in range(2, 10):
            if idx>2 and os.path.exists(os.path.join(PATH_LOG, run)):
                os.system("rm -r %s"%os.path.join(PATH_LOG, run))
                print('removing %s ...'% os.path.join(PATH_LOG, run))

            model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                   num_maps=num_maps,
                                                   fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                   normalize_feature=normf)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            if args.use_gpu:
                model.cuda()
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att{}_gd_nf4_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
                n_gaussian, idx, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
            print(model_name)
            eval_loss = np.inf
            # eval_salicon_loss = np.inf
            cnt = 0
            args.n_epochs = 9
            for i_epoch in range(args.n_epochs):
                train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

                tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
                # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

                if tmp_eval_loss < eval_loss:
                    cnt = 0
                    eval_loss = tmp_eval_loss
                    print('Saving model ...')
                    save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
                else:
                    cnt += 1


                if cnt >= args.patience:
                    break

    elif phase == 'train_alt':
        print('lr %.4f' % args.lr)
        #########################################3
        # model = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                     normalize_feature=normf)

        checkpoint = torch.load(os.path.join(path_models,
            'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
            n_gaussian, normf))  # checkpoint is a dict, containing much info

        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(n_gaussian))  # checkpoint is a dict, containing much info
        model_aux.load_state_dict(checkpoint['state_dict'])
        for param in model_aux.parameters():
            param.requires_grad = False

        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have ###############################################3
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_aug_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ##########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model_aux)

        ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_alt(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                              train_dataloader, args, model_name)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if cnt >= args.patience:
                break

    elif phase == 'train_alt_alpha':
        print('lr %.4f' % args.lr)

        #########################################

        # model = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                     normalize_feature=normf)

        if normf == True:
            # # A4 fdim1024 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
            #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

            # A4 fdim512 layer34 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_alt2_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07.pt').format(
            #     n_gaussian, ALPHA, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

            # A4 fdim512 layer34 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(
            #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

            # A4 fdim512 layer34 boi 5 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_boi{}_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM, BOI_SIZE))  # checkpoint is a dict, containing much info

            # # A4 fdim512 layer34 lstm_cw_1 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_lstm_cw_1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch09.pt').format(
            #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

            # # A4 alt2_2 fdim512 layer34 lstm_cw_1 0.95
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_alt2_2_0.95_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_lstm_cw_1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09.pt').format(
                n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

            # one5 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info

            # 0.50
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
            #     n_gaussian, ALPHA, normf, MAX_BNUM))  # checkpoint is a dict, containing much info

            # 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, ALPHA, normf, MAX_BNUM))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05.pt').format(
            #     n_gaussian, normf, MAX_BNUM))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00.pt').format(

            #     n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #    'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02.pt').format(
            #    n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info


        elif normf == 'Ndiv':
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info

        elif normf == 'BR':
            # A BR fdim1024 0.95
            checkpoint = torch.load(os.path.join(path_models,
                                                 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf, MAX_BNUM))  # checkpoint is a dict, containing much info


        else:
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info

        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        saved_state_dict = checkpoint['state_dict']
        if list(saved_state_dict.keys())[0][:7] == 'module.':
            new_params = model.state_dict().copy()
            for k, y in saved_state_dict.items():
                new_params[k[7:]] = y
        else:
            new_params = saved_state_dict.copy()
        model_aux.load_state_dict(new_params)

        for param in model_aux.parameters():
            param.requires_grad = False

        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have ###############################################3

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
        #     dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature,
        #     dilate)  # _gcn_all
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt3_3_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature,dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt2_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature,dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, BOI_SIZE,kmax, kmin, alpha, num_maps, fix_feature,dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt3_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
        #     dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_2_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_aug_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(

        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(

        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)

        ##########################################

        print(model_name)

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model_aux)


        # ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,

        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()

        h_loss = HLoss_th()
        # h_loss = HLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()

            h_loss = h_loss.cuda()

            # optimizer = optimizer.cuda()

        eval_loss = np.inf

        # eval_salicon_loss = np.inf

        cnt = 0

        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_alt_alpha(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                                    train_dataloader, args, model_name)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)

            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:

                cnt = 0

                eval_loss = tmp_eval_loss

                print('Saving model ...')

                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)

            else:

                cnt += 1

            if cnt >= args.patience:
                break

    elif phase == 'train_alt_msl_alpha':
        print('lr %.4f' % args.lr)

        #########################################

        # model = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                          num_maps=num_maps,
        #                                          fix_feature=fix_feature, dilate=dilate, use_grid=True,
        #                                          normalize_feature=normf)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                              num_maps=num_maps,
        #                                              fix_feature=fix_feature, dilate=dilate, use_grid=True,
        #                                              normalize_feature=normf)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                     normalize_feature=normf)

        if normf == True:
            # A4 fdim1024 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
            #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

            # A4 fdim512 layer34 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_alt2_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07.pt').format(
            #     n_gaussian, ALPHA, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

            # A4 fdim512 layer34 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(
            #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_lstm_cw_1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch09.pt').format(
            #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_alt2_0.95_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_lstm_cw_1_msl_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11.pt').format(
                n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info

        # # one5 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info

            # 0.50
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
            #     n_gaussian, ALPHA, normf, MAX_BNUM))  # checkpoint is a dict, containing much info

            # 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, ALPHA, normf, MAX_BNUM))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05.pt').format(
            #     n_gaussian, normf, MAX_BNUM))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00.pt').format(

            #     n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #    'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02.pt').format(
            #    n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info


        elif normf == 'Ndiv':
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info

        else:
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info

        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        model_aux.load_state_dict(checkpoint['state_dict'])

        for param in model_aux.parameters():
            param.requires_grad = False

        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have ###############################################3

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature,
        #     dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_msl_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature,
        #     dilate)  # _gcn_all
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt3_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_1_msl_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature,
            dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
        #     dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_2_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_aug_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(

        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(

        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)

        ##########################################

        print(model_name)

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model_aux)

        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]

        # ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=max(tgt_sizes), img_w=max(tgt_sizes))
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,

        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()

        h_loss = HLoss_th()
        # h_loss = HLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()

            h_loss = h_loss.cuda()

            # optimizer = optimizer.cuda()
        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        eval_loss = np.inf

        # eval_salicon_loss = np.inf

        cnt = 0

        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_alt_alpha_msl(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                                    train_dataloader, args, model_name, tgt_sizes)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)

            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:

                cnt = 0

                eval_loss = tmp_eval_loss

                print('Saving model ...')

                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)

            else:

                cnt += 1

            if cnt >= args.patience:
                break

    elif phase == 'train_sup':
        print('lr %.4f' % args.lr)
        #########################################3
        # model = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                     normalize_feature=normf)

        if normf==True:
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info
            #checkpoint = torch.load(os.path.join(path_models,
            #    'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02.pt').format(
            #    n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05.pt').format(
                n_gaussian, normf, MAX_BNUM))  # checkpoint is a dict, containing much info


        elif normf=='Ndiv':
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
        else:
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info



        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(n_gaussian))  # checkpoint is a dict, containing much info
        model_aux.load_state_dict(checkpoint['state_dict'])
        for param in model_aux.parameters():
            param.requires_grad = False

        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have ###############################################3
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_aug_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ##########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model_aux)

        # ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_sup(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                              train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if cnt >= args.patience:
                break

    elif phase == 'train_sup_alpha':
        print('lr %.4f' % args.lr)
        #########################################3
        # model = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                     normalize_feature=normf)

        if normf==True:
            # A4 fdim512 layer34 lstm_cw_1 0.95
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_lstm_cw_1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch09.pt').format(
                n_gaussian, normf, MAX_BNUM, FEATURE_DIM))  # checkpoint is a dict, containing much info
            # # A4 fdim512 layer34 boi 5 0.95
            # checkpoint = torch.load(os.path.join(path_models,
            #                                      'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms4_fdim{}_34_boi{}_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf, MAX_BNUM, FEATURE_DIM, BOI_SIZE))  # checkpoint is a dict, containing much info

        # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05.pt').format(
            #     n_gaussian, normf, MAX_BNUM))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info
            #checkpoint = torch.load(os.path.join(path_models,
            #    'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02.pt').format(
            #    n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info

        elif normf=='Ndiv':
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
        else:
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info



        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(n_gaussian))  # checkpoint is a dict, containing much info
        saved_state_dict = checkpoint['state_dict']
        if list(saved_state_dict.keys())[0][:7] == 'module.':
            new_params = model.state_dict().copy()
            for k, y in saved_state_dict.items():
                new_params[k[7:]] = y
        else:
            new_params = saved_state_dict.copy()
        model_aux.load_state_dict(new_params)

        # model_aux.load_state_dict(checkpoint['state_dict'])
        for param in model_aux.parameters():
            param.requires_grad = False

        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have ###############################################3
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, BOI_SIZE, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_2_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_aug_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ##########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model_aux)

        # ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_sup_alpha(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                              train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if cnt >= args.patience:
                break

    elif phase == 'train_bst':
        print('lr %.4f' % args.lr)
        #########################################3
        # model = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                     normalize_feature=normf)

        if normf==True:
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05.pt').format(
                n_gaussian, normf, MAX_BNUM))  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info
            #checkpoint = torch.load(os.path.join(path_models,
            #    'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02.pt').format(
            #    n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info

        elif normf=='Ndiv':
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
        else:
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info



        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(n_gaussian))  # checkpoint is a dict, containing much info
        # model_aux.load_state_dict(checkpoint['state_dict'])
        for param in model_aux.parameters():
            param.requires_grad = False

        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have ###############################################3
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_bst_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_2_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, ALPHA, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_aug_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ##########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model_aux)

        # ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_bst(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                              train_dataloader, args, model_name)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if cnt >= args.patience:
                break

    if phase == 'train_sm_aug':

        print('lr %.4f' % args.lr)

        prior = 'nips08'

        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,

        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A_sm12(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)  #################

        # # #

        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_smb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
            dilate)  # _gcn_all

        #

        # prior = 'nips08'
        # model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        print(model_name)

        if args.use_gpu:
            model.cuda()

        # if torch.cuda.device_count()>1:
        #     model = torch.nn.DataParallel(model)
        # folder_name = 'Preds/MIT1003'
        # rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        # rf_path = os.path.join(args.path_out, folder_name, rf_folder)
        # assert os.path.exists(rf_path)

        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)

        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)
        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf

        cnt = 0
        # args.n_epochs = 5

        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)
            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)

            else:
                cnt += 1

            if cnt >= args.patience:
                break

    elif phase == 'train_sm_sup':
        print('lr %.4f' % args.lr)
        #########################################3
        # model = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        model = Wildcat_WK_hd_gs_compf_cls_att_A_sm12(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A_sm12(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                     normalize_feature=normf)

        if normf==True:
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info
            #checkpoint = torch.load(os.path.join(path_models,
            #    'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02.pt').format(
            #    n_gaussian, normf))  # checkpoint is a dict, containing much info
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info

        elif normf=='Ndiv':
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
        else:
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info



        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(n_gaussian))  # checkpoint is a dict, containing much info
        model_aux.load_state_dict(checkpoint['state_dict'])
        for param in model_aux.parameters():
            param.requires_grad = False

        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have ###############################################3
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_smb_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_aug_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ##########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model_aux)


        ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_sup(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                              train_dataloader, args)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if cnt >= args.patience:
                break

    elif phase == 'train_sup_msl':
        print('lr %.4f' % args.lr)
        prior = 'nips08'
        #########################################
        # model = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_G(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False)
        #
        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(n_gaussian))  # checkpoint is a dict, containing much info

        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                     normalize_feature=normf)

        if normf==True:
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info

        elif normf=='Ndiv':
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
        else:
            checkpoint = torch.load(os.path.join(path_models,
                'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03.pt').format(
                n_gaussian, normf))  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(path_models,
            #     'resnet50_wildcat_wk_hd_cbA{}_sup2_compf_cls_att_gd_nf4_norm{}_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04.pt').format(
            #     n_gaussian, normf))  # checkpoint is a dict, containing much info



        # checkpoint = torch.load(os.path.join(path_models,
        #         'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01.pt').format(n_gaussian))  # checkpoint is a dict, containing much info
        model_aux.load_state_dict(checkpoint['state_dict'])
        for param in model_aux.parameters():
            param.requires_grad = False

        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have ###############################################3
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_sup2_msl_compf_cls_att_gd_nf4_norm{}_hb_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
            n_gaussian, normf, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_aug_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ##########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model_aux)

        ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_sup_msl(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                              train_dataloader, args, tgt_sizes)

            tmp_eval_loss = eval_Wildcat_WK_hd_compf_salicon(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if cnt >= args.patience:
                break

    elif phase == 'train_ils':
        print('lr %.4f' % args.lr)

        # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)
        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=False) #################
        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)  #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False) #################
        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have
        # model_name = 'resnet50_wildcat_wk_sft_cbA{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        ##############################
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
            dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ###########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)

        # ds_train = MS_COCO_map_full_ils(mode='train', img_h=input_h, img_w=input_w, normalize_feature=normf)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, normalize_feature=normf)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = ILSVRC_map_full(mode='train', img_h=input_h, img_w=input_w)

        ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate_map = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_ilsvrc_map_rn, #collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_ilsvrc_rn, #collate_fn_coco_rn,
                                     shuffle=False, num_workers=2)

        eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                         shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        logits_loss = torch.nn.CrossEntropyLoss()
        # logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)
            # tmp_eval_loss = 0
            tmp_eval_loss = eval_Wildcat_WK_hd_compf(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss,
                                                                 eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if tmp_eval_salicon_loss < eval_salicon_loss:
                eval_salicon_loss = tmp_eval_salicon_loss
                print('Map loss decrease ...')

            if cnt >= args.patience:
                break

    elif phase == 'train_ils_tgt':
        print('lr %.4f' % args.lr)

        # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)
        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=False) #################
        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_tgt_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)  #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False) #################
        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have
        # model_name = 'resnet50_wildcat_wk_sft_cbA{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        ##############################
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_tgt{}_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
            n_gaussian, normf, ilsvrc_num_tgt_classes, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
            dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ###########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)

        # ds_train = MS_COCO_map_full_ils(mode='train', img_h=input_h, img_w=input_w, normalize_feature=normf)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, normalize_feature=normf)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = ILSVRC_map_full(mode='train', img_h=input_h, img_w=input_w)

        ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate_map = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_ilsvrc_map_rn, #collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_ilsvrc_rn, #collate_fn_coco_rn,
                                     shuffle=False, num_workers=2)

        eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                         shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        logits_loss = torch.nn.CrossEntropyLoss()
        # logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)
            # tmp_eval_loss = 0
            tmp_eval_loss = eval_Wildcat_WK_hd_compf(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss,
                                                                 eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if tmp_eval_salicon_loss < eval_salicon_loss:
                eval_salicon_loss = tmp_eval_salicon_loss
                print('Map loss decrease ...')

            if cnt >= args.patience:
                break

    elif phase == 'train_ils_tgt_aug':
        print('lr %.4f' % args.lr)

        # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)
        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=False) #################
        model = Wildcat_WK_hd_gs_compf_cls_att_A4(n_classes=ilsvrc_num_tgt_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)  #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False) #################
        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have
        # model_name = 'resnet50_wildcat_wk_sft_cbA{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        ##############################
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_aug7_compf_cls_att_gd_nf4_norm{}_hb_{}_tgt{}_rf{}_hth{}_ils4_fdim{}_34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, normf, MAX_BNUM,ilsvrc_num_tgt_classes, rf_weight, hth_weight, FEATURE_DIM,kmax, kmin, alpha, num_maps, fix_feature,
            dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_aug7_compf_cls_att_gd_nf4_norm{}_hb_tgt{}_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, normf, ilsvrc_num_tgt_classes, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
        #     dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ###########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)

        # ds_train = MS_COCO_map_full_ils(mode='train', img_h=input_h, img_w=input_w, normalize_feature=normf)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, normalize_feature=normf)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = ILSVRC_map_full_aug(mode='train', img_h=input_h, img_w=input_w)

        ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate_map = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_ilsvrc_map_rn, #collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_ilsvrc_rn, #collate_fn_coco_rn,
                                     shuffle=False, num_workers=2)

        eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                         shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        logits_loss = torch.nn.CrossEntropyLoss()
        # logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)
            # tmp_eval_loss = 0
            tmp_eval_loss = eval_Wildcat_WK_hd_compf(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss,
                                                                 eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if tmp_eval_salicon_loss < eval_salicon_loss:
                eval_salicon_loss = tmp_eval_salicon_loss
                print('Map loss decrease ...')

            if cnt >= args.patience:
                break

    elif phase == 'train_cw_ils_tgt_aug':
        print('lr %.4f' % args.lr)

        # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)
        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=False) #################
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=ilsvrc_num_tgt_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)  #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=False) #################
        #
        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_3_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # Note that _4 and _5 go not have res in rn, but _, _2, _3 have
        # model_name = 'resnet50_wildcat_wk_sft_cbA{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        ##############################
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 n_gaussian, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_aug7_compf_cls_att_gd_nf4_norm{}_hb_{}_tgt{}_rf{}_hth{}_ils4_fdim{}_34_cw_3_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
            n_gaussian, normf, MAX_BNUM,ilsvrc_num_tgt_classes, rf_weight, hth_weight, FEATURE_DIM,kmax, kmin, alpha, num_maps, fix_feature,
            dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_aug7_compf_cls_att_gd_nf4_norm{}_hb_tgt{}_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #     n_gaussian, normf, ilsvrc_num_tgt_classes, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
        #     dilate)  # _gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_sameb2_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_compf_sameb2_gs_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                 hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn

        # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate)

        # model_name = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        #                                hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate)
        ###########################################
        print(model_name)

        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)

        # ds_train = MS_COCO_map_full_ils(mode='train', img_h=input_h, img_w=input_w, normalize_feature=normf)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, normalize_feature=normf)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = ILSVRC_map_full_aug(mode='train', img_h=input_h, img_w=input_w)

        ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate_map = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_ilsvrc_map_rn, #collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.batch_size, collate_fn=collate_fn_ilsvrc_rn, #collate_fn_coco_rn,
                                     shuffle=False, num_workers=2)

        eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                         shuffle=False, num_workers=2)

        # eval_map_dataloader = DataLoader(ds_validate_map, batch_size=args.batch_size, collate_fn=collate_fn_salicon,
        #                              shuffle=False, num_workers=2)

        logits_loss = torch.nn.CrossEntropyLoss()
        # logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ######################3
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_cw(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)
            # tmp_eval_loss = 0
            tmp_eval_loss = eval_Wildcat_WK_hd_compf_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map_cw(i_epoch, model, logits_loss, h_loss,
                                                                 eval_map_dataloader, args)

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1

            if tmp_eval_salicon_loss < eval_salicon_loss:
                eval_salicon_loss = tmp_eval_salicon_loss
                print('Map loss decrease ...')

            if cnt >= args.patience:
                break



    # generate maps
    elif phase == 'test':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        e_num = 5 #1 2 3 5 6

        # prior = 'nips08'
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_gbvs_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_sup2_msl_compf_cls_att_gd_nf4_norm{}_hb_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        #best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_aug3_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #    n_gaussian, normf, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_norms_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #    n_gaussian, normf, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #   n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_fdim{}_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #   n_gaussian, normf, FEATURE_DIM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, ALPHA, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all


        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, BOI_SIZE,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224_epoch{:02d}'.format(
        #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt3_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_1_msl_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
                 n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt2_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, BOI_SIZE,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, BOI_SIZE,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt3_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_mxp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        #
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_sf_2_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt3_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_3_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        #
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, 'smb', rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        print("Testing %s ..."%best_model_file)
        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        test_Wildcat_WK_hd_compf(model, folder_name, best_model_file, test_dataloader, args)


        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
        # args.batch_size = 1
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
        #                              shuffle=False, num_workers=2)
        # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        E_NUM = [1]
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)

        for e_num in E_NUM:
            # e_num = 4 #1 2 3 5 6


            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_gbvs_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_sup2_msl_compf_cls_att_gd_nf4_norm{}_hb_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
            #best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_aug3_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #    n_gaussian, normf, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_norms_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #    n_gaussian, normf, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #   n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_fdim{}_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #   n_gaussian, normf, FEATURE_DIM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_compf_cls_att_gd_nf4_norm{}_hb_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, ALPHA, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_0.1_rf{}_hth{}_2_3_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,
                dilate, e_num)  # _gcn_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, BOI_SIZE,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, RN_GROUP, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_6_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, RN_GROUP, kmax, kmin, alpha,
            #     num_maps, fix_feature, dilate, e_num)  # _gcn_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_0.25_2_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #     n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha,
            #     num_maps, fix_feature, dilate, e_num)  # _gcn_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_2_nips08_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_2_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, GRID_SIZE,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
            #
            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, BOI_SIZE,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_sup2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, BOI_SIZE,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt3_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_mxp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
            #
            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_sf_2_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt3_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_3_nips08_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #          n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
            #
            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #     n_gaussian, normf, 'smb', rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

            print("Testing %s ..."%best_model_file)
            # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            test_Wildcat_WK_hd_compf_cw(model, folder_name, best_model_file, test_dataloader, args)


            # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
            # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw_sa':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        E_NUM = [4]
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)

        for e_num in E_NUM:

            if ATT_RES:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all

            else:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all

            print("Testing %s ..."%best_model_file)
            # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            test_Wildcat_WK_hd_compf_cw_sa(model, folder_name, best_model_file, test_dataloader, args)


            # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
            # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw_sa_sp':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        E_NUM = [4]
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)

        for e_num in E_NUM:

            if ATT_RES:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all

            else:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all

            print("Testing %s ..."%best_model_file)
            # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_cw_sa(model, folder_name, best_model_file, test_dataloader, args)
            test_Wildcat_WK_hd_compf_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args)


            # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
            # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_multiscale':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)

        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)

        # ------------------------------------------
        #model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                   fix_feature=fix_feature, dilate=dilate)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        e_num = 7 #1 2 3 5 6
        # best_model_file = 'resnet50_wildcat_wk_epoch%02d'%e_num
        # best_model_file = 'resnet101_wildcat_wk_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_ms_epoch{:02d}'.format(
        #     kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt2_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
                 n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_signorm_nosigmap_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_sft_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # # #
        print("Testing %s ..."%best_model_file)
        # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
        # args.batch_size = 1
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
        #                              shuffle=False, num_workers=2)
        # test_Wildcat_WK_hd_compf(model, folder_name, best_model_file, test_dataloader, args)


        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw_multiscale':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)

        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)

        # ------------------------------------------
        #model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                   fix_feature=fix_feature, dilate=dilate)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/PASCAL-S'
        # folder_name = 'Preds/MIT300'
        # best_model_file = 'no_training'
        e_num = 9 #1 2 3 5 6
        # best_model_file = 'resnet50_wildcat_wk_epoch%02d'%e_num
        # best_model_file = 'resnet101_wildcat_wk_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_ms_epoch{:02d}'.format(
        #     kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt2_2_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_fdim{}_34_lstm_cw_1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
                 n_gaussian, ALPHA, normf, MAX_BNUM, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all

        # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_signorm_nosigmap_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_sft_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # # #
        print("Testing %s ..."%best_model_file)
        # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
        # args.batch_size = 1
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
        #                              shuffle=False, num_workers=2)
        # test_Wildcat_WK_hd_compf(model, folder_name, best_model_file, test_dataloader, args)


        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        # ds_test = PASCAL_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
        ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
        # ds_test = MIT300_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)
        test_Wildcat_WK_hd_compf_multiscale_cw_rank(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_ils_tgt':
        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_tgt_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_tgt_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                          num_maps=num_maps,
        #                                          fix_feature=fix_feature, dilate=dilate, use_grid=True,
        #                                          normalize_feature=normf)

        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                   fix_feature=fix_feature, dilate=dilate)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        e_num = 2  # 1 2 3 5 6
        # best_model_file = 'resnet50_wildcat_wk_epoch%02d'%e_num
        # best_model_file = 'resnet101_wildcat_wk_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_ms_epoch{:02d}'.format(
        #     kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_tgt{}_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, ilsvrc_num_tgt_classes, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate,
        #     e_num)  ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_tgt{}_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, ilsvrc_num_tgt_classes, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate,
        #     e_num)  ####_all

        best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_aug7_compf_cls_att_gd_nf4_norm{}_hb_{}_tgt{}_rf{}_hth{}_ils4_fdim{}_34_cw_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            n_gaussian, normf, MAX_BNUM, ilsvrc_num_tgt_classes, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature,
            dilate, e_num)  ####_all

        # # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_signorm_nosigmap_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_sft_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # # #
        print("Testing %s ..." % best_model_file)
        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        test_Wildcat_WK_hd_compf(model, folder_name, best_model_file, test_dataloader, args)

        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes), normalize_feature=normf)  # N=4,
        # args.batch_size = 1
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
        #                             shuffle=False, num_workers=2)
        # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw_ils_tgt':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=ilsvrc_num_tgt_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                 normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=ilsvrc_num_tgt_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                          num_maps=num_maps,
        #                                          fix_feature=fix_feature, dilate=dilate, use_grid=True,
        #                                          normalize_feature=normf)

        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # # ------------------------------------------
        # model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                   fix_feature=fix_feature, dilate=dilate)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        e_num = 2  # 1 2 3 5 6
        # best_model_file = 'resnet50_wildcat_wk_epoch%02d'%e_num
        # best_model_file = 'resnet101_wildcat_wk_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_ms_epoch{:02d}'.format(
        #     kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_tgt{}_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, ilsvrc_num_tgt_classes, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate,
        #     e_num)  ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_tgt{}_rf{}_hth{}_ils_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, normf, ilsvrc_num_tgt_classes, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate,
        #     e_num)  ####_all

        best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_aug7_compf_cls_att_gd_nf4_norm{}_hb_{}_tgt{}_rf{}_hth{}_ils4_fdim{}_34_cw_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            n_gaussian, normf, MAX_BNUM, ilsvrc_num_tgt_classes, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature,
            dilate, e_num)  ####_all

        # # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     n_gaussian, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_signorm_nosigmap_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_sft_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # # #
        print("Testing %s ..." % best_model_file)
        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        test_Wildcat_WK_hd_compf_cw(model, folder_name, best_model_file, test_dataloader, args)

        # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes), normalize_feature=normf)  # N=4,
        # args.batch_size = 1
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
        #                             shuffle=False, num_workers=2)
        # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_gs':
        model = Wildcat_WK_hd_gs_compf_cls_att_A(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate)

        # model = Wildcat_WK_hd_compf_rn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # model = Wildcat_WK_hd_compf(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate)
        #
        # # ------------------------------------------
        #model = Wildcat_WK_hd_compf_x(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                   fix_feature=fix_feature, dilate=dilate)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        e_num = 2 #3 4
        # best_model_file = 'resnet50_wildcat_wk_epoch%02d'%e_num
        # best_model_file = 'resnet101_wildcat_wk_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_ms_epoch{:02d}'.format(
        #     kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
            n_gaussian, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_compf_rn_3_nf_all_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hd_gcn_compf_grid7_sig_nf_all_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num) ####_all
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_signorm_nosigmap_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # best_model_file = 'resnet50_wildcat_wk_hth{}_ms_sft_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_448_epoch{:02d}'.format(
        #     hth_weight,kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
        # # #
        print("Testing %s ..."%best_model_file)
        ds_test = MIT1003_full(N=4,return_path=True, img_h=input_h, img_w=input_w)  #
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        test_Wildcat_WK_hd_compf_gs(model, folder_name, best_model_file, test_dataloader, args)




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_out", default=base_path + 'WF/',
                        type=str,
                        help="""set output path for the trained model""")
    parser.add_argument("--batch_size", default=80*torch.cuda.device_count(),  #cw 72(26xxx) or larger #56(512) can be larger #52 (1024) # 16 5000M, can up to 32 or 64 for larger dataset
                        type=int, # cw512 *80* ; cw1024 *64*; cw512 one5 *32*; cw512 one0 *32(15553),48*; CW512 448input *24*; cw512_101 *42*
                        help="""Set batch size""") # cw512 msl *64*
    parser.add_argument("--n_epochs", default=500, type=int,
                        help="""Set total number of epochs""")
    parser.add_argument("--lr", type=float, default=1e-4, # 1e-2, # 5e-3,
                        help="""Learning rate for training""")
    parser.add_argument("--patience", type=int, default=5,
                        help="""Patience for learning rate scheduler (default 3)""")
    parser.add_argument("--use_gpu", type=bool, default=False,
                        help="""Whether use GPU (default False)""")
    parser.add_argument("--clip", type=float, default=1e-2,
                        help="""Glip gradient norm of relation net""")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    main_Wildcat_WK_hd_compf_map(args)