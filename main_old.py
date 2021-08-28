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

from load_data_old import MS_COCO_full, SALICON_full, MIT300_full, MIT1003_full, MS_COCO_map_full, PASCAL_full, SALICON_test,\
    MS_COCO_map_full_aug, MS_COCO_map_full_aug_sf, ILSVRC_full, ILSVRC_map_full, ILSVRC_map_full_aug, MS_COCO_ALL_map_full_aug,\
    MS_COCO_map_full_aug_prior, MS_COCO_ALL_map_full_aug_prior
from load_data import collate_fn_coco_rn, collate_fn_salicon_rn, collate_fn_mit1003_rn, \
                        collate_fn_coco_map_rn, collate_fn_coco_map_rn_multiscale, \
                        collate_fn_ilsvrc_rn, collate_fn_ilsvrc_map_rn, collate_fn_mit300_rn

from models import Wildcat_WK_hd_gs_compf_cls_att_A, Wildcat_WK_hd_gs_compf_cls_att_A_multiscale, \
                Wildcat_WK_hd_gs_compf_cls_att_A2,\
                Wildcat_WK_hd_gs_compf_cls_att_A2_sm12, Wildcat_WK_hd_gs_compf_cls_att_A_sm12,\
                Wildcat_WK_hd_gs_compf_cls_att_A3, Wildcat_WK_hd_gs_compf_cls_att_A3_sm12,\
                Wildcat_WK_hd_gs_compf_cls_att_A4, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_multiscale, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw, Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_x,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nomlp, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_vis, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_rank, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_catX, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_noobj, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art, \
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art_sp, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art_sp,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art_sp, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art_sp,\
    Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sft, Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_rank_rebuttal

from custom_loss import HLoss_th, loss_HM, HLoss_th_3, HLoss_th_2, HLoss_th_210423
from config import *
from utils import *

from tensorboardX import SummaryWriter

cps_weight = 1.0
hth_weight = 0.1 #0.1 #1.0 #
hdsup_weight = 0.1  # 0.1, 0.1
rf_weight = 0.1 #0.1 #1.0 #

# run = '0123_2'
# run = '0123_3'
# run = '0123_4'
# run = '210423'
# run = '210424_sgd_2'
# run = 'basemodel_210427_sgd'  # [wrong] sub mean
# run = 'basemodel_210427_adam' # [wrong] sub mean
# run = 'basemodel_210508_adam' # do not sub mean
# run = 'basemodel_alt_210502_adam' # 1e-4
run = 'basemodel_alt_210509_adam' # init adam
run = 'basemodel_alt_210509_adam_2' # init sgd
# run = 'basemodel_210503_sgd'  # [wrong] sub mean
# run = 'basemodel_210504_sgd'  # [wrong] sub mean
run = 'basemodel_210511_sgd'    # 0.01-->0.1, not good
run = 'sa_art_210511_adam'   # 1e-5
run = 'sa_art_210511_adam_2' # 1e-4
run = 'sa_sp_fixf_210511_adam' # 1e-5, basemodel init
run = 'sa_sp_alt_210511_adam' # 1e-5, using sa_sp trained from scratch nss 1.47
# run = 'sa_sp_alt_210511_sgd' # 1e-2, using sa_sp trained from scratch nss 1.47
run = 'sa_art_210514_adam' # 1e-5, init from basemodel_alt_210509_adam_2 1.7505
run = 'sa_sp_fixf_210515_adam' # 1e-5, init from basemodel_alt_210509_adam_2 1.7505
run = 'sa_sp_210516_adam' # 1e-5, init from sa_sp_fixf_210515_adam 1.7802
# run = 'sa_sp_210516_sgd' # 1e-2, init from sa_sp_fixf_210515_adam 1.7802
run = 'sa_sp_210517_adam' # 1e-5, init from sa_sp_fixf_210515_adam 1.7802
run = 'sa_sp_210517_adam_2' # 1e-4, init from sa_sp_fixf_210515_adam 1.7802
run = 'sa_sp_alt_210517_adam' # 1e-5, init from sa_sp_210517_adam 1.7329
##run = 'sa_sp_alt_210517_sgd' # 1e-2, init from sa_sp_210517_adam 1.7329
#run = 'sa_sp_alt_210520_sgd' # 1e-2, init from sa_sp_210517_adam 1.7329
# run = 'sa_sp_alt_210518_adam' # 1e-5, init from sa_sp_fixf_210515_adam 1.7802
run = 'sa_sp_alt_210522_adam' # 1e-5, init from sa_sp_210517_adam 1.7329; init both
run = 'sa_sp_alt_210523_adam' # 1e-5, init from sa_sp_210517_adam 1.7329; init both; aux_maps = aux_maps * ALT_RATIO
run = 'sa_sp_alt_210524_adam' # 1e-5, init from sa6_sp_fixf_210515_adam 1.7802; init both; no ALT_RATIO
run = 'sa_sp_alt_210526_sgd' # 1e-2, init from sa_sp_fixf_210515_adam 1.7802; init both; no ALT_RATIO
run = 'sa_sp_210528_all_adam' # 1e-4, init from sa_sp_fixf_210515_adam 1.7802; train COCO_ALL
run = 'basemodel_210528_all_sgd' # 1e-2, init from sa_sp_fixf_210515_adam 1.7802; train COCO_ALL
# run = 'sa_sp_fixf_210529_all_adam' # 1e-5, init from sa_art_210514_adam 1.7342; train COCO_ALL
run = 'basemodel_alt_210531_all_adam' # 1e-5, init from basemodel_210528_all_sgd 1.4810; train COCO_ALL
run = 'sa_art_210607_all_adam' # 1e-5, init from basemodel_alt_210531_all_adam 1.6821; train COCO_ALL
run = 'sa_sp_fixf_210610_all_adam' # 1e-5, init from sa_art_210607_all_adam 1.6912; train COCO_ALL
run = 'tmp' #

run = 'basemodel_21822_sgd_mcg' # lr 0.01
run = 'basemodel_alt_210825_adam'

'''old run folder'''
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_bms_thm'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BMS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_bms_thm_fixf'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BMS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_bms_thm_fixf_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BMS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_bms_thm_ftf_2_3_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BMS_R) # 1.0
# run = 'hd_gs_A{}_alt_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_bms_thm_ftf_2_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BMS_R) # 1.0
# run = 'hd_gs_A{}_alt_4_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_bms_thm'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, BMS_R) # 1.0

# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_thm_{}_fixf_2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, GBVS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_thm_{}_fixf_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, GBVS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_thm_{}_ftf_2_2_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, GBVS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_thm_{}'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, GBVS_R) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_gbvs_rf{}_hth{}_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_alt_2_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_thm_{}'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, GBVS_R) # 1.0

# run = 'hd_gs_G{}_alt_5_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_G{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_fixf_3'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_G{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_fixf_2_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_G{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_ftf_2_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_nobs_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_fixf_sp'.format(MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_nobs_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_ftf_2_sp'.format(MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_nobs_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_fixf'.format(MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_nobs_alt_gd_nf4_normT_eb_{}_audg7_rf{}_hth{}_a'.format(MAX_BNUM, rf_weight, hth_weight) # 1.0

# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_twocls_2_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_2_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_all_8_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_all_9_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_proa_{}_aug7_rf{}_hth{}_2_a'.format(n_gaussian, MAX_BNUM, PRO_RATIO, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_sft_2_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_fixf_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_ftf_2_4_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_fixf_3'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_a_fixf_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0

# run = 'hd_gs_A{}_alt_2_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_noGrid_2_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_noGrid_a_fixf'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_noGrid_a_fixf_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_noGrid_a_ftf_2_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_nopsal_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_nopsal_a_fixf'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_nopsal_a_fixf_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_nopsal_a_ftf_2_4_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_alt_2_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_noobj_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_norn_2_a'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_norn_a_fixf_2'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_norn_a_fixf_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_rf{}_hth{}_norn_a_ftf_2_sp'.format(n_gaussian, MAX_BNUM, rf_weight, hth_weight) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0


# run = 'hd_gs_A{}_gd_nf4_normT_eb_pll_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_fdim{}_2_a'.format(n_gaussian, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_sup2_gd_nf4_normT_eb_sm_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normF_eb_sm_aug2_a'.format(n_gaussian) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normF_eb_{}_aug5_0.2_2_a'.format(n_gaussian, MAX_BNUM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normFF_eb_{}_aug7_a_A5_fdim{}'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_one5'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_{}_hth_2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM, GBVS_R) # 1.0
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

# run = 'hd_gs_A{}_alt_r{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp'.format(n_gaussian, ALT_RATIO, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_aalt_all_2_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_aalt_all_4_{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp'.format(n_gaussian, ALT_RATIO, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_all_3_{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_sum_two_cw_sa_art_ftf_2_nob_mres_sp'.format(n_gaussian, ALT_RATIO, MAX_BNUM, FEATURE_DIM) # 1.0 
###run = 'hd_gs_A{}_all_5_{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_bms_cw_sa_art_ftf_2_nob_mres_sp'.format(n_gaussian, ALT_RATIO, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_val_4_{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_gbvs_cw_sa_art_ftf_2_nob_mres_sp'.format(n_gaussian, ALT_RATIO, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_aalt_val_4_{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_bms_cw_sa_art_ftf_2_nob_mres_sp'.format(n_gaussian, ALT_RATIO, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_aaalt_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_aalt_2_nob_mres_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_aalt_3_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_alt_3_nob_mres_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_alt_6_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_mres_5_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_all_8_a_A4_fdim{}_34_cw_sa_art_ftf_2_mres_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_all_5_proa_{}_a_A4_fdim{}_34_cw_sa_art_ftf_2_mres_sp'.format(n_gaussian, MAX_BNUM, PRO_RATIO,FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_2_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_alt_4_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_mres_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_alt_nob_mres_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_alt_nob_catX_2_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_art_ftf_2_catX_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_new_sp'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_new_ftf_2'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_new_fixf'.format(n_gaussian, MAX_BNUM, FEATURE_DIM) # 1.0 
# run = 'hd_gs_A{}_alt_self_gd_nf4_normT_eb_{}_aug7_a_A4_fdim{}_34_cw_sa_new_fixf'.format(n_gaussian, ALPHA, MAX_BNUM, FEATURE_DIM) # 1.0 
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
#  tensorboard --logdir=/raid/QX/WF/log --bind_all --port=6000 # will display all the subfolders within it
#  tensorboard --logdir=/research/dept2/qxlai/WF/log2/hd_gs_A16_all_5_1.05_gd_nf4_normT_eb_50_aug7_a_A4_fdim512_34_bms_cw_sa_art_ftf_2_nob_mres_sp --bind_all --port=6001 # will display all the subfolders within it
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


        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            # if torch.cuda.device_count() < 2:
            # # if True:
            #     if model.features[0].weight.grad is not None:
            #         writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
            #         writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
            #     writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)
            #
            #     if hasattr(model, 'relation_net'):
            #         writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
            #         writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
            #
            #         # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)
            #
            #     if hasattr(model, 'centerbias'):
            #         if hasattr(model.centerbias, 'fc1'):
            #             if model.centerbias.fc1.weight.grad is not None:
            #                 writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
            #                 writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
            #                 # writer.add_scalar('Grad_hd/gen_g_feature',
            #                 #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
            #         else:
            #             if model.centerbias.params.grad is not None:
            #                 writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
            #                 # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
            #                 #                   niter)
            #
            #     if hasattr(model, 'box_head'):
            #         if hasattr(model.box_head, 'fc6'):
            #             writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
            #             writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            #
            # else:
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



    # print("Train [{}]\tAverage cps_loss:{:.4f}"
    #       "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch,
    #                                              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
    #                                                                np.mean(np.array(total_map_loss))))
    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),np.mean(np.array(total_map_loss))

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
        rf_maps = torch.relu(rf_maps - torch.mean(rf_maps.view(rf_maps.size(0), -1), dim=-1, keepdim=True).unsqueeze(2)) # for gbvs_thm
        rf_maps = GBVS_R * rf_maps # for gbvs

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

def train_Wildcat_WK_hd_compf_map_cw_bms(epoch, model, optimizer, logits_loss, info_loss, dataloader, args):
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
        # rf_maps = GBVS_R * rf_maps # for gbvs
        rf_maps = torch.relu(rf_maps - torch.mean(rf_maps.view(rf_maps.size(0), -1), dim=-1, keepdim=True).unsqueeze(2)) # for bms_thm

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

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)
            #
            # if torch.cuda.device_count() < 2:
            #     if model.features[0].weight.grad is not None:
            #         writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
            #         writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
            #     writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)
            #
            #     if hasattr(model, 'relation_net'):
            #         writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
            #         writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
            #
            #         # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)
            #
            #     if hasattr(model, 'centerbias'):
            #         if hasattr(model.centerbias, 'fc1'):
            #             if model.centerbias.fc1.weight.grad is not None:
            #                 writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
            #                 writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
            #                 # writer.add_scalar('Grad_hd/gen_g_feature',
            #                 #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
            #         else:
            #             if model.centerbias.params.grad is not None:
            #                 writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
            #                 # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
            #                 #                   niter)
            #
            #     if hasattr(model, 'box_head'):
            #         writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
            #         writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            # else:
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

    # print("Train [{}]\tAverage cps_loss:{:.4f}"
    #       "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

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

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    # print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}\tAverage map_loss:{:.4f}".format(epoch,
    #           np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))
    #
    # return np.mean(np.array(total_cps_loss))+np.mean(np.array(total_h_loss)) , np.mean(np.array(total_map_loss)) # uncomment for hth_2_x
    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)) , np.mean(np.array(total_map_loss)) # uncomment for hth_2_x

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
        # MIT1003 & PASCAL-S image, boxes, sal_map, fix_map(, image_name) #
        ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X
        # MIT300 image, boxes(, image_name)
        # ori_inputs, ori_boxes, boxes_nums, img_name = X
        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()

        ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
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
        rf_loss = torch.nn.BCELoss()(torch.clamp((pred_maps_all/len(tgt_sizes)), min=0.0, max=1.0), sal_map)
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



        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            # if torch.cuda.device_count() < 2:
            # # if True:
            #     if model.features[0].weight.grad is not None:
            #         writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
            #         writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
            #     if model.classifier[0].weight.grad is not None:
            #         writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)
            #
            #     if hasattr(model, 'relation_net'):
            #         if model.relation_net.pair_pos_fc1.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
            #             writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
            #         if hasattr(model.relation_net, 'self_attention') and model.relation_net.self_attention.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/sa', model.relation_net.self_attention.weight.grad.abs().mean().item(), niter)
            #         # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)
            #
            #     if hasattr(model, 'centerbias'):
            #         if hasattr(model.centerbias, 'fc1'):
            #             if model.centerbias.fc1.weight.grad is not None:
            #                 writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
            #                 writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
            #                 # writer.add_scalar('Grad_hd/gen_g_feature',
            #                 #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
            #         else:
            #             if model.centerbias.params.grad is not None:
            #                 writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
            #                 # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
            #                 #                   niter)
            #
            #     if hasattr(model, 'box_head'):
            #         if model.box_head.fc6.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
            #             writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            #
            #     #if hasattr(model, 'self_attention'):
            #     #    writer.add_scalar('Grad_hd/sa', model.self_attention.weight.grad.abs().mean().item(), niter)
            #
            #
            # else:
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

    # print("Train [{}]\tAverage cps_loss:{:.4f}"
    #       "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
    #                                                                np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))

def train_Wildcat_WK_hd_compf_map_alt_alpha_sa(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, name_model):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
       checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
       # model_aux.load_state_dict(checkpoint['state_dict'])
       saved_state_dict = checkpoint['state_dict']
       new_params = model_aux.state_dict().copy()

       if list(saved_state_dict.keys())[0][:7] == 'module.':
           for k, y in saved_state_dict.items():
               if 'self_attention' not in k:
                   new_params[k[7:]] = y

       else:
           for k, y in saved_state_dict.items():
               if 'self_attention' not in k:
                   new_params[k] = y

       model_aux.load_state_dict(new_params)

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

        cps_logits, pred_maps, att_scores = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # if epoch > 0:
        _, aux_maps, _ = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # aux_maps = aux_maps - aux_maps.min()
        aux_maps = aux_maps - torch.min(torch.min(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # aux_maps = aux_maps * 1.1
        # print('aux_maps', aux_maps.size(), 'prior_maps', prior_maps.size())
        rf_maps = ALPHA*aux_maps.detach().squeeze() + (1-ALPHA)*prior_maps
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
                                                  torch.clamp(rf_maps, min=0.0, max=1.0))

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

        if i % tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            if torch.cuda.device_count() < 2:
                # if True:
                if model.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]',
                                      model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                if model.classifier[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model, 'relation_net'):
                    if model.relation_net.pair_pos_fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/pair_pos_fc1',
                                          model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/linear_out',
                                          model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
                    if hasattr(model.relation_net,
                               'self_attention') and model.relation_net.self_attention.weight.grad is not None:
                        writer.add_scalar('Grad_hd/sa',
                                          model.relation_net.self_attention.weight.grad.abs().mean().item(), niter)
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
                        writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(),
                                          niter)
                        writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(),
                                          niter)

                # if hasattr(model, 'self_attention'):
                #    writer.add_scalar('Grad_hd/sa', model.self_attention.weight.grad.abs().mean().item(), niter)


            else:
                if model.module.features[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(),
                                      niter)
                    writer.add_scalar('Grad_hd/f_layer4[-1]',
                                      model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
                if model.module.classifier[0].weight.grad is not None:
                    writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(),
                                      niter)
                # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
                # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

                if hasattr(model.module, 'relation_net'):
                    if model.module.relation_net.pair_pos_fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/pair_pos_fc1',
                                          model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/linear_out',
                                          model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
                    if hasattr(model.module.relation_net,
                               'self_attention') and model.module.relation_net.self_attention.weight.grad is not None:
                        writer.add_scalar('Grad_hd/sa',
                                          model.module.relation_net.self_attention.weight.grad.abs().mean().item(),
                                          niter)
                    # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

                if hasattr(model.module.centerbias, 'fc1'):
                    if model.module.centerbias.fc1.weight.grad is not None:
                        writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(),
                                          niter)
                        writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(),
                                          niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature',
                        #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
                else:
                    if model.module.centerbias.params.grad is not None:
                        writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(),
                                          niter)
                        # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                        #                   niter)

                if hasattr(model.module, 'box_head'):
                    if model.module.box_head.fc6.weight.grad is not None:
                        writer.add_scalar('Grad_hd/box_head_fc6',
                                          model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                        writer.add_scalar('Grad_hd/box_head_fc7',
                                          model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)

                # if hasattr(model.module, 'self_attention'):
                #    writer.add_scalar('Grad_hd/sa', model.module.self_attention.weight.grad.abs().mean().item(), niter)



    print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
          "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_loss)),
                                                 np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
                                                                   np.mean(np.array(total_map_loss))))

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

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    # print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}\tAverage map_loss:{:.4f}".format(epoch,
    #           np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))

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

        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            # writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            # writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            # *** for drawing the loss plots ***
            # writer.add_scalar('Train/L_cls', cps_losses.item(), niter)
            # writer.add_scalar('Train/L_prior', rf_losses.item() / rf_weight, niter)
            # writer.add_scalar('Train/L_info', h_losses.item() / hth_weight, niter)

            # if torch.cuda.device_count() < 2:
            # # if True:
            #     if model.features[0].weight.grad is not None:
            #         writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
            #         writer.add_scalar('Grad_hd/f_layer4[-1]', model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
            #     if model.classifier[0].weight.grad is not None:
            #         writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)
            #
            #     if hasattr(model, 'relation_net'):
            #         if model.relation_net.pair_pos_fc1.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/pair_pos_fc1', model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
            #             writer.add_scalar('Grad_hd/linear_out', model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
            #         if hasattr(model.relation_net, 'self_attention') and model.relation_net.self_attention.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/sa', model.relation_net.self_attention.weight.grad.abs().mean().item(), niter)
            #         # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)
            #
            #     if hasattr(model, 'centerbias'):
            #         if hasattr(model.centerbias, 'fc1'):
            #             if model.centerbias.fc1.weight.grad is not None:
            #                 writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
            #                 writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
            #                 # writer.add_scalar('Grad_hd/gen_g_feature',
            #                 #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
            #         else:
            #             if model.centerbias.params.grad is not None:
            #                 writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
            #                 # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
            #                 #                   niter)
            #
            #     if hasattr(model, 'box_head'):
            #         if model.box_head.fc6.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(), niter)
            #             writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(), niter)
            #
            #     #if hasattr(model, 'self_attention'):
            #     #    writer.add_scalar('Grad_hd/sa', model.self_attention.weight.grad.abs().mean().item(), niter)
            #
            # else:
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

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))
    # print("Train [{}]\tAverage cps_loss:{:.4f}"
    #       "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch,
    #                                              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
    #                                                                np.mean(np.array(total_map_loss))))

def train_Wildcat_WK_hd_compf_map_alt_alpha_sa_sp(epoch, model, model_aux, optimizer, logits_loss, info_loss, dataloader, args, path_models):
    model.train()

    N = len(dataloader)
    total_loss = list()
    total_h_loss = list()
    total_cps_loss = list()
    total_map_loss = list()

    # comment for sup mode
    # if os.path.exists(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))):
    #    checkpoint = torch.load(os.path.join(args.path_out,'{}_epoch{:02d}.pt'.format(name_model, epoch-1))) # checkpoint is a dict, containing much info
    #    # model_aux.load_state_dict(checkpoint['state_dict'])
    #    saved_state_dict = checkpoint['state_dict']
    #    new_params = model_aux.state_dict().copy()
    #
    #    if list(saved_state_dict.keys())[0][:7] == 'module.':
    #        for k, y in saved_state_dict.items():
    #            if 'self_attention' not in k:
    #                new_params[k[7:]] = y
    #
    #    else:
    #        for k, y in saved_state_dict.items():
    #            if 'self_attention' not in k:
    #                new_params[k] = y
    #
    #    model_aux.load_state_dict(new_params)
    # pdb.set_trace()
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

    # pdb.set_trace()

    # alt_ratio_current = ALT_RATIO**epoch
    # for i, X in enumerate(dataloader):
    bar = tqdm(dataloader)
    for i, X in enumerate(bar):
        optimizer.zero_grad()

        # COCO image, label, boxes(, image_name)
        # ILSVRC  image, label, boxes(, image_name)
        # inputs, gt_labels, box_features, boxes, boxes_nums, rf_maps = X
        # inputs, gt_labels, boxes, boxes_nums, _ = X
        inputs, gt_labels, boxes, boxes_nums, prior_maps = X

        prior_maps = torch.relu(prior_maps - torch.mean(prior_maps.view(prior_maps.size(0), -1), dim=-1, keepdim=True).unsqueeze(2))  # for bms_thm
        # prior_maps = GBVS_R * prior_maps # for gbvs

        if args.use_gpu:
            inputs = inputs.cuda()
            gt_labels = gt_labels.to(inputs.device)
            boxes = boxes.to(inputs.device)
            prior_maps = prior_maps.to(inputs.device)
        inputs, gt_labels, boxes, prior_maps = \
            torch.autograd.Variable(inputs), torch.autograd.Variable(gt_labels), \
            torch.autograd.Variable(boxes), torch.autograd.Variable(prior_maps)

        cps_logits, pred_maps, _, att_scores = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # if epoch > 0:
        _, aux_maps, _, _ = model_aux(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # aux_maps = aux_maps - aux_maps.min()
        aux_maps = aux_maps - torch.min(torch.min(aux_maps, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # aux_maps = aux_maps * ALT_RATIO # comment this for ftf_2_mres training from fixf
        # aux_maps = aux_maps * alt_ratio_current
        # print('aux_maps', aux_maps.size(), 'prior_maps', prior_maps.size())
        rf_maps = ALPHA*aux_maps.detach().squeeze() + (1-ALPHA)*prior_maps
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
                                                  torch.clamp(rf_maps, min=0.0, max=1.0))

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
            # print("Train [{}][{}/{}]\tcps_loss:{:.4f}({:.4f})"
            #       "\th_loss:{:.4f}({:.4f})\trf_loss:{:.4f}({:.4f})"
            #       "\tatt_scores:{:.4f}/{:.4f}/{:.4f}".format(
            #     epoch, i, int(N),
            #     cps_losses.item(), np.mean(np.array(total_cps_loss)),
            #     h_losses.item(), np.mean(np.array(total_h_loss)),
            #     rf_losses.item(), np.mean(np.array(total_map_loss)),
            #     att_scores.max().item(), att_scores.min().item(), torch.argmax(att_scores, dim=0).item()))
        bar.set_description("Train [{}][{}/{}] | cps:{:.4f}({:.4f})"
              " | h:{:.4f}({:.4f}) | rf:{:.4f}({:.4f})"
              " | att_scores:{:.4f}/{:.4f}/{:.4f}".format(
            epoch, i, int(N),
            cps_losses.item(), np.mean(np.array(total_cps_loss)),
            h_losses.item(), np.mean(np.array(total_h_loss)),
            rf_losses.item(), np.mean(np.array(total_map_loss)),
            att_scores.max().item(), att_scores.min().item(), torch.argmax(att_scores, dim=0).item()))


        if i % tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            # # writer.add_scalar('Train_hd/Loss', losses.item(), niter)
            # writer.add_scalar('Train_hd/Cps_loss', cps_losses.item(), niter)
            # writer.add_scalar('Train_hd/Rf_loss', rf_losses.item(), niter)

            writer.add_scalar('Train/L_cls', cps_losses.item(), niter)
            writer.add_scalar('Train/L_prior', rf_losses.item()/rf_weight, niter)
            writer.add_scalar('Train/L_info', h_losses.item()/hth_weight, niter)

            # if torch.cuda.device_count() < 2:
            #     # if True:
            #     if model.features[0].weight.grad is not None:
            #         writer.add_scalar('Grad_hd/features0', model.features[0].weight.grad.abs().mean().item(), niter)
            #         writer.add_scalar('Grad_hd/f_layer4[-1]',
            #                           model.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
            #     if model.classifier[0].weight.grad is not None:
            #         writer.add_scalar('Grad_hd/classifier0', model.classifier[0].weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
            #     # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)
            #
            #     if hasattr(model, 'relation_net'):
            #         if model.relation_net.pair_pos_fc1.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/pair_pos_fc1',
            #                               model.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
            #             writer.add_scalar('Grad_hd/linear_out',
            #                               model.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
            #         if hasattr(model.relation_net,
            #                    'self_attention') and model.relation_net.self_attention.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/sa',
            #                               model.relation_net.self_attention.weight.grad.abs().mean().item(), niter)
            #         # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)
            #
            #     if hasattr(model.centerbias, 'fc1'):
            #         if model.centerbias.fc1.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/gs_fc1', model.centerbias.fc1.weight.grad.abs().mean().item(), niter)
            #             writer.add_scalar('Grad_hd/gs_fc2', model.centerbias.fc2.weight.grad.abs().mean().item(), niter)
            #             # writer.add_scalar('Grad_hd/gen_g_feature',
            #             #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
            #     else:
            #         if model.centerbias.params.grad is not None:
            #             writer.add_scalar('Grad_hd/gs_params', model.centerbias.params.grad.abs().mean().item(), niter)
            #             # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
            #             #                   niter)
            #
            #     if hasattr(model, 'box_head'):
            #         if model.box_head.fc6.weight.grad is not None:
            #             writer.add_scalar('Grad_hd/box_head_fc6', model.box_head.fc6.weight.grad.abs().mean().item(),
            #                               niter)
            #             writer.add_scalar('Grad_hd/box_head_fc7', model.box_head.fc7.weight.grad.abs().mean().item(),
            #                               niter)
            #
            #     # if hasattr(model, 'self_attention'):
            #     #    writer.add_scalar('Grad_hd/sa', model.self_attention.weight.grad.abs().mean().item(), niter)


            # else:
            if model.module.features[0].weight.grad is not None:
                writer.add_scalar('Grad_hd/features0', model.module.features[0].weight.grad.abs().mean().item(),
                                  niter)
                writer.add_scalar('Grad_hd/f_layer4[-1]',
                                  model.module.features[-1][-1].conv3.weight.grad.abs().mean().item(), niter)
            if model.module.classifier[0].weight.grad is not None:
                writer.add_scalar('Grad_hd/classifier0', model.module.classifier[0].weight.grad.abs().mean().item(),
                                  niter)
            # writer.add_scalar('Grad_hd/rn_lin1', model.relation_net.lin1.weight.grad.abs().mean().item(), niter)
            # writer.add_scalar('Grad_hd/rn_lin3', model.relation_net.lin3.weight.grad.abs().mean().item(), niter)

            if hasattr(model.module, 'relation_net'):
                if model.module.relation_net.pair_pos_fc1.weight.grad is not None:
                    writer.add_scalar('Grad_hd/pair_pos_fc1',
                                      model.module.relation_net.pair_pos_fc1.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/linear_out',
                                      model.module.relation_net.linear_out.weight.grad.abs().mean().item(), niter)
                if hasattr(model.module.relation_net,
                           'self_attention') and model.module.relation_net.self_attention.weight.grad is not None:
                    writer.add_scalar('Grad_hd/sa',
                                      model.module.relation_net.self_attention.weight.grad.abs().mean().item(),
                                      niter)
                # writer.add_histogram('Hist_hd/linear_out', model.relation_net.linear_out.weight.detach().cpu().numpy(), niter)

            if hasattr(model.module.centerbias, 'fc1'):
                if model.module.centerbias.fc1.weight.grad is not None:
                    writer.add_scalar('Grad_hd/gs_fc1', model.module.centerbias.fc1.weight.grad.abs().mean().item(),
                                      niter)
                    writer.add_scalar('Grad_hd/gs_fc2', model.module.centerbias.fc2.weight.grad.abs().mean().item(),
                                      niter)
                    # writer.add_scalar('Grad_hd/gen_g_feature',
                    #                   model.gen_g_feature.weight.grad.abs().mean().item(), niter)
            else:
                if model.module.centerbias.params.grad is not None:
                    writer.add_scalar('Grad_hd/gs_params', model.module.centerbias.params.grad.abs().mean().item(),
                                      niter)
                    # writer.add_scalar('Grad_hd/gen_g_feature', model.gen_g_feature.weight.grad.abs().mean().item(),
                    #                   niter)

            if hasattr(model.module, 'box_head'):
                if model.module.box_head.fc6.weight.grad is not None:
                    writer.add_scalar('Grad_hd/box_head_fc6',
                                      model.module.box_head.fc6.weight.grad.abs().mean().item(), niter)
                    writer.add_scalar('Grad_hd/box_head_fc7',
                                      model.module.box_head.fc7.weight.grad.abs().mean().item(), niter)

            # if hasattr(model.module, 'self_attention'):
            #    writer.add_scalar('Grad_hd/sa', model.module.self_attention.weight.grad.abs().mean().item(), niter)



    # print("Train [{}]\tAverage loss:{:.4f}\tAverage cps_loss:{:.4f}"
    #       "\tAverage h_loss:{:.4f}\tAverage rf_loss:{:.4f}".format(epoch, np.mean(np.array(total_cps_loss)),
    #                                              np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
    #                                                                np.mean(np.array(total_map_loss))))
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


        if i%tb_log_interval == 0:
            niter = epoch * len(dataloader) + i
            writer.add_scalar('Eval_hd/Cps_loss', cps_losses.item(), niter)
            writer.add_scalar('Eval_hd/Map_loss', map_losses.item(), niter)

    # print("Eval [{}]\tAverage cps_loss:{:.4f}\tAverage h_loss:{:.4f}"
    #       "\tAverage map_loss:{:.4f}\tAverage obj_map_loss:{:.4f}".format(epoch,
    #           np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)),
    #         np.mean(np.array(total_map_loss)), np.mean(np.array(total_obj_map_loss))))

    return np.mean(np.array(total_cps_loss)), np.mean(np.array(total_h_loss)), np.mean(np.array(total_map_loss))

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

        cps_logits, pred_maps, att_maps, att_scores = model(img=inputs,
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

        cps_logits, pred_maps, att_maps, att_scores = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)

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
    att_folder = os.path.join(args.path_out, folder_name, best_model_file+postfix)
    if not os.path.exists(att_folder):
        os.makedirs(att_folder)
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
        _, pred_maps, att_maps, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
        # pred_maps = torch.nn.Sigmoid()(pred_maps)
        print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())

        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction_salgan(pred_maps.squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
                          postprocess_prediction(pred_maps.squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        scipy.misc.imsave(os.path.join(att_folder, img_name[0]+'.png'),
                          postprocess_prediction_my(att_maps.squeeze().detach().cpu().numpy(),
                                                    shape_r=ori_img.shape[0],
                                                    shape_c=ori_img.shape[1])) # the ratio is not right..


# https://github.com/rdroste/unisal/blob/192abd2affbb1824895118a914806663bb897aa1/unisal/train.py#L710
# https://github.com/rdroste/unisal/blob/192abd2affbb1824895118a914806663bb897aa1/unisal/train.py#L607
# SIM, AUC-J, s-AUC, https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py
# KLD, NSS, CC, https://github.com/rdroste/unisal/blob/master/unisal/utils.py
# from load_data import fixationProcessing
def test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, dataloader, args, tgt_sizes,
                                                 metrics=('kld', 'nss', 'cc', 'sim', 'aucj', 'aucs')):
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

    # pdb.set_trace()
    if args.use_gpu:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_multiscale')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    N = len(dataloader) // args.batch_size
    for i, X in enumerate(dataloader):
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

            _, pred_maps, _, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
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

def save_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, dataloader, args, tgt_sizes,
                                                 metrics=('kld', 'nss', 'cc', 'sim', 'aucj', 'aucs')):
    # if best_model_file != 'no_training':
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
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
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

# TODO: pending test for aucs
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


def test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp_rank(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
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

    # out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_img')
    # out_folder_om = os.path.join(args.path_out, folder_name, best_model_file + '_om')  # object mask
    # out_folder_gs = os.path.join(args.path_out, folder_name, best_model_file+'_gs') # gaussian bias
    # out_folder_sc = os.path.join(args.path_out, folder_name, best_model_file+'_sc') # attention score
    # out_folder_pd = os.path.join(args.path_out, folder_name, best_model_file+'_pd') # pred score
    out_folder_cw = os.path.join(args.path_out, folder_name, best_model_file+'_cw') # pred score


    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # if not os.path.exists(out_folder_om):
    #     os.makedirs(out_folder_om)
    # if not os.path.exists(out_folder_gs):
    #     os.makedirs(out_folder_gs)
    # if not os.path.exists(out_folder_sc):
    #     os.makedirs(out_folder_sc)
    # if not os.path.exists(out_folder_pd):
    #     os.makedirs(out_folder_pd)
    if not os.path.exists(out_folder_cw):
        os.makedirs(out_folder_cw)

    N = len(dataloader) // args.batch_size
    evals = list()
    img_names = list()
    for i, X in enumerate(dataloader):
        # MIT1003 & PASCAL-S image, boxes, sal_map, fix_map(, image_name)
        # ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X

        # SALICON image, label, boxes, sal_map, fix_map(, image_name)
        ori_inputs, label, ori_boxes, boxes_nums, sal_maps, _, img_name = X

        # MIT300 & SALICON test image, boxes(, image_name)
        # ori_inputs, ori_boxes, boxes_nums, img_name = X

        img_names.append(img_name[0])

        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()
            label = label.cuda()
            sal_maps = sal_maps.cuda()

        # ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_MIT300, 'images', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'val', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'test', img_name[0]+'.jpg')) # height, width, channel
        ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'train', img_name[0]+'.jpg')) # height, width, channel

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

            if tgt_s==input_w:
                # pred_logits, pred_maps, obj_masks, att_scores, gaussians = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
                _, pred_maps, _, _, _, cw_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            else:
                # _, pred_maps, _, _, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
                _, pred_maps, _, _, _, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        # print(pred_maps_all.squeeze().size())
        rf_loss = torch.nn.BCELoss()(torch.clamp((pred_maps_all/len(tgt_sizes)), min=0.0, max=1.0), sal_maps)
        evals.append(rf_loss.item())

        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder_om, img_name[0]+'.png'),
        #                   postprocess_prediction(obj_masks.squeeze().detach().cpu().numpy(), size=[ori_img.shape[0], ori_img.shape[1]]))
        # np.save(os.path.join(out_folder_gs, img_name[0]+'.npy'), gaussians.squeeze().detach().cpu().numpy())
        # np.save(os.path.join(out_folder_pd, img_name[0]+'.npy'), pred_logits.squeeze().detach().cpu().numpy())
        # np.save(os.path.join(out_folder_pd, img_name[0]+'_label.npy'), label.squeeze().detach().cpu().numpy())
        # np.save(os.path.join(out_folder_sc, img_name[0]+'.npy'), att_scores.squeeze().detach().cpu().numpy())
        np.save(os.path.join(out_folder_cw, img_name[0]+'.npy'), cw_maps.squeeze().detach().cpu().numpy())
        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction_salgan((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
        #                                             size=[ori_img.shape[0], ori_img.shape[1]])) # the ratio is not right..
    # inds = np.argsort(np.array(evals))
    # img_names = np.array(img_names)
    # img_names_sorted = img_names[inds]
    # lists = [line + '\n' for line in img_names_sorted]
    # with open(os.path.join(args.path_out, folder_name, best_model_file + '.txt'), 'w') as f:
    #     f.writelines(lists)

def test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp_rank_rebuttal(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
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

    # out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_img')
    out_folder = os.path.join(args.path_out, folder_name, best_model_file)
    out_folder_om = os.path.join(args.path_out, folder_name, best_model_file + '_om')  # object mask
    # out_folder_gs = os.path.join(args.path_out, folder_name, best_model_file+'_gs') # gaussian bias
    # out_folder_gs = os.path.join(args.path_out, folder_name, best_model_file+'_gs_v2') # gaussian bias; cw weights
    out_folder_gs = os.path.join(args.path_out, folder_name, best_model_file+'_gs_v3') # gaussian bias; cw&bbox weights
    # out_folder_sc = os.path.join(args.path_out, folder_name, best_model_file+'_sc') # attention score
    # out_folder_pd = os.path.join(args.path_out, folder_name, best_model_file+'_pd') # pred score
    out_folder_cw = os.path.join(args.path_out, folder_name, best_model_file+'_cw') # sementic maps


    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(out_folder_om):
        os.makedirs(out_folder_om)
    if not os.path.exists(out_folder_gs):
        os.makedirs(out_folder_gs)
    # if not os.path.exists(out_folder_sc):
    #     os.makedirs(out_folder_sc)
    # if not os.path.exists(out_folder_pd):
    #     os.makedirs(out_folder_pd)
    if not os.path.exists(out_folder_cw):
        os.makedirs(out_folder_cw)

    N = len(dataloader) // args.batch_size
    evals = list()
    img_names = list()
    # for i, X in enumerate(dataloader):
    for i, X in enumerate(tqdm(dataloader)):
        # MIT1003 & PASCAL-S image, boxes, sal_map, fix_map(, image_name)
        ori_inputs, ori_boxes, boxes_nums, sal_maps, _, img_name = X

        # # SALICON image, label, boxes, sal_map, fix_map(, image_name)
        # ori_inputs, label, ori_boxes, boxes_nums, sal_maps, _, img_name = X

        # MIT300 & SALICON test image, boxes(, image_name)
        # ori_inputs, ori_boxes, boxes_nums, img_name = X

        img_names.append(img_name[0])

        if args.use_gpu:
            ori_inputs = ori_inputs.cuda()
            ori_boxes = ori_boxes.cuda()
            # label = label.cuda()
            sal_maps = sal_maps.cuda()

        ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_MIT300, 'images', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'val', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'test', img_name[0]+'.jpg')) # height, width, channel
        # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'train', img_name[0]+'.jpg')) # height, width, channel

        ori_size = ori_inputs.size(-1)
        pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
        for tgt_s in tgt_sizes:
            inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
            boxes = torch.zeros_like(ori_boxes)
            boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
            boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
            boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
            boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

            if tgt_s==input_w:
                # pred_logits, pred_maps, obj_masks, att_scores, gaussians = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
                # pred_comp_logits, sal_map, obj_att_maps, att_scores, gaussian_maps, cw_maps_return
                _, pred_maps, obj_maps, _, gaus_maps, cw_maps = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            else:
                # _, pred_maps, _, _, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
                _, pred_maps, _, _, _, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
            # pred_maps = torch.nn.Sigmoid()(pred_maps)
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
            pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

        # print(pred_maps_all.squeeze().size())
        rf_loss = torch.nn.BCELoss()(torch.clamp((pred_maps_all/len(tgt_sizes)), min=0.0, max=1.0), sal_maps)
        evals.append(rf_loss.item())
        # pdb.set_trace() # batch size should be 1 ...
        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
        #                                          size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder_om, img_name[0]+'.png'),
        #                   postprocess_prediction(obj_maps.squeeze().detach().cpu().numpy(), size=[ori_img.shape[0], ori_img.shape[1]]))
        scipy.misc.imsave(os.path.join(out_folder_gs, img_name[0]+'.png'),
                          postprocess_prediction(gaus_maps.squeeze().detach().cpu().numpy(),
                                                 size=[ori_img.shape[0], ori_img.shape[1]]))
        # scipy.misc.imsave(os.path.join(out_folder_cw, img_name[0]+'.png'),
        #                   postprocess_prediction(cw_maps.squeeze().detach().cpu().numpy(), size=[ori_img.shape[0], ori_img.shape[1]]))
        # np.save(os.path.join(out_folder_gs, img_name[0]+'.npy'), gaussians.squeeze().detach().cpu().numpy())
        # np.save(os.path.join(out_folder_pd, img_name[0]+'.npy'), pred_logits.squeeze().detach().cpu().numpy())
        # np.save(os.path.join(out_folder_pd, img_name[0]+'_label.npy'), label.squeeze().detach().cpu().numpy())
        # np.save(os.path.join(out_folder_sc, img_name[0]+'.npy'), att_scores.squeeze().detach().cpu().numpy())
        # np.save(os.path.join(out_folder_cw, img_name[0]+'.npy'), cw_maps.squeeze().detach().cpu().numpy())
        # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
        #                   postprocess_prediction_salgan((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
        #                                             size=[ori_img.shape[0], ori_img.shape[1]])) # the ratio is not right..
    inds = np.argsort(np.array(evals))
    img_names = np.array(img_names)
    img_names_sorted = img_names[inds]
    lists = [line + '\n' for line in img_names_sorted]
    with open(os.path.join(args.path_out, folder_name, best_model_file + '.txt'), 'w') as f:
        f.writelines(lists)

def test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp_gs(model, folder_name, best_model_file, dataloader, args, tgt_sizes):
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

    # out_folder = os.path.join(args.path_out, folder_name, best_model_file+'_img')
    # out_folder_om = os.path.join(args.path_out, folder_name, best_model_file + '_om')  # object mask
    # out_folder_gs = os.path.join(args.path_out, folder_name, best_model_file+'_gs') # gaussian bias
    # out_folder_sc = os.path.join(args.path_out, folder_name, best_model_file+'_sc') # attention score
    # out_folder_pd = os.path.join(args.path_out, folder_name, best_model_file+'_pd') # pred score


    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # if not os.path.exists(out_folder_om):
    #     os.makedirs(out_folder_om)
    # if not os.path.exists(out_folder_gs):
    #     os.makedirs(out_folder_gs)
    # if not os.path.exists(out_folder_sc):
    #     os.makedirs(out_folder_sc)
    # if not os.path.exists(out_folder_pd):
    #     os.makedirs(out_folder_pd)

    # N = len(dataloader) // args.batch_size
    # evals = list()
    # img_names = list()
    for tidx in range(10):
        out_folder_gs = os.path.join(args.path_out, folder_name, best_model_file + '_gs_%02d'%tidx)  # gaussian bias
        if not os.path.exists(out_folder_gs):
            os.makedirs(out_folder_gs)

        for i, X in enumerate(dataloader):
            # MIT1003 & PASCAL-S image, boxes, sal_map, fix_map(, image_name)
            # ori_inputs, ori_boxes, boxes_nums, _, _, img_name = X

            # SALICON image, label, boxes, sal_map, fix_map(, image_name)
            # ori_inputs, label, ori_boxes, boxes_nums, sal_maps, _, img_name = X
            ori_inputs, _, ori_boxes, boxes_nums, _, _, img_name = X

            # MIT300 & SALICON test image, boxes(, image_name)
            # ori_inputs, ori_boxes, boxes_nums, img_name = X

            # img_names.append(img_name[0])

            if args.use_gpu:
                ori_inputs = ori_inputs.cuda()
                ori_boxes = ori_boxes.cuda()
                # label = label.cuda()
                # sal_maps = sal_maps.cuda()

            # ori_img = scipy.misc.imread(os.path.join(PATH_MIT1003, 'ALLSTIMULI', img_name[0] + '.jpeg'))  # height, width, channel
            # ori_img = scipy.misc.imread(os.path.join(PATH_PASCAL, 'images', img_name[0]+'.jpg')) # height, width, channel
            # ori_img = scipy.misc.imread(os.path.join(PATH_MIT300, 'images', img_name[0]+'.jpg')) # height, width, channel
            # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'val', img_name[0]+'.jpg')) # height, width, channel
            # ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'test', img_name[0]+'.jpg')) # height, width, channel
            ori_img = scipy.misc.imread(os.path.join(PATH_SALICON, 'images', 'train', img_name[0]+'.jpg')) # height, width, channel

            ori_size = ori_inputs.size(-1)
            pred_maps_all = torch.zeros(ori_inputs.size(0), 1, output_h, output_w).to(ori_inputs.device)
            for tgt_s in tgt_sizes:
                inputs = F.interpolate(ori_inputs, size=(tgt_s, tgt_s), mode='bilinear', align_corners=True)
                boxes = torch.zeros_like(ori_boxes)
                boxes[:, :, 0] = ori_boxes[:, :, 0] / ori_size*tgt_s
                boxes[:, :, 2] = ori_boxes[:, :, 2] / ori_size*tgt_s
                boxes[:, :, 1] = ori_boxes[:, :, 1] / ori_size*tgt_s
                boxes[:, :, 3] = ori_boxes[:, :, 3] / ori_size*tgt_s

                if tgt_s==input_w:
                    pred_logits, pred_maps, obj_masks, att_scores, gaussians = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
                else:
                    _, pred_maps, _, _, _ = model(img=inputs, boxes=boxes, boxes_nums=boxes_nums)
                # pred_maps = torch.nn.Sigmoid()(pred_maps)
                # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
                # print(pred_maps.squeeze(1).size(), HLoss_th()(pred_maps.squeeze(1)).item())
                pred_maps_all += F.interpolate(pred_maps, size=(output_h, output_w), mode='bilinear', align_corners=True)

            # print(pred_maps_all.squeeze().size())
            np.save(os.path.join(out_folder_gs, img_name[0]+'_params.npy'), gaussians.squeeze().detach().cpu().numpy())
            # gauss = gaussians.squeeze().detach().cpu().numpy()
            # for g in range(gauss.shape[0]):
            #     scipy.misc.imsave(os.path.join(out_folder_gs, '%s_%02d.png' % (img_name[0], g)),
            #            scipy.misc.imresize(gauss[g, :, :], (56, 56)))

            # scipy.misc.imsave(os.path.join(out_folder, img_name[0]+'.png'),
            #                   postprocess_prediction_salgan((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
            #                                             size=[ori_img.shape[0], ori_img.shape[1]])) # the ratio is not right..


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
        losses = logits_loss(pred_logits, gt_labels)  # use bce loss with sigmoid
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
                          postprocess_prediction_thm((pred_maps_all/len(tgt_sizes)).squeeze().detach().cpu().numpy(),
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
    path_models = os.path.join(args.path_out, 'Models', args.model_name)
    if not os.path.exists(path_models):
        os.makedirs(path_models)

    # phase = 'test_cw_multiscale' # noobj
    # phase = 'test'
    # phase = 'test_cw'
    # phase = 'test_cw_sa'
    # phase = 'test_cw_sa_multiscale' # for _sa_art, without object mask
    # phase = 'test_cw_sa_sp_multiscale' # ***
    # phase = 'test_cw_sa_sp_multiscale_rank'
    # phase = 'test_cw_sa_sp'

    # phase = 'train_cw_aug'    ### base model
    # phase = 'train_cw_aug_gbvs' ### base model with gbvs and bms, other priors
    # phase = 'train_cw_alt_alpha' ### obtain f
    # phase = 'train_cw_aug_sa_new'
    # phase = 'train_cw_aug_sa_art' ### obtain fixf
    # phase = 'train_alt_alpha_sa_new'
    # phase = 'train_cw_aug_sa'
    # phase = 'train_cw_aug_sa_sp_fixf' ### sa_new_sp, sa_art_sp, obtain fixf_sp
    # phase = 'train_cw_aug_sa_sp' ### sa_new_sp, sa_art_sp, obtain ftf_2
    # phase = 'train_all_cw_aug_sa_sp' ### train model with the whole MS_COCO
    # phase = 'train_cw_aug_alt_alpha_sa_sp' ### obtain alt_ftf_2, and ftf_2_mres with grad
    # phase = 'train_aug' # for rf==0, because it is hard to train w/o pre_cls_loss
    # phase = 'test_cw_sa_sp_multiscale_rank_rebuttal' ### for generating outputs of submodules

    phase = args.phase

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

    # *** 210427 ***
    if phase == 'train_cw_aug':
        print('lr %.4f'%args.lr)


        # prior='gbvs'
        prior='nips08'

        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_noobj(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) # global center bias

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) # global center bias

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nomlp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # # #
        '''model_name'''
        # # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_0.1_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_proa_{}_aug7_{}_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, PRO_RATIO, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_2_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noobj_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_2_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_all_3_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_all_8_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # revisit other augmentations
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_3_{}_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        #
        # # no mlp as box head; use avg pool
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_nomlp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_nomlp_sigmoid_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # previsou one5 is actually one2 ... sad ...
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # previsou one5 is actually one2 ... sad ...
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # prior = 'nips08'
        # #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # print(model_name)
        '''optimizer'''
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)
        #
        # # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
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
        # logits_loss = torch.nn.BCELoss()
        # h_loss = HLoss_th() #####
        h_loss = HLoss_th_2() #210427
        # h_loss = HLoss_th_2()
        # h_loss = HLoss_th_3()
        # h_loss = HLoss()


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

    elif phase == 'train_cw_aug_gbvs':
        print('lr %.4f'%args.lr)


        prior='gbvs'
        # prior='bms'
        # prior='nips08'

        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nomlp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # # #
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_thm_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_hth_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, BMS_R, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_2_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, GBVS_R, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_thm_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
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
        # h_loss = HLoss_th_3()
        h_loss = HLoss_th_2() # final loss in the article
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
            # train_Wildcat_WK_hd_compf_map_cw_bms(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss, _ = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
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

    # *** 210511 ***
    elif phase == 'train_cw_aug_sa_art':
        print('lr %.4f'%args.lr)

        prior='nips08'
        # prior='bms'
        # prior='gbvs'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=normf) # noGrid

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################


        '''init model'''
        # # finetune init =====================================
        # if ATT_RES:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                      '_hth0.1_ms4_fdim512_34_cw_sa_new_fix_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # else:
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #     #                                      '_hth0.1_ms4_fdim512_34_cw_sa_new_fix_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
        #     #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                          'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                          '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
        #                             map_location='cuda:0')
        #
        #
        #
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

        # # init fixf (trained using train_cw_aug/train_cw_aug_gbvs)---------------------------------------------
        # # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08'+
        # #                                      '_rf0.1_hth0.1_ms4_fdim512_34_cw_3_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch06.pt'),
        # #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_alt_3_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_bms_thm_rf0.1' +
        # #                                      '_hth0.1_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'), map_location='cuda:0')
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_alt_2_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_gbvs_0.5_thm_rf0.1'+
        #                                      '_hth0.1_2_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch11.pt'), map_location='cuda:0')
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.0_hth0.0'+
        #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04.pt'), map_location='cuda:0')
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.0_hth0.1'+
        #                                      '_twocls_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'), map_location='cuda:0')
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.0'+
        #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'), map_location='cuda:0')

        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbG16_alt_5_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1'+
        #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'), map_location='cuda:0')
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_nobs_alt_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1'+
        #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'), map_location='cuda:0')
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_norn_2_rf0.1_hth0.1'+
        #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04.pt'), map_location='cuda:0')
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet101_wildcat_wk_hd_cbA16_alt_2_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_noGrid_2_rf0.1_hth0.1'+
        #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06.pt'), map_location='cuda:0')
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_nopsal_rf0.1_hth0.1'+
        #                                      '_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch09.pt'), map_location='cuda:0')

        # **** last one *****
        # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_alt_2_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_gbvs_0.5_thm_rf0.1'+
        #                                      '_hth0.1_2_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch11.pt'), map_location='cuda:0')

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
        nss_value = checkpoint['nss']
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
        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        # h_loss = HLoss_th()
        h_loss = HLoss_th_2()
        # h_loss = HLoss()

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


    elif phase == 'train_alt_alpha_sa_new':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_lstm_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin,
                                                              alpha=alpha,
                                                              num_maps=num_maps,
                                                              fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                              normalize_feature=normf)
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
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_self_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_fixf_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, ALPHA, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        else:
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_self_{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_fixf_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, ALPHA, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        # if ATT_RES:
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_ft_3_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # else:
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_ft_3_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # previsou one5 is actually one2 ... sad ...
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

        # # finetuen init ---------------------------------------------
        # if ATT_RES:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                      '_hth0.1_ms4_fdim512_34_cw_sa_new_fix_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # else:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                          'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                          '_hth0.1_ms4_fdim512_34_cw_sa_new_fix_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
        #                             map_location='cuda:0')  # checkpoint is a dict, containing much info
        #
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

        # init with final model ---------------------------------------------
        checkpoint = torch.load(os.path.join(args.path_out, 'Models',
                                             'resnet50_wildcat_wk_hd_cbA16_alt2_2_0.95_compf_cls_att_gd_nf4_normTrue_hb_50_aug7' +
                                             '_nips08_rf0.1_hth0.1_ms4_fdim512_34_lstm_cw_1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09.pt'),
                                map_location='cuda:0')  # checkpoint is a dict, containing much info


        saved_state_dict = checkpoint['state_dict']
        new_params = model.state_dict().copy()

        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                if 'feature_refine' not in k:
                    new_params[k[7:]] = y

        else:
            for k, y in saved_state_dict.items():
                if 'feature_refine' not in k:
                    new_params[k] = y

        model.load_state_dict(new_params)

        # -------- model max ---------
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
        # fix -------------------------------------------------------
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

        # -------------------------------------------------

        if args.use_gpu:
            model.cuda()
            model_aux.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)
            model_aux = torch.nn.DataParallel(model)
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
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-7) ###for finetuning, best 1e-6
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        # print('real learning rate 1e-7.')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_alt_alpha_sa(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                              train_dataloader, args, model_name)

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

    # *** 210512 ***
    elif phase == 'train_cw_aug_sa_sp_fixf':
        print('lr %.4f'%args.lr)


        # prior='bms'
        # prior='gbvs'
        prior='nips08'

        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_catX(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=normf) # noGrid

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #


        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        '''model_name'''
        # # # # #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # revisit other augmentations
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_3_{}_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # if ATT_RES:
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # else:
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_catX_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #
        #     # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #     # model_name = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #     model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #         n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #     # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #     # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #
        #
        # # if ATT_RES:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # else:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_nob_fixf_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # # if ATT_RES:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # else:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # # previsou one5 is actually one2 ... sad ...
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # previsou one5 is actually one2 ... sad ...
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # prior = 'nips08'
        # #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # print(model_name)

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
        nss_value = checkpoint['nss']
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
        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th_2()
        # h_loss = HLoss_th()
        # h_loss = HLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) ######################
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

        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_catX(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=normf) ##noGrid

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # # # #
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
        '''model_name'''
        # if ATT_RES:
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # else:
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_catX_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # --------------------------
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_4_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #         n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_4_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_0123'.format(
        #         n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_4_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_0123_4'.format(
        #         n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #
        #     # model_name = '210423'
        #     model_name = run
        #
        #     # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #     # model_name = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #     # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #     # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_ftf_2_4_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #     # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #
        # # if ATT_RES:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # else:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_nob_fixf_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # # if ATT_RES:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # else:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # # previsou one5 is actually one2 ... sad ...
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # previsou one5 is actually one2 ... sad ...
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # prior = 'nips08'
        # #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # print(model_name)

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


        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        # h_loss = HLoss_th_210423()
        h_loss = HLoss_th_2()  # *** default
        # h_loss = HLoss_th()
        # h_loss = HLoss()

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

    elif phase == 'train_all_cw_aug_sa_sp':
        print('lr %.4f'%args.lr)


        # prior='sum_two'
        # prior='gbvs'
        prior='bms'
        # prior='nips08'

        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_catX(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # # # #
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
            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                                        n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

        else:
            # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_catX_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
            # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
            # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_5_proa_{}_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #     n_gaussian, normf, MAX_BNUM, PRO_RATIO, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

            # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_9_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

            model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_5_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
                n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

            # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
            # ===============================
            # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

            # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #                             n_gaussian, normf, MAX_BNUM, prior, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all

            # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)

            # model_name = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
            # model_name = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
            # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
            # model_name = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
            #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all

        # if ATT_RES:
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # else:
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_nob_fixf_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # if ATT_RES:
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # else:
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # previsou one5 is actually one2 ... sad ...
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

        # # init
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
        # fine tune final model ====================================
        if ATT_RES:
            checkpoint = torch.load(os.path.join(args.path_out, 'Models',
                                             'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
                                             '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01.pt'),
                                map_location='cuda:0')  # checkpoint is a dict, containing much info
        else:
            # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
            #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
            #                                  '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
            #                     map_location='cuda:0')  # checkpoint is a dict, containing much info

            # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
            #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_bms_thm_rf0.1_hth0.1'+
            #                                  '_ms4_fdim512_34_cw_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
            #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
            #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_gbvs_0.5_thm_rf0.1_hth0.1'+
            #                                  '_ms4_fdim512_34_cw_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
            #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
            #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.0'+
            #                                  '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
            #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
            #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.0_hth0.1'+
            #                                  '_ms4_sa_art_fixf_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
            #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
            # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
            #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
            #                                  '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
            #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
            checkpoint = torch.load(os.path.join(args.path_out, 'Models',
                                                 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1' +
                                                 '_ms4_fdim512_34_cw_sa_art_ftf_2_mres_2_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
                                    map_location='cuda:0')  # checkpoint is a dict, containing much info



        saved_state_dict = checkpoint['state_dict']
        new_params = model.state_dict().copy()

        if list(saved_state_dict.keys())[0][:7] == 'module.':
            for k, y in saved_state_dict.items():
                if 'feature_refine' not in k:
                    # if 'gen_g_feature.weight' in k:
                    #     #pdb.set_trace()
                    #     new_params[k[7:]] = torch.cat([y, torch.zeros_like(y[:,-1:,:,:])], dim=1)
                    # else:
                    #     new_params[k[7:]] = y
                    new_params[k[7:]] = y
        else:
            for k, y in saved_state_dict.items():
                if 'feature_refine' not in k:
                    # if 'gen_g_feature.weight' in k:
                    #     #0pdb.set_trace()
                    #     new_params[k] = torch.cat([y, torch.zeros_like(y[:,-1:,:,:])], dim=1)
                    # else:
                    #     new_params[k] = y
                    new_params[k] = y

        model.load_state_dict(new_params)

        # # init with sa_art_fixf models ====================================
        # if ATT_RES:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                      '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        #
        # else:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                      '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
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
        #     model.module.relation_net.self_attention.weight.requires_grad = True
        #     model.module.relation_net.self_attention.bias.requires_grad = True
        # else:
        #     model.relation_net.self_attention.weight.requires_grad = True
        #     model.relation_net.self_attention.bias.requires_grad = True
        #
        # # -----------------------------------------


        if args.use_gpu:
            model.cuda()
        if torch.cuda.device_count()>1:
            model = torch.nn.DataParallel(model)

        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        # ds_train = MS_COCO_ALL_map_full_aug_prior(mode='all', img_h=input_h, img_w=input_w, prior=prior)
        ds_train = MS_COCO_ALL_map_full_aug(mode='all', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='val', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
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
        h_loss = HLoss_th_2()
        # h_loss = HLoss_th()
        # h_loss = HLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) ######################
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('finetune lr rate: 1e-5')
        # print('relation lr factor: 1.0')

        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        # eval_salicon_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        for i_epoch in range(args.n_epochs):
            train_Wildcat_WK_hd_compf_map_cw_sa_sp(i_epoch, model, optimizer, logits_loss, h_loss, train_dataloader, args)

            tmp_eval_loss, map_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            if tmp_eval_loss < eval_loss:
                cnt = 0
                eval_loss = tmp_eval_loss
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            elif tmp_eval_loss < 0.930 and map_loss < 0.1630:
                cnt = 0
                print('Saving model ...')
                save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            else:
                cnt += 1


            if cnt >= args.patience:
                break

    elif phase == 'train_cw_aug_alt_alpha_sa_sp':
        print('lr %.4f'%args.lr)


        prior='nips08'
        # prior='bms'
        # # model = Wildcat_WK_sft_gs_compf_cls_att(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                  fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin,
                                                               alpha=alpha, num_maps=num_maps,
                                                               fix_feature=fix_feature, dilate=dilate, use_grid=True,
                                                               normalize_feature=normf)  #################

        '''model_name'''
        # # # # #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_ly34_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_GS{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,GRID_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_try_g1g1_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # revisit other augmentations
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug9_3_{}_rf{}_hth{}_ms4_fdim{}_34_cw_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # # if ATT_RES:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_alt_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # else:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_alt_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # if ATT_RES:
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_alt_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_6_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # else:
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_alt_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_5_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     ## model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_4_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     ##                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_aalt_all_4_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_aalt_val_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_aalt_val_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_0123'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_6_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_aaalt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_aalt_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_aalt_3_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_alt_3_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #     # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_r{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #     #                             n_gaussian, ALT_RATIO, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        #     #
        # # if ATT_RES:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # else:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_nob_fixf_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # # if ATT_RES:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # else:
        # #     model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        # #
        # # # previsou one5 is actually one2 ... sad ...
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_rng{}_sgd_8_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,RN_GROUP,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # previsou one5 is actually one2 ... sad ...
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_dcr_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one5_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_boi{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,BOI_SIZE,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms6_fdim{}_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # prior = 'nips08'
        # #model = Wildcat_WK_hd_gs_compf_cls_att_A_sm(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) #################
        #
        # #model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_sm_aug2_rf{}_hth{}_ms_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one2_224'.format(
        # #                               n_gaussian, normf, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # print(model_name)

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
        # if ATT_RES:
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #                                      '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01.pt'),
        #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # else:
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #     #                                  '_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_bms_thm_rf0.1_hth0.1'+
        #     #                                  '_ms4_fdim512_34_cw_sa_art_ftf_2_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #                                          'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1' +
        #                                          '_ms4_fdim512_34_cw_sa_art_ftf_2_mres_2_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #                             map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_aalt_2_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #     #                                  '_hth0.1_ms4_fdim512_34_cw_sa_art_alt_3_nob_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_alt_3_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #     #                                  '_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_nob_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #     #                                  '_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_nob_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        #     #                                  'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        #     #                                  '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_nob_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
        #     #                     map_location='cuda:0')  # checkpoint is a dict, containing much info
        #     #
        #
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
        #
        # # # init with sa_art_fixf models ====================================
        # # if ATT_RES:
        # #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        # #                                      '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_rres_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'),
        # #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # #
        # # else:
        # #     checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1'+
        # #                                      '_hth0.1_ms4_fdim512_34_cw_sa_art_fixf_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02.pt'),
        # #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # #
        # # saved_state_dict = checkpoint['state_dict']
        # # new_params = model.state_dict().copy()
        # #
        # # if list(saved_state_dict.keys())[0][:7] == 'module.':
        # #     for k, y in saved_state_dict.items():
        # #         if 'feature_refine' not in k:
        # #             new_params[k[7:]] = y
        # #
        # # else:
        # #     for k, y in saved_state_dict.items():
        # #         if 'feature_refine' not in k:
        # #             new_params[k] = y
        # #
        # # model.load_state_dict(new_params)
        #
        # # init with final model ===========================================
        # # checkpoint = torch.load(os.path.join(args.path_out, 'Models',
        # #                                      'resnet50_wildcat_wk_hd_cbA16_alt2_2_0.95_compf_cls_att_gd_nf4_normTrue_hb_50_aug7' +
        # #                                      '_nips08_rf0.1_hth0.1_ms4_fdim512_34_lstm_cw_1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09.pt'),
        # #                         map_location='cuda:0')  # checkpoint is a dict, containing much info
        # # saved_state_dict = checkpoint['state_dict']
        # # new_params = model.state_dict().copy()
        # #
        # # if list(saved_state_dict.keys())[0][:7] == 'module.':
        # #     for k, y in saved_state_dict.items():
        # #         if 'feature_refine' not in k:
        # #             new_params[k[7:]] = y
        # #
        # # else:
        # #     for k, y in saved_state_dict.items():
        # #         if 'feature_refine' not in k:
        # #             new_params[k] = y
        # #
        # # model.load_state_dict(new_params)
        # #
        # # # fix ============================================
        # # # for param in model.parameters():
        # # #     if 'self_attention' not in param.name:
        # # #         param.requires_grad = False
        # # for param in model.parameters():
        # #     param.requires_grad = False
        # #
        # # if torch.cuda.device_count() > 1:
        # #     model.module.relation_net.self_attention.weight.requires_grad = True
        # #     model.module.relation_net.self_attention.bias.requires_grad = True
        # # else:
        # #     model.relation_net.self_attention.weight.requires_grad = True
        # #     model.relation_net.self_attention.bias.requires_grad = True
        # #
        # # # -----------------------------------------
        #
        # # -------- model max ---------
        # if list(saved_state_dict.keys())[0][:7] == 'module.':
        #     new_params = model.state_dict().copy()
        #     for k, y in saved_state_dict.items():
        #         new_params[k[7:]] = y
        # else:
        #     new_params = saved_state_dict.copy()
        # model_aux.load_state_dict(new_params)
        #
        # # model_aux.load_state_dict(checkpoint['state_dict'])
        # for param in model_aux.parameters():
        #     param.requires_grad = False
        #
        #
        # # if args.use_gpu:
        # #     model.cuda()
        # #     model_aux.cuda()
        # # if torch.cuda.device_count()>1:
        # #     model = torch.nn.DataParallel(model)
        # #     model_aux = torch.nn.DataParallel(model_aux)
        #
        # if torch.cuda.device_count()>1:
        #     model = torch.nn.DataParallel(model).cuda()
        #     model_aux = torch.nn.DataParallel(model_aux).cuda()
        # else:
        #     model.cuda()
        #     model_aux.cuda()
        # cudnn.benchmark = True

        # model_name = args.model_name
        # print(model_name)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  ######################
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) ######################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)  #################
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        print('finetune lr rate: %f' % args.lr)
        # print('relation lr factor: 1.0')

        s_epoch = 0
        # nss_value = 0
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

            model.load_state_dict(new_params) # try to load to initial model, too, as we have done previously.
                                              # but we do not load for basemodel_alt, following previous efforts.

            logger = Logger(os.path.join(path_models, 'log.txt'), title=title)
            # logger.set_names(['Epoch', 'LR', 'T_cps', 'V_cps', 'T_h', 'V_h', 'T_map', 'V_map', 'Nss'])
            logger.set_names(['Epoch', 'LR', 'Train_cps', 'Val_cps', 'Train_h', 'Val_h', 'Train_map', 'Val_map', 'Nss'])

        for param in model_aux.parameters():
            param.requires_grad = False

        # if args.use_gpu:
        #     model.cuda()
        #     model_aux.cuda()
        #
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        #     model_aux = torch.nn.DataParallel(model_aux)

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            model_aux = torch.nn.DataParallel(model_aux).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()


        #folder_name = 'Preds/MIT1003'
        #rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00' # T:0, F:3
        ##rf_folder = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03' # T:0, F:3
        #rf_path = os.path.join(args.path_out, folder_name, rf_folder)

        # assert os.path.exists(rf_path)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='val', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_ALL_map_full_aug(mode='all', img_h=input_h, img_w=input_w, prior=prior)
        # ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w)
        # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)

        # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w)

        # train_dataloader = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn_coco_map_rn_multiscale,
        #                               shuffle=True, num_workers=2)

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch * gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=4)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch * gpu_number, collate_fn=collate_fn_salicon_rn,
                                     shuffle=False, num_workers=4)

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
        # logits_loss = torch.nn.CrossEntropyLoss()
        logits_loss = torch.nn.BCEWithLogitsLoss()
        # logits_loss = torch.nn.BCELoss()
        h_loss = HLoss_th_2()
        # h_loss = HLoss_th()
        # h_loss = HLoss()


        if args.use_gpu:
            logits_loss = logits_loss.cuda()
            h_loss = h_loss.cuda()
            # optimizer = optimizer.cuda()

        eval_loss = np.inf
        eval_map_loss = np.inf
        cnt = 0
        # args.n_epochs = 5
        cnt = 0
        print('Initial nss value: %.4f' % nss_value)
        for i_epoch in range(s_epoch, args.n_epochs):
            adjust_learning_rate(optimizer, i_epoch, args.schedule)
            is_best = False
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_alt_alpha_sa_sp(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                                       train_dataloader, args, path_models)
            # tmp_eval_loss, map_loss = eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw_sa_sp(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)
            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # scheduler.step()

            # if tmp_eval_loss < eval_loss:
            #     cnt = 0
            #     eval_loss = tmp_eval_loss
            #     print('Saving model ...')
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            # elif map_loss < eval_map_loss:
            #     cnt = 0
            #     eval_map_loss =  map_loss
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
        h_loss = HLoss_th_2()
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
        h_loss = HLoss_th_2()
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
        h_loss = HLoss_th_2()
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
    # **** 210502 ***
    elif phase == 'train_cw_alt_alpha':
        print('lr %.4f' % args.lr)
        # prior = 'gbvs'
        # prior = 'bms'
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

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                 num_maps=num_maps,
                                                 fix_feature=fix_feature, dilate=dilate, use_grid=True, # False for no grid
                                                 normalize_feature=normf)

        model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_cw(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
                                                     num_maps=num_maps,
                                                     fix_feature=fix_feature, dilate=dilate, use_grid=True, # False for no grid
                                                     normalize_feature=normf)
        
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_noobj(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                          num_maps=num_maps,
        #                                          fix_feature=fix_feature, dilate=dilate, use_grid=False, # False for no grid
        #                                          normalize_feature=normf)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_noobj(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                              num_maps=num_maps,
        #                                              fix_feature=fix_feature, dilate=dilate, use_grid=False, # False for no grid
        #                                              normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                          num_maps=num_maps,
        #                                          fix_feature=fix_feature, dilate=dilate, use_grid=True,
        #                                          normalize_feature=normf)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                              num_maps=num_maps,
        #                                              fix_feature=fix_feature, dilate=dilate, use_grid=True,
        #                                              normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                          num_maps=num_maps,
        #                                          fix_feature=fix_feature, dilate=dilate, use_grid=True,
        #                                          normalize_feature=normf)
        #
        # model_aux = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha,
        #                                              num_maps=num_maps,
        #                                              fix_feature=fix_feature, dilate=dilate, use_grid=True,
        #                                              normalize_feature=normf)


        '''checkpoint'''
        # global center bias
        # checkpoint = torch.load(os.path.join(path_models,'resnet50_wildcat_wk_hd_cbG16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_'+
        #                                                  'rf0.1_hth0.1_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01.pt'))

        # no center bias
        # checkpoint = torch.load(os.path.join(path_models,'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_'+
        #                                                  'rf0.1_hth0.1_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'))

        # bms prior
        # checkpoint = torch.load(os.path.join(path_models,'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_bms_thm_'+
        #                         'rf0.1_hth0.1_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'))

        # no grid
        # checkpoint = torch.load(os.path.join(path_models,'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_'+
        #                                                  'noGrid_2_rf0.1_hth0.1_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'))

        # no object
        # checkpoint = torch.load(os.path.join(path_models,'resnet101_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_'+
        #                                                  'noobj_rf0.1_hth0.1_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch07.pt'))

        # gbvs prior
        # checkpoint = torch.load(os.path.join(path_models,'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_gbvs_0.5_thm_'+
        #                         'rf0.1_hth0.1_2_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'))

        # no prior loss
        # checkpoint = torch.load(os.path.join(path_models, 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_'+
        # 'rf0.1_hth0.0_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt'))

        # gbvs prior
        # checkpoint = torch.load(
        #     os.path.join(path_models, 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_bms_thm_' +
        #                  'rf0.1_hth0.1_ms4_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03.pt'))

        # *** 210427 model ***
        # model_aux_path = os.path.join(args.path_out, 'Models', args.init_model, args.bestname)
        # checkpoint_aux = torch.load(model_aux_path)
        #
        # saved_state_dict = checkpoint_aux['state_dict']
        # if list(saved_state_dict.keys())[0][:7] == 'module.':
        #     new_params = model.state_dict().copy()
        #     for k, y in saved_state_dict.items():
        #         new_params[k[7:]] = y
        # else:
        #     new_params = saved_state_dict.copy()
        # model_aux.load_state_dict(new_params)
        # # model.load_state_dict(new_params) # this might be faster? change lr from 1e-4 to 1e-5 then


        '''model name'''
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_4_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # model_name = 'resnet101_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_2_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet101_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noobj_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #                                 n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate) #_gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbG{}_alt_5_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_{}_thm_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #     n_gaussian, normf, MAX_BNUM, prior, GBVS_R, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_nobs_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_0.25_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #     n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # _gcn_all
        #
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # others
        # # model_name = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_twocls_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224'.format(
        # #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate)  # rf0.1, hth0.0

        model_name = args.model_name
        print(model_name)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  ############################
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  ############################
        # optimizer = torch.optim.Adam(model.get_config_optim(args.lr, 1.0, 0.1), lr=args.lr)

        print('relation lr factor: 1.0')
        # print('alt learning rate: 1e-5')

        s_epoch = 0
        # nss_value = 0
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

        # if args.use_gpu:
        #     model.cuda()
        #     model_aux.cuda()
        #
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        #     model_aux = torch.nn.DataParallel(model_aux)

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
            model_aux = torch.nn.DataParallel(model_aux).cuda()
            cudnn.benchmark = True
        gpu_number = torch.cuda.device_count()
        # # ds_train = MS_COCO_map_full(mode='train', img_h=input_h, img_w=input_w)
        ds_train = MS_COCO_map_full_aug(mode='train', img_h=input_h, img_w=input_w, prior = prior) # , N=48 # ***
        # # ds_train = ILSVRC_full(mode='train', img_h=input_h, img_w=input_w)
        # # ds_validate = ILSVRC_full(mode='val', img_h=input_h, img_w=input_w)
        # ds_train = MS_COCO_ALL_map_full_aug(mode='all', img_h=input_h, img_w=input_w, prior=prior)  # *******

        ds_validate = SALICON_full(mode='val', img_h=input_h, img_w=input_w) # , N=24

        train_dataloader = DataLoader(ds_train, batch_size=args.train_batch * gpu_number, collate_fn=collate_fn_coco_map_rn,
                                      shuffle=True, num_workers=2)

        eval_dataloader = DataLoader(ds_validate, batch_size=args.test_batch * gpu_number, collate_fn=collate_fn_salicon_rn,
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
        # logits_loss = torch.nn.BCELoss()

        h_loss = HLoss_th_2()
        # h_loss = HLoss()

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
            train_cps, train_h, train_map = train_Wildcat_WK_hd_compf_map_cw_alt_alpha(i_epoch, model, model_aux, optimizer, logits_loss, h_loss,
                                                    train_dataloader, args, path_models)

            val_cps, val_h, val_map = eval_Wildcat_WK_hd_compf_salicon_cw(i_epoch, model, logits_loss, h_loss, eval_dataloader, args)

            # tmp_eval_salicon_loss = eval_Wildcat_WK_hd_compf_map(i_epoch, model, logits_loss, h_loss, eval_map_dataloader, args)

            # if tmp_eval_loss < eval_loss:
            #
            #     cnt = 0
            #
            #     eval_loss = tmp_eval_loss
            #
            #     print('Saving model ...')
            #
            #     save_model(model, optimizer, i_epoch, path_models, eval_loss, name_model=model_name)
            #
            # else:
            #
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
        E_NUM = list(range(9))
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)

        for e_num in E_NUM:

            if ATT_RES:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_ftf_2_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all

            else:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_ftf_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
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

    elif phase == 'test_cw_sa_multiscale':
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=normf) # noGrid
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        E_NUM = [3,4]
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'

        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)

        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        for e_num in E_NUM:

            if ATT_RES:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_ftf_2_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all

            else:
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_ftf_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #              n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                             n_gaussian, normf, MAX_BNUM, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_sa_art_fixf_3_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_sa_art_fixf_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_sa_art_fixf_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_fixf_2_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_fixf_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_fixf_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

            print("Testing %s ..."%best_model_file)
            # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            test_Wildcat_WK_hd_compf_multiscale_cw_sa(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)


            # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
            # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw_sa_sp':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        # folder_name = 'Preds/PASCAL-S'
        folder_name = 'Preds/MIT1003'
        # best_model_file = 'no_training'
        E_NUM = [0]
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)

        for e_num in E_NUM:

            if ATT_RES:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                #

            else:
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all

            print("Testing %s ..."%best_model_file)
            # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_cw_sa(model, folder_name, best_model_file, test_dataloader, args)
            test_Wildcat_WK_hd_compf_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args)
            # test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args)


            # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
            # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw_sa_sp_multiscale':
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_rank(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) # for storing all the elements

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                           fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        # folder_name = 'Preds/PASCAL-S'
        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        # folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'
        E_NUM = [1]
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

        for e_num in E_NUM:

            if ATT_RES:
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'ours'

            else: #resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_multiscale
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_9_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_aalt_val_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,fix_feature, dilate, e_num)  # _gcn_all
                #best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_5_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_5_proa_{}_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, PRO_RATIO, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_4_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_aalt_3_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_alt_3_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all

                # --------------sa_art_fixf_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

                # --------------sa_art_ftf_2_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

                # --------------alt sa_art_ftf_2_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all
                best_model_file = 'ours'

            print("Testing %s ..."%best_model_file)
            # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_cw_sa(model, folder_name, best_model_file, test_dataloader, args)
            # test_Wildcat_WK_hd_compf_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args)

            test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes, metrics=('nss',))
            # # test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp_rank(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)
            # evaluate(args, folder_name, best_model_file, metrics=('aucs',)) #'aucj'

            # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
            # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw_sa_sp_multiscale_210822':
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_rank(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) # for storing all the elements

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                           fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        model_name = args.model_name
        print(model_name)

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


    elif phase == 'test_cw_sa_sp_multiscale_rank_rebuttal':
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_rank(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) # for storing all the elements
        #
        # #model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                    fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)
        #
        # # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        # #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_rank_rebuttal(n_classes=coco_num_classes, kmax=kmax, kmin=kmin,
                                                                    alpha=alpha, num_maps=num_maps,
                                                                    fix_feature=fix_feature, dilate=dilate,
                                                                    use_grid=True, normalize_feature=normf)
        if args.use_gpu:
            model.cuda()

        # folder_name = 'Preds/PASCAL-S'
        folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        # folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'
        E_NUM = [1]
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'
        #prior = 'bms'
        # prior = 'gbvs'
        args.batch_size = 1

        # ds_test = SALICON_full(return_path=True, img_h=input_h, img_w=input_w, mode='train')  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
        #                          shuffle=False, num_workers=2)
        # ds_test = SALICON_test(return_path=True, img_h=input_h, img_w=input_w, mode='test')  # N=4,
        # # # ds_test = MIT300_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.test_batch, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)
        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]

        for e_num in E_NUM:

            if ATT_RES:
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all
                #

            else: #resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_multiscale
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                ###best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                ###    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                best_model_file = 'ours'
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_9_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_aalt_val_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,fix_feature, dilate, e_num)  # _gcn_all
                #best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_5_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_all_5_proa_{}_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, PRO_RATIO, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_4_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_aalt_3_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_alt_3_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all

                # --------------sa_art_fixf_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

                # --------------sa_art_ftf_2_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                ###best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                ###    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

                # --------------alt sa_art_ftf_2_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all


            print("Testing %s ..."%best_model_file)
            # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_cw_sa(model, folder_name, best_model_file, test_dataloader, args)
            # test_Wildcat_WK_hd_compf_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args)

            # test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)
            # test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp_rank(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)
            test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp_rank_rebuttal(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)


            # tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
            # ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_multiscale(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

    elif phase == 'test_cw_sa_sp_multiscale_rank':
        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp_rank(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                            fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf) # for storing all the elements

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nopsal_sa_art_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_sa_new_sp(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        if args.use_gpu:
            model.cuda()

        # folder_name = 'Preds/PASCAL-S'
        # folder_name = 'Preds/MIT1003'
        # folder_name = 'Preds/MIT300'
        # folder_name = 'Preds/SALICON' #validation set
        # folder_name = 'Preds/SALICON_test'
        folder_name = 'Preds/SALICON_train'
        # best_model_file = 'no_training'
        E_NUM = [0]
        # E_NUM.extend(list(range(5,16)))
        prior = 'nips08'
        args.batch_size = 1

        ds_test = SALICON_full(return_path=True, img_h=input_h, img_w=input_w, mode='train', N=4)  # N=4,
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_salicon_rn,
                                 shuffle=False, num_workers=2)
        # ds_test = SALICON_test(return_path=True, img_h=input_h, img_w=input_w, mode='test')  # N=4,
        # # # ds_test = MIT300_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
        #                              shuffle=False, num_workers=2)

        # ds_test = PASCAL_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  # N=4,
        # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
        #                              shuffle=False, num_workers=2)

        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]

        for e_num in E_NUM:

            if ATT_RES:
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_rres_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                    fix_feature, dilate, e_num)  # _gcn_all
                #

            else: #resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_multiscale
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                    n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_4_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_aalt_3_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_alt_3_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_nob_mres_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_new_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps,
                #     fix_feature, dilate, e_num)  # _gcn_all

                # --------------sa_art_fixf_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_fixf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_fixf_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

                # --------------sa_art_ftf_2_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_{}_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, GBVS_R, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,dilate, e_num)

                # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet50_wildcat_wk_hd_nobs_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_norn_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_nopsal_rf{}_hth{}_ms4_sa_art_ftf_2_3_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
                # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_noGrid_rf{}_hth{}_ms4_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

                # --------------alt sa_art_ftf_2_sp-----------------------
                # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,FEATURE_DIM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all


            print("Testing %s ..."%best_model_file)
            # ds_test = MIT1003_full(return_path=True, img_h=input_h, img_w=input_w)  #N=4,
            # args.batch_size = 1
            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
            #                              shuffle=False, num_workers=2)
            # test_Wildcat_WK_hd_compf_cw_sa(model, folder_name, best_model_file, test_dataloader, args)
            # test_Wildcat_WK_hd_compf_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args)

            # test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)
            test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp_rank(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)
            # test_Wildcat_WK_hd_compf_multiscale_cw_sa_sp_gs(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)


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
        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        #model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_multiscale(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                    fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=normf) # noGrid

        model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_noobj(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
                         fix_feature=fix_feature, dilate=dilate, use_grid=False, normalize_feature=normf) #################

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_norn(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_gbs(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

        # model = Wildcat_WK_hd_gs_compf_cls_att_A4_cw_nobs(n_classes=coco_num_classes, kmax=kmax, kmin=kmin, alpha=alpha, num_maps=num_maps,
        #                     fix_feature=fix_feature, dilate=dilate, use_grid=True, normalize_feature=normf)

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

        tgt_sizes = [int(224 * i) for i in (0.5, 0.75, 1.0, 1.25, 1.50, 2.0)]
        # ds_test = PASCAL_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
        ds_test = MIT1003_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
        # ds_test = MIT300_full(return_path=True, img_h=max(tgt_sizes), img_w=max(tgt_sizes))  # N=4,
        args.batch_size = 1
        test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit1003_rn,
                                     shuffle=False, num_workers=2)

        E_NUM = [5] #
        # E_NUM = [1,2,8] # _gbs
        # E_NUM = [1,2,3,4,5,7] # _nobs
        for e_num in E_NUM:
            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_3_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_bms_thm_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num)

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_0.5_thm_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num)

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_gbvs_0.25_rf{}_hth{}_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #                             n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num)

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf0.0_hth0.1_twocls_2_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #                             n_gaussian, normf, MAX_BNUM,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num)

            # best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_noGrid_2_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #                                 n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

            best_model_file = 'resnet101_wildcat_wk_hd_cbA{}_alt_2_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_noobj_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
                                            n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_compf_cls_att_gd_nf4_norm{}_hb_{}_proa_{}_aug7_nips08_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #     n_gaussian, normf, MAX_BNUM, PRO_RATIO, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

            #best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_all_4_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #    n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature,dilate, e_num)  # _gcn_all

            # best_model_file = 'resnet50_wildcat_wk_hd_cbG{}_alt_4_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #                                 n_gaussian, normf, MAX_BNUM, rf_weight, hth_weight,kmax,kmin,alpha,num_maps,fix_feature, dilate, e_num) #_gcn_all

            # best_model_file = 'resnet50_wildcat_wk_hd_nobs_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_nips08_rf{}_hth{}_ms4_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
            #     normf, MAX_BNUM, rf_weight, hth_weight, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all

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

            # test_dataloader = DataLoader(ds_test, batch_size=args.batch_size, collate_fn=collate_fn_mit300_rn,
            #                              shuffle=False, num_workers=2)
            test_Wildcat_WK_hd_compf_multiscale_cw(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)
            # test_Wildcat_WK_hd_compf_multiscale_cw_rank(model, folder_name, best_model_file, test_dataloader, args, tgt_sizes=tgt_sizes)

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
    # if epoch == 1:
    #     for param_group in optimizer.param_groups: # uncomment this for basemodel_210511_sgd
    #         param_group['lr'] *= 10.0

    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma


if __name__ == '__main__':
    args = parse_arguments()

    main_Wildcat_WK_hd_compf_map(args)