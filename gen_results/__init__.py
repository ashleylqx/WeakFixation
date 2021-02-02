# original folder
# H:/Codes/WF/Preds/SALICON_train

# **** for all images ****
# object_mask/ # contains 10k bbox images
# pred_and_label/ # label npy and gaussian(?) npy
# pred_map/ # 10k pred saliency maps
# semantics_cw # 10k, feature map of the spatial semantics, not fused yet.
# ====== these files are saved by main.py, 'test_cw_sa_sp_multiscale_rank'


# **** for selected images ******
# proposal_map # map of each bbox
# semantic_map # channel of top-three class; visualized img&class map
# tmp_image*/  # img+pred+gt map+bbox map
# ====== these files are saved by H:/Codes/code_forMetrics_new/Demo_wf_select.m
#                  and visualized by Draw_semantics_maps.m


# for MIT1003 ++++++++++++++++++++++++++++++++
# original:
# cw_map:
# cb_map:
# bbox_map:
# best_model_file = 'resnet50_wildcat_wk_hd_cbA{}_alt_compf_cls_att_gd_nf4_norm{}_hb_{}_aug7_{}_rf{}_hth{}_ms4_fdim{}_34_cw_sa_art_ftf_2_mres_2_sp_kmax{}_kmin{}_a{}_M{}_f{}_dl{}_one3_224_epoch{:02d}'.format(
#                     n_gaussian, normf, MAX_BNUM, prior, rf_weight, hth_weight, FEATURE_DIM, kmax, kmin, alpha, num_maps, fix_feature, dilate, e_num)  # _gcn_all
# ============= main.py 'test_cw_sa_sp_multiscale_rank_rebuttal'


# full model:
# resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_2_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00
    # rename this model to ours.pt
# cp resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_2_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch00.pt ours.pt
