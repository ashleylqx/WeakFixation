
%% Demo.m 
% All the codes in "code_forMetrics" are from MIT Saliency Benchmark (https://github.com/cvzoya/saliency). Please refer to their webpage for more details.

% load global parameters, you should set up the "ROOT_DIR" to your own path
% for data.
clear all
% METRIC_DIR = 'code_forMetrics';
% addpath(genpath(METRIC_DIR));

%% path to store evaluation results
CACHE = ['cache_wf/'];
if ~exist(CACHE, 'dir')
    mkdir(CACHE);
end
%%
% options.Result_path = '/home/qx/WF/Preds/';
% options.DS_path = '/home/qx/DataSets/';
% base_path = '/media/qx/dgx-r69-1/';
% base_path = '/home/qx/';
base_path = '/research/dept2/qxlai/';

Datasets{1} = 'MIT1003';
%Datasets{2} = 'PASCAL-S';
%Datasets{3} = 'DUTOMRON';
%Datasets{4} = 'TORONTO';
% Datasets{2} = 'Hollywood-2';
% Datasets{3} = 'DHF1K';

% options.Result_path = [base_path, 'WF/Preds/', DataSets{i}, '/'];
options.DS_path = [base_path, 'DataSets/'];


Metrics{5} = 'NSS'; 
Metrics{2} = 'similarity'; 
Metrics{4} = 'CC';
Metrics{1} = 'AUC_Judd';
Metrics{3} = 'AUC_shuffled';

%Results{1} = 'MotionAwareTwoStream';
Results{1} = 'UNISAL';
Results{2} = 'scribble_saliency';
Results{3} = 'scwssod';
Results{4} = 'val_8_sum_three_multiscale';
Results{5} = 'val_7_sum_two_1_multiscale';
% Results{1} = 'SALLSTM';
% Results{1} = 'resnet50_wildcat_wk_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch03';
% Results{1} = 'resnet50_wildcat_wk_comp2_conv_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch00';
% Results{1} = 'resnet50_wildcat_wk_comp_conv_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch12';
% Results{1} = 'resnet50_wildcat_wk_comp_conv_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch27';
% Results{1} = 'resnet50_wildcat_wk_hth3_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch07'; % not bad,but is wrong
% Results{1} = 'resnet50_wildcat_wk_hth0.1_3_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch04';
% Results{1} = 'resnet50_wildcat_wk_hth0.1_2_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch05';
% Results{1} = 'resnet50_wildcat_wk_hth0.2_3_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch04';
% Results{2} = 'resnet50_wildcat_wk_hth0.1_ms_2_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch13'; 
% Results{3} = 'resnet50_wildcat_wk_hth0.2_ms_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch03';
% Results{1} = 'resnet50_wildcat_wk_hth0.2_ms_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08';% recent best
% Results{2} = 'resnet50_wildcat_wk_hth0.2_ms_2_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08';
% % Results{3} = 'resnet50_wildcat_wk_hth0.2_ms_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch10';
% % Results{4} = 'resnet50_wildcat_wk_hth0.2_ms_2_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch11';
% % Results{5} = 'resnet50_wildcat_wk_hth0.2_ms_2_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch10';
% Results{3} = 'resnet50_wildcat_wk_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08';
% Results{4} = 'resnet50_wildcat_wk_hth0.2_ms_kmax1_kmin1_a1.0_M1_fFalse_dlFalse_448_epoch11';
% % Results{5} = 'resnet50_wildcat_wk_hth0.2_ms_2_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch12';
% % Results{5} = 'resnet50_wildcat_wk_hth0.2_ms_2_kmax1_kmin1_a0.7_M4_fFalse_dlFalse_448_epoch14';
% % Results{5} = 'resnet50_wildcat_wk_hth0.2_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch00';% wrong testing
% % Results{6} = 'resnet50_wildcat_wk_hth0.2_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch00_nosigmap';
% Results{5} = 'resnet50_wildcat_wk_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch11';
% % Results{6} = 'resnet50_wildcat_wk_ms_hth0.0_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch05';
% % Results{7} = 'resnet50_wildcat_wk_hth0.2_ms_kmax1_kmin1_a1.0_M1_fFalse_dlFalse_448_epoch19';
% % Results{8} = 'resnet50_wildcat_wk_hth0.2_ms_kmax1_kminNone_a1.0_M1_fFalse_dlFalse_448_epoch22';
% Results{6} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch01';
%%==Results{1} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch04';
% Results{2} = 'resnet50_wildcat_wk_compf_self_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch04';
% Results{2} = 'resnet50_wildcat_wk_compf_ms_hth0.2_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch13';
% Results{2} = 'resnet50_wildcat_wk_hth0.05_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08';
% Results{3} = 'resnet50_wildcat_wk_hth0.15_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06';
% Results{4} = 'resnet50_wildcat_wk_hth0.01_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch05';
% Results{2} = 'resnet50_wildcat_wk_hth0.15_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch10';
% Results{3} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch14';
% Results{2} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch10';
%%==Results{2} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch13';
% Results{3} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10';
% Results{4} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09';
% Results{5} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
%%==Results{3} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch14';
% Results{4} = 'resnet50_wildcat_wk_hth0.1_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch11';
% Results{5} = 'resnet50_wildcat_wk_hth0.0_ms_signorm_nosigmap_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch03';
% Results{4} = 'resnet50_wildcat_wk_compf_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06';
% Results{4} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08';
% Results{4} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06';
% % Results{6} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch04';
% Results{5} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06_1';
% Results{6} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06_2';

% Results{4} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{5} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{6} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% 
% Results{7} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch10';
% Results{8} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch09';
% Results{9} = 'resnet50_wildcat_wk_compf_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch06';

% Results{4} = 'resnet50_wildcat_wk_compf_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch06';
% 
% Results{5} = 'resnet50_wildcat_wk_compf_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{6} = 'resnet50_wildcat_wk_compf_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{7} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08';
% Results{8} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch07';
% Results{9} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06';
% Results{10} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch05';
% 
% Results{11} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch05';
% 
% Results{12} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% 
% Results{13} = 'resnet50_wildcat_wk_compf_cb8_2_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06'; %
% Results{14} = 'resnet50_wildcat_wk_compf_cb8_2_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04'; % 0.9244
% 
% Results{15} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{16} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';%0.9427
% Results{17} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{18} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{19} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch05';
% 
% Results{20} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06';

%%==Results{4} = 'gbvs';
%%==Results{5} = 'itti';
%%==Results{6} = 'ACoL';

% Results{6} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch07';
% Results{7} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch05';
% Results{8} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch07';
% Results{6} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch06';
% Results{12} = 'res50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch05';

% Results{7} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch05';
% Results{8} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch07';
% Results{9} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch10';
% 
% Results{10} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch08_1';
% Results{11} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch08_2';
% Results{12} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch06_1';
% Results{13} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch06_2';
% Results{14} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch05_1';
% Results{15} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch05_2';
% 
% 
% Results{16} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch09';
% Results{17} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch07';
% Results{18} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch04';

% Results{19} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch09_1';
% Results{20} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch09_2';
% Results{21} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch08_1';
% Results{22} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch08_2';
% Results{23} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch07_1';
% Results{24} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch07_2';
% Results{25} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch05_1';
% Results{26} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch05_2';

% Results{27} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch14_1';
% Results{28} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch14_2';
% Results{29} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_1';
% Results{30} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_2';
% Results{31} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_1';
% Results{32} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_2';
% Results{33} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_1';
% Results{34} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_2';
% Results{35} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_1';
% Results{36} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_2';
% Results{37} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_1';
% Results{38} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_2';

% Results{6} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08_1';
% Results{7} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08_2';
% Results{8} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch05_1';
% Results{9} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch05_2';
% 
% Results{10} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10_1';
% Results{11} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10_2';
% Results{12} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_1';
% Results{13} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_2';
% 
% Results{14} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch13_1';
% Results{15} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch13_2';
% Results{16} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch10_1';
% Results{17} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_epoch10_2';
% 
% Results{18} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch09_1';
% Results{19} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch09_2';
% Results{20} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08_1';
% Results{21} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch08_2';
% Results{22} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch07_1';
% Results{23} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch07_2';
% Results{24} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06_1';
% Results{25} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch06_2';

% ----------------------new3 bias -------------------------------
% Results{6} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch06';
% Results{7} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch05';
% Results{8} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch04';
% 
% Results{9} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch08';
% Results{10} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch05';
% Results{11} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch04';
% 
% Results{12} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch06';
% Results{13} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch05';
% Results{14} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch03';
% 
% Results{15} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch07';
% Results{16} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch05';
% Results{17} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch04';

% Results{6} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch09';
% Results{7} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch06';
% Results{8} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch04';
% 
% Results{9} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch07';
% Results{10} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch05';
% Results{11} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch04';
% 
% Results{12} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch07';
% Results{13} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch05';
% 
% Results{14} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch05';
% Results{15} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch04';
% Results{16} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch03';

% ------------------------------new4 bias----------------------------
% Results{6} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch05';
% Results{7} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch06';
% Results{8} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch07';
% Results{9} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch08';
% 
% Results{10} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{11} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% 
% Results{12} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{13} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{14} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{15} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% 
% Results{16} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch08';
% Results{17} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch06';
% Results{18} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch05';
% 
% Results{19} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch05';
% 
% Results{20} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{21} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{22} = 'resnet50_wildcat_wk_compf_cb8_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% 
% Results{23} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{24} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% 
% Results{25} = 'resnet50_wildcat_wk_compf_cb8_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new4_epoch05';


% ----------------------------
% Results{6} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch11_1';
% Results{7} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch11_2';
% Results{8} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch14_1';
% Results{9} = 'resnet50_wildcat_wk_compf_self_x_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch14_2';
% 
% Results{10} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch10_1';
% Results{11} = 'resnet50_wildcat_wk_compf_self_two_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlFalse_448_epoch10_2';
% 
% Results{12} = 'resnet50_wildcat_wk_compf_cb8_G_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch11';
% 
% Results{13} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{14} = 'resnet50_wildcat_wk_compf_cb8_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';

%------cb16------------
% Results{6} = 'resnet50_wildcat_wk_compf_cb16_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch10';
% Results{7} = 'resnet50_wildcat_wk_compf_cb16_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch08';
% Results{8} = 'resnet50_wildcat_wk_compf_cb16_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch07';
% Results{9} = 'resnet50_wildcat_wk_compf_cb16_G_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one_224_new3_epoch06';
% 
% Results{10} = 'resnet50_wildcat_wk_compf_cb16_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch10';
% Results{11} = 'resnet50_wildcat_wk_compf_cb16_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch09';
% Results{12} = 'resnet50_wildcat_wk_compf_cb16_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch05';
% Results{13} = 'resnet50_wildcat_wk_compf_cb16_2_self_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new3_epoch03';
% 
% Results{14} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{15} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';
% Results{16} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';
% Results{17} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{18} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';

%-----------------------hd-------------------------
% Results{6} = 'resnet50_wildcat_wk_hd_compf_nosig_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{7} = 'resnet50_wildcat_wk_hd_compf_nosig_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% 
% Results{8} = 'resnet50_wildcat_wk_hd_compf_divs_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{9} = 'resnet50_wildcat_wk_hd_compf_divs_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% 
% Results{10} = 'resnet50_wildcat_wk_hd_compf_x_sameb_hth0.0_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{11} = 'resnet50_wildcat_wk_hd_compf_x_sameb_hth0.0_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{12} = 'resnet50_wildcat_wk_hd_compf_x_sameb_hth0.0_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% 
% Results{13} = 'resnet50_wildcat_wk_hd_compf_x_divs_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{14} = 'resnet50_wildcat_wk_hd_compf_x_divs_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% 
% Results{15} = 'resnet50_wildcat_wk_hd_compf_x_nosig_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{16} = 'resnet50_wildcat_wk_hd_compf_x_nosig_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{17} = 'resnet50_wildcat_wk_hd_compf_x_nosig_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';

%---------multiply-------------------------
%--------------okkkkkkkkkkk
% Results{6} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{7} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{8} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% 
% Results{9} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{10} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{11} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';

%---------ok
% Results{12} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{13} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{14} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% 
% Results{15} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{16} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{17} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';

%-----------------o
% Results{18} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{19} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{20} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% 
% Results{21} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{22} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{23} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';

%-----------------ok
% Results{24} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{25} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_sft';
% Results{26} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_hd';
% 
% Results{27} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{28} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';
% Results{29} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% 
% Results{30} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{31} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{32} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_three_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';

%------------------------ok
% Results{33} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{34} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{35} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% 
% Results{36} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{37} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{38} = 'resnet50_wildcat_wk_compf_x_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';

%------------------------ok
% Results{39} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{40} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_sft';
% Results{41} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_hd';
% not exit epoch02
% Results{42} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{43} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{44} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_res_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';

%--------------------------o
% Results{45} = 'resnet50_wildcat_wk_compf_divs_sep_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{46} = 'resnet50_wildcat_wk_compf_divs_sep_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{47} = 'resnet50_wildcat_wk_compf_divs_sep_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% 
% Results{48} = 'resnet50_wildcat_wk_compf_divs_sep_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{49} = 'resnet50_wildcat_wk_compf_divs_sep_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{50} = 'resnet50_wildcat_wk_compf_divs_sep_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';


% Results{6} = 'resnet50_wildcat_wk_hd_gcn_compf_divs_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{7} = 'resnet50_wildcat_wk_hd_gcn_compf_x_divs_nr_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{6} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{7} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_sft';
% Results{8} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_hd';
% 
% Results{9} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{10} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';
% Results{11} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% 
% Results{12} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{13} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{14} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% 
% Results{15} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';
% Results{16} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_sft';
% Results{17} = 'resnet50_wildcat_wk_compf_divs_sep_l_three_c8_multi_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_hd';


% Results{6} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch00';

% Results{7} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';

% Results{7} = 'vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{8} = 'vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{9} = 'vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03'; %
% 
% Results{10} = 'vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_224_new4_epoch08';
% Results{11} = 'vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_224_new4_epoch05';
% Results{12} = 'vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_224_new4_epoch04'; %
% 
% Results{13}  ='vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch10';
% Results{14}  ='vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch08';
% Results{15}  ='vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch07';
% Results{16}  ='vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch05';
% Results{17}  ='vgg16_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch04'; %
% 
% Results{18} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch06';
% Results{19} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch05';
% Results{20} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch04';
% Results{21} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch03';
% Results{22} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch02'; %
% 
% Results{23} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch12';
% Results{24} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch11';%
% Results{25} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';
% Results{26} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{27} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';%
% 
% Results{39} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch11'; %**
% Results{28} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';
% Results{29} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{30} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{31} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% 
% Results{40} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch11';
% Results{32} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';
% Results{33} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{34} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{35} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% 
% Results{36} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';
% Results{37} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{38} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';

%%==Results{7} = 'nips08';
%%==Results{8} = 'CAM_res50';
%%==Results{9} = 'CAM_sqz1_1';

% Results{10} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch12';
% Results{11} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch13';
% Results{12} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch14_pascal';%pascal
% 
% Results{13} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch05';
% Results{14} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch04';
% 
% Results{15} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch06';
% Results{16} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch07';
% Results{17} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch08';%
% 
% Results{18} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch07';
% Results{19} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch12';
% Results{20} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch13';
% Results{21} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.7_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch14';%---
% 
% Results{22} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch09';
% Results{23} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch10';
% Results{24} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch12';
% Results{25} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch14';
% Results{26} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug0.2_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_new4_epoch16';% home
% 
% Results{27} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{28} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{29} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';
% Results{30} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch10';
% Results{31} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch12';
% Results{32} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_aug1.0_2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';% home
% 
% Results{33} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch11';
% Results{34} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch10';
% Results{35} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{36} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{37} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{38} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02'; %

% Results{39} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch01';
% Results{40} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% Results{41} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{42} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{43} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{44} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';
% Results{45} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';%
% 
% Results{46} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.01_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{47} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.01_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{48} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.01_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{49} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.01_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{50} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.01_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';
% Results{51} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.01_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';

% Results{10} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.15_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{11} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.15_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{12} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{13} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% 
% Results{14} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% 
% Results{15} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.15_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{16} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% 
% Results{17} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{18} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';

% Results{10} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{11} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{12} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';

% Results{13} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{14} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% 
% Results{16} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% 
% Results{19} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{20} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% 
% Results{22} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{23} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';


% Results{15} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.15_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{16} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.15_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';
% Results{17} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.15_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch14';
% 
% Results{18} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{19} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';
% Results{20} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch10';
% Results{21} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch12';
% 
% Results{22} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';
% Results{23} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch13';
% Results{24} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch14';
% Results{25} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_rf0.05_aug0.2_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch17';
% 
% Results{26} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{27} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{28} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';
% 
% Results{29} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{30} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{31} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% 
% Results{32} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{33} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{34} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% % 
% Results{35} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{36} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{37} = 'resnet50_wildcat_wk_compf_divs_sep_l_final_c8_multi_rf0.2_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% % 

% Results{10} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.75_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{11} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.75_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{12} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.75_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% 
% Results{13} = 'resnet50_wildcat_wk_compf_divs_sep_l_two_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{14} = 'resnet50_wildcat_wk_compf_divs_sep_l_two_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_divs_sep_l_two_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';

% Results{10} = 'resnet50_wildcat_wk_compf_divs_sep_l_two_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{11} = 'resnet50_wildcat_wk_compf_divs_sep_l_two_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{12} = 'resnet50_wildcat_wk_compf_divs_sep_l_two_c8_multi_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% 
% Results{13} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{14} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';
% 
% Results{16} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{17} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_divs_sep_l_two2_c8_multi_rf0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';

% Results{10} = 'resnet50_wildcat_wk_hd_compf_sameb4_gs_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{11} = 'resnet50_wildcat_wk_hd_compf_sameb4_gs_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';

% Results{10} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% Results{11} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch01';
% 
% Results{12} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% Results{13} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch01';
% 
% Results{14} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch01';
% 
% Results{15} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{16} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{17} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% 
% Results{18} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{19} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{20} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_augms_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';


% Results{10} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.0_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% Results{11} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.0_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{12} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.0_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{13} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.0_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{14} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.0_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';

% Results{15} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% Results{16} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{17} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{18} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{19} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{20} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';
% Results{21} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_ps_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';

% Results{10} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% Results{11} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{12} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.1_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% 
% Results{22} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.0_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% Results{23} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.0_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% 
% 
% Results{14} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch02';
% Results{15} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch03';
% Results{16} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch04';
% Results{17} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch05';
% Results{18} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch06';
% Results{19} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch07';
% Results{20} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch08';
% Results{21} = 'resnet50_wildcat_wk_compf_cb16_2_self_x_res_pss_rf0.05_hth0.1_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_new4_epoch09';

% Results{10} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05'; %0.8173
% Results{11} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06'; %0.8639

% Results{10} = 'resnet50_wildcat_hd_rn_cls6_eb256_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03'; % 0.7994
% Results{11} = 'resnet50_wildcat_hd_rn_cls6_eb256_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04'; % 0.7340

% Results{10} = 'resnet50_wildcat_hd_rn_cls6_eb128_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';%0.7504
% Results{11} = 'resnet50_wildcat_hd_rn_cls6_eb128_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';%0.7659

% Results{10} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch05';%0.7216
% Results{11} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch04';%0.7230
% Results{12} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03';%0.7681
% 
% Results{13} = 'resnet50_wildcat_hd_rn_cls6_eb256_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one4_224_epoch04';%0.6031
% Results{14} = 'resnet50_wildcat_hd_rn_cls6_eb256_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch04';%0.6423
% 
% Results{15} = 'resnet50_wildcat_hd_rn_cls6_eb128_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch04';%0.6213

% Results{10} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_ad_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';

% Results{10} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_mtad1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{11} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_mtad2_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';


% Results{10} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{12} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';

% Results{16} = 'resnet50_wildcat_hd_sft_compf_cls_att_avg_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_cls_att_avg_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{18} = 'resnet50_wildcat_hd_sft_compf_cls_att_avg_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_cls_att_avg_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{20} = 'resnet50_wildcat_hd_sft_compf_cls_att_avg_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_cls_att_avg_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';

% Results{22} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{23} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{24} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% Results{25} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{26} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{27} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';
% 
% Results{28} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_h_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{29} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_h_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{30} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_h_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% Results{31} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_h_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{32} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_h_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{33} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_h_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';
% 
% Results{34} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{35} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{36} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% Results{37} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{38} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{39} = 'resnet50_wildcat_hd_sft_compf_cls_att_mtad_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';

% Results{10} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g2_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{11} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g2_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{12} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g2_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% 
% Results{13} = 'resnet50_wildcat_hd_gs_A_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{14} = 'resnet50_wildcat_hd_gs_A_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{15} = 'resnet50_wildcat_hd_gs_A_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% 
% Results{16} = 'resnet50_wildcat_hd_gs_G_g1_compf_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';

% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% 
% Results{12} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% 
% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_both_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_both_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% 
% Results{16} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsf_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsf_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% 
% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';

% Results{20} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% 
% Results{22} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{23} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% 
% Results{24} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{25} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% %------------
% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';

% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_both_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_both_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';

% Results{16} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsf_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsf_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';

% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';

% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';

% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_both_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_both_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% 
% Results{16} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_both_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_both_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% 
% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsf_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsf_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% 
% Results{20} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% 
% Results{22} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{23} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_hdsup_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';

% Results{24} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{25} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';

% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{12} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfsup_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% 
% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_4_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_4_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';


% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{12} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';

% Results{16} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{20} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';

% Results{22} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf_alt_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{23} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf_alt_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{24} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf_alt_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{25} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsf_alt_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01_hd';
% 
% Results{26} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{27} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{28} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{29} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{30} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% Results{31} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_hd';

% Results{32} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{33} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01_hd';
% Results{34} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{35} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% 
% Results{36} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup_alt_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{37} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup_alt_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{38} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup_alt_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{39} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nophdsup_alt_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';

% Results{40}  ='resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{41} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01_hd';
% Results{42}  ='resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{43} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd1_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% 
% Results{44}  ='resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{45} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{46}  ='resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% Results{47} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd2_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_hd';

% Results{48}  ='resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd_alt_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{49} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd_alt_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{50}  ='resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd_alt_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{51} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopsfhd_alt_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';


% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{12} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth_alt_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth_alt_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';

% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth1_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{12} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth_alt_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth_alt_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';

% Results{12} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_nopboth2_3_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_hd';
% 
% Results{16} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{18} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';
% 
% Results{19} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{20} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_cls_att_mul_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';

% Results{19} = 'resnet50_wildcat_hd_sf_mul_fs_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{20} = 'resnet50_wildcat_hd_sf_mul_fs_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{21} = 'resnet50_wildcat_hd_sf_mul_fs_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';

% Results{16} = 'resnet50_wildcat_hd_sf_mul_fs_hd_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{17} = 'resnet50_wildcat_hd_sf_mul_fs_hd_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{18} = 'resnet50_wildcat_hd_sf_mul_fs_hd_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';

% Results{16} = 'resnet50_wildcat_hd_sf_mul_fs_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{17} = 'resnet50_wildcat_hd_sf_mul_fs_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{18} = 'resnet50_wildcat_hd_sf_mul_fs_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';

% Results{22} = 'resnet50_wildcat_hd_sf_mul_fs2_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{23} = 'resnet50_wildcat_hd_sf_mul_fs2_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{24} = 'resnet50_wildcat_hd_sf_mul_fs2_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';
% 
% Results{25} = 'resnet50_wildcat_hd_sf_mul_fs2_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{26} = 'resnet50_wildcat_hd_sf_mul_fs2_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{27} = 'resnet50_wildcat_hd_sf_mul_fs2_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% 
% Results{28} = 'resnet50_wildcat_hd_sf_mul_fs2_hd_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{29} = 'resnet50_wildcat_hd_sf_mul_fs2_hd_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{30} = 'resnet50_wildcat_hd_sf_mul_fs2_hd_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';
% 
% Results{31} = 'resnet50_wildcat_hd_sf_mul_fs2_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{32} = 'resnet50_wildcat_hd_sf_mul_fs2_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{33} = 'resnet50_wildcat_hd_sf_mul_fs2_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% 
% 
% Results{16} = 'resnet50_wildcat_hd_sft_compf_cls_att_max_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_cls_att_max_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{18} = 'resnet50_wildcat_hd_sft_compf_cls_att_max_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';
% 
% Results{19} = 'resnet50_wildcat_hd_sft_compf_cls_att_max_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{20} = 'resnet50_wildcat_hd_sft_compf_cls_att_max_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_cls_att_max_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_sft';

% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG16_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG16_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{12} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG16_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG16_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG16_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG16_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01_hd';

% Results{16} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch08';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch08_hd';
% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch07';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch07_hd';
% Results{20} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{22} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{23} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{24} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{25} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbG8_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01_hd';
% % resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch07
% Results{16} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch07';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch07_hd';
% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{20} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{22} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{23} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{24} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{25} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.15_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01_hd';

% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{20} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{22} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{23} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{24} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{25} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_hdsup0.05_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01_hd';

% Results{10} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{12} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_sft';
% Results{16} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_hd';
% Results{18} = 'resnet50_wildcat_hd_sft_compf_cls_att_cat_lr_nb_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_sft';

% Results{16} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{17} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{18} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{19} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{20} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{21} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{22} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% Results{23} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_hd';
% Results{24} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch07';
% Results{25} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA16_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch07_hd';
% 
% Results{26} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA8_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{27} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA8_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{28} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA8_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{29} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_cbA8_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% 
% 
% 
% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{12} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{13} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{14} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% Results{15} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_all_sfhd_new_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_hd';

% Results{10} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_2c_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_sft_compf_att_cls_sep_l_2c_all_sfhd_hdsup0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';

% Results{10} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{12} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';
% Results{13} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{14} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% 
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% 
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{31} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{32} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{33} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% 
% Results{34} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{35} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{36} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{37} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{38} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{39} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';
% Results{40} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{41} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{42} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cls_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';

% Results{13} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{14} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';
% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% 
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';

% Results{10} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{11} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{12} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{13} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{14} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';
% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch14';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch14_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch14_sft';
% 
% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_sft';

% Results{22} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{23} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{24} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% Results{25} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{26} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{27} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_sft';
% Results{28} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% Results{29} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_hd';
% Results{30} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_sft';
% Results{31} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06';
% Results{32} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06_hd';
% Results{33} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06_sft';
% Results{34} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch08';
% Results{35} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch08_hd';
% Results{36} = 'resnet50_wildcat_hd_sf_cat_fs3_bo_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch08_sft';

% Results{10} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{11} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{12} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{13} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{14} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';

% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_3_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% 
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';
% 
% Results{13} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{14} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_sft';
% 
% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% 
% Results{13} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{14} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';
% 
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';

% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';
% 
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{13} = 'resnet50_wildcat_wk_sft_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{14} = 'resnet50_wildcat_wk_sft_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
%
% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{14} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05'; % *******
% Results{15} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';

% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_sft';
% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11_sft';
% 
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';

% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_sft';

% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_sft';
% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_sft';
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch12';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch12_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch12_sft';
% Results{31} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch13';
% Results{32} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch13_hd';
% Results{33} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map1_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch13_sft';
% 
% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_sft';
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch12';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch12_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch12_sft';

% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04_sft';
% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_sft';
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10_sft';

% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06_sft';
% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_hdsf_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10_sft';

% Results{13} = 'resnet50_wildcat_wk_sft_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{14} = 'resnet50_wildcat_wk_sft_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{15} = 'resnet50_wildcat_wk_sft_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{16} = 'resnet50_wildcat_wk_sft_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';

% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{14} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{15} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{16} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{17} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';

% Results{11} = 'resnet50_wildcat_wk_hd_cbG16_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{13} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';

% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_map2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';

% Results{19} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{20} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_hd';
% Results{21} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02_sft';
% Results{22} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{23} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_hd';
% Results{24} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03_sft';
% Results{25} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{26} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_hd';
% Results{27} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04_sft';
% Results{28} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% Results{29} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_hd';
% Results{30} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05_sft';
% Results{31} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06';
% Results{32} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06_hd';
% Results{33} = 'resnet50_wildcat_hd_sf_cat_fs3_sf_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch06_sft';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_alt2_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_alt2_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_alt2_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbG8_alt2_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{14} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{15} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% 
% Results{16} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';

% Results{25} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09';
% Results{26} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_hd';
% Results{27} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_sft';
% Results{28} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10';
% Results{29} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10_hd';
% Results{30} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10_sft';
% Results{31} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch13';
% Results{32} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch13_hd';
% Results{33} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch13_sft';
% Results{34} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch17';
% Results{35} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch17_hd';
% Results{36} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA8_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch17_sft';

% Results{14} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{15} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{16} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_alt2_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_alt2_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_alt2_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att3_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att3_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';

% Results{13} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{14} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_sft';
% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbG16_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_sft';

% Results{13} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{14} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_hd';
% Results{15} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_sft';
% Results{16} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{17} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_hd';
% Results{18} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05_sft';
% Results{19} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{20} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_hd';
% Results{21} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07_sft';
% Results{22} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09';
% Results{23} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_hd';
% Results{24} = 'resnet50_wildcat_wk_compf_cls_att_two_sup_cbA16_cbG8_2_sfhd_0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09_sft';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att8_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att8_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att8_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{13} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att8_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{14} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att8_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{15} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att9_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att9_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att9_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{18} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att9_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att2_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att2_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att2_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att2_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{14} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att3_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{15} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att3_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% 
% Results{16} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att5_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{17} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att5_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{18} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att9_rf0.0_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{19} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_32_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{20} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_32_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{21} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_32_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% 
% Results{13} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_128_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{14} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_128_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{15} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_128_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% 
% Results{16} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_256_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{17} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_256_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{18} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_256_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';

% Results{18} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_16_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{19} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_16_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{20} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_16_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{21} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_16_compf_x_all_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_alt3_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_alt3_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_alt3_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbG8_alt3_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{14} = 'resnet50_wildcat_wk_hd_cbG8_alt3_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{15} = 'resnet50_wildcat_wk_hd_cbG8_alt3_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{16} = 'resnet50_wildcat_wk_hd_cbG8_alt3_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{17} = 'resnet50_wildcat_wk_hd_cbG8_alt3_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09';
% 
% Results{18} = 'resnet50_wildcat_wk_hd_cbG8_alt4_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{19} = 'resnet50_wildcat_wk_hd_cbG8_alt4_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{20} = 'resnet50_wildcat_wk_hd_cbG8_alt4_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{21} = 'resnet50_wildcat_wk_hd_cbG8_alt4_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{22} = 'resnet50_wildcat_wk_hd_cbG8_alt4_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% 
% Results{23} = 'resnet50_wildcat_wk_hd_cbG8_alt5_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{24} = 'resnet50_wildcat_wk_hd_cbG8_alt5_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{25} = 'resnet50_wildcat_wk_hd_cbG8_alt5_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{26} = 'resnet50_wildcat_wk_hd_cbG8_alt5_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{27} = 'resnet50_wildcat_wk_hd_cbG8_alt5_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{28} = 'resnet50_wildcat_wk_hd_cbG8_alt5_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';%
% 
% Results{29} = 'resnet50_wildcat_wk_hd_cbG8_alt6_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{30} = 'resnet50_wildcat_wk_hd_cbG8_alt6_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{31} = 'resnet50_wildcat_wk_hd_cbG8_alt6_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{32} = 'resnet50_wildcat_wk_hd_cbG8_alt6_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{33} = 'resnet50_wildcat_wk_hd_cbG8_alt6_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{34} = 'resnet50_wildcat_wk_hd_cbG8_alt6_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% 
% Results{35} = 'resnet50_wildcat_wk_hd_cbG8_alt7_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{36} = 'resnet50_wildcat_wk_hd_cbG8_alt7_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';
% 
% Results{37} = 'resnet50_wildcat_wk_hd_cbG8_alt8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{38} = 'resnet50_wildcat_wk_hd_cbG8_alt8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{37} = 'resnet50_wildcat_wk_hd_cbG8_alt8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{38} = 'resnet50_wildcat_wk_hd_cbG8_alt8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{39} = 'resnet50_wildcat_wk_hd_cbG8_alt8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch08';
% Results{40} = 'resnet50_wildcat_wk_hd_cbG8_alt8_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch09';
% 
% Results{41} = 'resnet50_wildcat_wk_hd_cbG8_alt9_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{42} = 'resnet50_wildcat_wk_hd_cbG8_alt9_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';

% Results{10} = 'resnet50_wildcat_hd_gs_A16_compf_all_gd_hth0.1_ils_kmax1_kminNone_a0.7_M1_fFalse_dlTrue_one3_224_epoch01';
% Results{11} = 'resnet50_wildcat_hd_gs_A16_compf_all_gd_hth0.1_ils_kmax1_kminNone_a0.7_M1_fFalse_dlTrue_one3_224_epoch02';
% Results{12} = 'resnet50_wildcat_hd_gs_A16_compf_all_gd_hth0.1_ils_kmax1_kminNone_a0.7_M1_fFalse_dlTrue_one3_224_epoch03';

% Results{10} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_nf_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch01';
% Results{11} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_nf_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch02';
% Results{12} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_nf_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch03';
% Results{13} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_nf_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch04';

% Results{10} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_f_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch01';
% Results{11} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_f_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch02';
% Results{12} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_f_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch03';

% Results{10} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_s_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch01';
% Results{11} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_s_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch02';
% Results{12} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_s_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch03';
% Results{13} = 'resnet50_wildcat_hd_rn_cls_att2_norm_g1_compf_all_gd_s_hth0.1_ms_kmax1_kminNone_a0.7_M2_fFalse_dlTrue_one3_224_epoch04';
% 
% Results{10} = 'resnet50_wildcat_hd_gs_G8_g1_compf_all_gd_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{11} = 'resnet50_wildcat_hd_gs_G8_g1_compf_all_gd_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{12} = 'resnet50_wildcat_hd_gs_G8_g1_compf_all_gd_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% 
% Results{13} = 'resnet50_wildcat_hd_gs_A16_g1_compf_all_gd_s_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_s_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_s_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_s2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';

% Results{10} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{11} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% 
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% 
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf3_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf3_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf3_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% % 
% Results{17} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf3_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{18} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf3_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{19} = 'resnet50_wildcat_wk_hd_cbG8_compf_cls_att_gd_nf3_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{20} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_s2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{21} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_s2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{18} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{19} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';

% Results{15} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{18} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';

% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{18} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{19} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';

% Results{10} = 'resnet50_wildcat_hd_gs_A16_g1_compf_all_gd_nf4_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{11} = 'resnet50_wildcat_hd_gs_A16_g1_compf_all_gd_nf4_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';
% Results{12} = 'resnet50_wildcat_hd_gs_A16_g1_compf_all_gd_nf4_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch05';
% 
% Results{13} = 'resnet50_wildcat_hd_gs_G8_g1_compf_all_gd_nf4_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch01';
% Results{14} = 'resnet50_wildcat_hd_gs_G8_g1_compf_all_gd_nf4_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch02';
% Results{15} = 'resnet50_wildcat_hd_gs_G8_g1_compf_all_gd_nf4_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch03';
% Results{16} = 'resnet50_wildcat_hd_gs_G8_g1_compf_all_gd_nf4_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224_epoch04';

% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_f4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{18} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_s4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{19} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_s4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{20} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_s4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% 
% Results{16} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{18} = 'resnet50_wildcat_wk_hd_cbG8_alt_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att2_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att2_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att2_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att2_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{18} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att2_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att3_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att3_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att3_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% % Results{17} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att3_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% % Results{18} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att3_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att3_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00_multiscale';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att3_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01_multiscale';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att3_gd_nf4_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02_multiscale';

% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf2_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_multiscale';

% Results{41} = 'resnet50_wildcat_wk_hd_cbG8_alt9_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{42} = 'resnet50_wildcat_wk_hd_cbG8_alt9_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{43} = 'resnet50_wildcat_wk_hd_cbG8_alt9_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{44} = 'resnet50_wildcat_wk_hd_cbG8_alt9_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{45} = 'resnet50_wildcat_wk_hd_cbG8_alt9_compf_cls_att_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';


% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_ils_kmax1_kminNone_a0.7_M1_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_ils_kmax1_kminNone_a0.7_M1_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_rf0.1_hth0.1_ms_ils_kmax1_kminNone_a0.7_M1_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03_multiscale';
% % 
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00_multiscale';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% 

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% 
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_alt_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch10';
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch11';
% Results{18} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch14';
% Results{19} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch28';
% Results{20} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch30';
% Results{21} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch31';
% Results{22} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_aug_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch33';


% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt131_rf0.01_hth0.01_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% 
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt91_rf0.015_2_hth0.015_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt91_rf0.015_2_hth0.015_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt91_rf0.015_2_hth0.015_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% 
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt91_rf0.02_2_hth0.02_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt91_rf0.02_2_hth0.01_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt91_rf0.02_2_hth0.02_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt131_rf0.01_hth0.01_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt131_rf0.01_hth0.01_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% 
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt149_rf0.015_hth0.015_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt149_rf0.015_hth0.015_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt149_rf0.015_hth0.015_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% 
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_tgt131_rf0.1_hth0.1_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_norms0.2_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_norms0.2_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_tgt85_rf1.0_hth1.0_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one5_224_epoch05';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one0_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one0_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one0_224_epoch05';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb3_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb3_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb3_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb2_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb2_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb2_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_smb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_70_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_70_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_70_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_70_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.5_2_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.5_2_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.5_2_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.5_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.1_2_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.1_2_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.1_2_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.1_2_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.8_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.8_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_sup2_0.8_compf_cls_att_gd_nf4_normTrue_hb_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug6_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug6_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug8_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_500_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_500_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_500_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_300_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

%%==Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_fdim512_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
%%==Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_fdim512_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% 
%%==Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_fdim256_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_fdim256_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_fdim256_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch04';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_fdim256_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_gbvs_rf0.04_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% 
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch05';
% 
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10} = 'resnet50_wildcat_wk_hd_cbA16_sup2_msl_compf_cls_att_gd_nf4_normFalse_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';
% 
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normNdiv_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch06';

% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_sup3_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_sup3_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_sup3_compf_cls_att_gd_nf4_normTrue_hb_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% 
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_tgt91_rf0.02_2_hth0.02_ils_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';

% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normFalse_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normFalse_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normFalse_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_sup2_compf_cls_att_gd_nf4_normFalse_hb_sm_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';

% Results{10}  ='resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_aug2_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{11} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_aug2_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% 
% Results{12} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_aug3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch00';
% Results{13} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_aug3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch01';
% Results{14} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_aug3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{15} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_aug3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch03';
% Results{16} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normFalse_hb_aug3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';
% 
% Results{17} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_aug3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch02';
% Results{18} = 'resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_aug3_nips08_rf0.1_hth0.1_ms_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one2_224_epoch07';

%%
for i = 1:length(Datasets)
    disp(Datasets{i});
    options.Result_path = [base_path, 'WF/Preds/', Datasets{i}, '/'];
    for k = 4:5 %length(Results)
%     for k =length(Results):-1:1    
        % saliency prediction results
        % options.SALIENCY_DIR = [options.Result_path Datasets{i} '/' Results{k} '/'];
        options.SALIENCY_DIR = [options.Result_path Results{k} '/'];
        disp(Results{k});
        
        % dataset path
        %if isequal(Datasets{i}, 'MIT1003')
        %    options.DS_GT_DIR = [options.DS_path, Datasets{i} '/ALLFIXATIONMAPS/'];
        %    options.IMG_DIR = [options.DS_path, Datasets{i}, '/ALLSTIMULI/']; 
        %elseif isequal(Datasets{i}, 'SALICON')
        %    options.DS_GT_DIR = [options.DS_path, Datasets{i} '/maps/val/'];
        %    options.IMG_DIR = [options.DS_path, Datasets{i}, '/images/val/']; % 
        %elseif isequal(Datasets{i}, 'PASCAL-S')
        %    options.DS_GT_DIR = [options.DS_path, Datasets{i} '/maps/'];
        %    options.IMG_DIR = [options.DS_path, Datasets{i}, '/images/'];     
        %end
        if isequal(Datasets{i}, 'MIT1003')
            options.DS_GT_DIR = [options.DS_path, Datasets{i} '/ALLFIXATIONMAPS/'];
            options.IMG_DIR = [options.DS_path, Datasets{i}, '/ALLSTIMULI/']; 
        elseif isequal(Datasets{i}, 'SALICON')
            options.DS_GT_DIR = [options.DS_path, Datasets{i} '/maps/val/'];
            options.IMG_DIR = [options.DS_path, Datasets{i}, '/images/val/']; % 
        elseif isequal(Datasets{i}, 'PASCAL-S')||isequal(Datasets{i}, 'DUTOMRON')||isequal(Datasets{i}, 'toronto')||isequal(Datasets{i}, 'TORONTO')
            options.DS_GT_DIR = [options.DS_path, Datasets{i} '/maps/'];
            options.IMG_DIR = [options.DS_path, Datasets{i}, '/images/'];
        elseif isequal(Datasets{i}, 'CAT2000')
            options.DS_GT_DIR = [options.DS_path, Datasets{i} '/train/maps/'];
            options.IMG_DIR = [options.DS_path, Datasets{i}, '/train/images/*/']; %    
        end
        
                
        %for j =1:2%length(Metrics)
        %    if ~exist([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'file')                 
        %        [result, allMetric, ~] = evaluationFunc_wf(options, Metrics{j});
%       %          [result, allMetric, ~] = evaluationFunc_wf_pascal(options, Metrics{j});
        %        save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
        %        % save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '_all.mat'], 'result');
        %        % std_value = std(allMetric); % calculate std value if you want to
        %    else
        %        load([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat']);
%       %          [result, allMetric, ~] = evaluationFunc_wf(options, Metrics{j});
%       %          save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
        %    end
        %    meanMetric{i}(k,j) = result;
        %    fprintf('%s :%.4f \n', Metrics{j}, result);
        %end
        for j =1:length(Metrics)
            if ~exist([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'file')
                if isequal(Datasets{i}, 'MIT1003')
                    [result, allMetric, ~] = evaluationFunc_wf(options, Metrics{j});
                elseif isequal(Datasets{i}, 'PASCAL-S')||isequal(Datasets{i}, 'DUTOMRON')||isequal(Datasets{i}, 'toronto')||isequal(Datasets{i}, 'TORONTO')
                    [result, allMetric, ~] = evaluationFunc_wf_pascal(options, Metrics{j});
                elseif isequal(Datasets{i}, 'CAT2000')
                    [result, allMetric, ~] = evaluationFunc_wf_cat2000(options, Metrics{j});
                end
                save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
                % save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '_all.mat'], 'result');
                % std_value = std(allMetric); % calculate std value if you want to
            else
                load([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat']);
%                 [result, allMetric, ~] = evaluationFunc_wf(options, Metrics{j});
%                 save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
            end
            meanMetric{i}(k,j) = result;
            fprintf('%s :%.4f \n', Metrics{j}, result);
        end
    end
end

%%
