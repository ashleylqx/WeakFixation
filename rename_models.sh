#!/bin/bash

cd /research/dept2/qxlai/WF/Models/

model1="resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_val_9_sum_three_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224"
target1="val_9_sum_three"
for num in {0..10}
do
    postfix=`printf "_epoch%02d.pt" $num`
    mycmd='ln -s ${model1}${postfix} ${target1}${postfix}'
    echo $mycmd
done

echo all done
