#!/bin/bash

cd /research/dept2/qxlai/WF/Models/

model1="resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_val_7_sum_three_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224"
target1="val_7_sum_three"
for num in {0..10}
do
    postfix=`printf "_epoch%02d.pt" $num`
    mycmd="ln -s ${model1}${postfix} ${target1}${postfix}"
    eval $mycmd
done

model2="resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_val_8_sum_three_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224"
target2="val_8_sum_three"
for num in {0..7}
do
    postfix=`printf "_epoch%02d.pt" $num`
    mycmd="ln -s ${model2}${postfix} ${target2}${postfix}"
    eval $mycmd
done

model3="resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_val_9_sum_three_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224"
target3="val_9_sum_three"
for num in {0..2}
do
    postfix=`printf "_epoch%02d.pt" $num`
    mycmd="ln -s ${model3}${postfix} ${target3}${postfix}"
    eval $mycmd
done

echo sum three done

model4="resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_val_6_sum_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224"
target4="val_6_sum"
for num in {0..4}
do
    postfix=`printf "_epoch%02d.pt" $num`
    mycmd="ln -s ${model4}${postfix} ${target4}${postfix}"
    eval $mycmd
done

model5="resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_val_7_sum_two_rf0.02_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224"
target5="val_7_sum_two_1"
for num in {0..19}
do
    postfix=`printf "_epoch%02d.pt" $num`
    mycmd="ln -s ${model5}${postfix} ${target5}${postfix}"
    eval $mycmd
done

model6="resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_val_7_sum_two_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224"
target6="val_7_sum_two_2"
for num in {0..19}
do
    postfix=`printf "_epoch%02d.pt" $num`
    mycmd="ln -s ${model6}${postfix} ${target6}${postfix}"
    eval $mycmd
done

model7="resnet50_wildcat_wk_hd_cbA16_compf_cls_att_gd_nf4_normTrue_hb_50_aug7_val_sum_rf0.1_hth0.1_ms4_fdim512_34_cw_sa_art_ftf_2_mres_sp_kmax1_kminNone_a0.7_M4_fFalse_dlTrue_one3_224"
target7="val_sum"
for num in {0..1}
do
    postfix=`printf "_epoch%02d.pt" $num`
    mycmd="ln -s ${model7}${postfix} ${target7}${postfix}"
    eval $mycmd
done

echo sum two done
echo all done
