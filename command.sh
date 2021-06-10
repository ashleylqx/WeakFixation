#!/bin/bash

# batch_size for all the gpus
CUDA_VISIBLE_DEVICES=0 python main.py --test-batch 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train-batch 52*4 --test-batch 24*4 # alter training
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train-batch 208 --test-batch 96 # alter training; train 398, test 53, 10/5 log interval


sbatch -p batch_72h --gres=gpu:4 -c 40 --constraint='centos7' my_Batch_0123.cmd
sbatch -p gpu_24h --gres=gpu:4 -c 40 --constraint='centos7' my_Batch_0123.cmd # 113857

# batch_size for single gpu card
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train-batch 48 --test-batch 16 --batch-size 4 # multi_scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train-batch 48 --test-batch 24 --batch-size 24 # single_scale test

# 210424_sgd
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.1 --n_epochs 90 --schedule 31 61 # single_scale test
# 210424_sgd_2
#     [27] Saving model with nss 1.4748 ...
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model_name 210424_sgd_2 --resume --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test

# ubuntu18 not suitable for env torch35_wk
sbatch -p batch_72h --gres=gpu:4 -c 40 --constrain highcpucount wk_Batch.cmd
sbatch -p batch_72h --gres=gpu:4 -c 40 --constrain highcpucount,centos7 wk_Batch.cmd
srun -p gpu_24h --gres=gpu:4 -c 40 --constrain highcpucount,centos7 --pty bash
# 135686

tensorboard --logdir=/research/dept2/qxlai/WF/log2/basemodel_alt_210502_adam --port=6001 --bind_all

# (1) basemodel **************************
    # log dir basemodel_21_04_27, wrongly named
CUDA_VISIBLE_DEVICES=0 python main.py --phase train_cw_aug --model_name basemodel_210428_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug --model_name basemodel_210428_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug --model_name basemodel_210428_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
    # [wrong] submit 136944, substract mean for nips08

    # [wrong] submit 138191, substract mean for nips08
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug --model_name basemodel_210503_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 200 --gamma 0.5 --schedule 25 50 75 100 125 150 175 200  # single_scale test

    # **** [good] *****
    # prior_maps do not substract mean for nips08, same training strategy with basemodel_210428_sgd
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug --model_name basemodel_210504_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61  # single_scale test
    # *** Done *** using train COCO_ALL; resume submit 146099
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug --model_name basemodel_210528_all_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61  # single_scale test
    # submit 138600, do not substract mean for nips08
    # 0.01-->0.1-->decay [seems not good]
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug --model_name basemodel_210511_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61  # single_scale test

    # [wrong]
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug --model_name basemodel_210428_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.0001 # single_scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug --resume basemodel_210428_adam --model_name basemodel_210428_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.0001 # single_scale test
  # do not sub mean for nips08
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug --model_name basemodel_210508_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.0001 # single_scale test

# (2) basemodel_alt *************************
CUDA_VISIBLE_DEVICES=0 python main.py --phase train_cw_alt_alpha --init_model basemodel_210428_sgd --model_name basemodel_alt_210502_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.0001 # single_scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_alt_alpha --init_model basemodel_210428_sgd --model_name basemodel_alt_210502_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.0001 # not good ...
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_alt_alpha --init_model basemodel_210428_sgd --model_name basemodel_alt_210502_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

    # **** [good] ****
    # stopped, becasue init is not as good as sgd; it has a long way to go
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_alt_alpha --init_model basemodel_210508_adam --model_name basemodel_alt_210509_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
    # ** done <choose> ***
    # submit to 140537; finish 181 epoch; best 1.7505
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_alt_alpha --init_model basemodel_210504_sgd --model_name basemodel_alt_210509_adam_2 --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
    # *** Done ***; submit to 146300; submit to 146971;
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_alt_alpha --init_model basemodel_210528_all_sgd --model_name basemodel_alt_210531_all_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
        # *** Done *** (300 epochs) submit to 147441
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_alt_alpha --init_model basemodel_210528_all_sgd --model_name basemodel_alt_210531_all_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 --n_epochs 300 # single_scale test



# (3) sa_art *********************************
    # load model basemodel_alt_210509_adam_2
CUDA_VISIBLE_DEVICES=2,3 python main.py --phase train_cw_aug_sa_art --init_model basemodel_alt_210509_adam_2 --model_name tmp --train-batch 1 --test-batch 1 --batch-size 1 --lr 1e-5 # single_scale test
    # smaller memory, 3357 MiB each
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_art --init_model basemodel_alt_210509_adam_2 --model_name sa_art_210511_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_art --init_model basemodel_alt_210509_adam_2 --model_name sa_art_210511_adam_2 --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-4 # single_scale test
    # not improved?
    # ** done ***
    # this is not improved; do not init nss as the best value next time;
CUDA_VISIBLE_DEVICES=0,1 python main.py --phase train_cw_aug_sa_art --init_model basemodel_alt_210509_adam_2 --model_name sa_art_210514_adam --train-batch 96 --test-batch 48 --batch-size 48 --lr 1e-5 # single_scale test
CUDA_VISIBLE_DEVICES=2,3 python main.py --phase train_cw_aug_sa_art --init_model basemodel_alt_210509_adam_2 --model_name tmp --train-batch 96 --test-batch 48 --batch-size 48 --lr 1e-5 # single_scale test

    # ***Running *** init 1.6821, submit to 147440
CUDA_VISIBLE_DEVICES=0,1 python main.py --phase train_cw_aug_sa_art --init_model basemodel_alt_210531_all_adam --model_name sa_art_210607_all_adam --train-batch 96 --test-batch 48 --batch-size 48 --lr 1e-5 # single_scale test

# (4) sa_sp_fixf *********************************
    # load model
CUDA_VISIBLE_DEVICES=0 python main.py --phase train_cw_aug_sa_sp_fixf --init_model basemodel_alt_210509_adam_2 --model_name tmp --train-batch 1 --test-batch 1 --batch-size 1 --lr 1e-5 # single_scale test
CUDA_VISIBLE_DEVICES=0 python main.py --phase train_cw_aug_sa_sp_fixf --init_model sa_art_210511_adam --model_name sa_sp_fixf_210511_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

    # run this directly yields improvements. we can omit (3) maybe.
      # 3211 MiB each
            # ** done (runtime error after several epochs) ***
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug_sa_sp_fixf --init_model basemodel_alt_210509_adam_2 --model_name sa_sp_fixf_210511_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp_fixf --init_model sa_art_210511_adam --model_name sa_sp_fixf_210511_adam_2 --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
      # ** done *** init nss 1.7342
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp_fixf --init_model sa_art_210514_adam --model_name sa_sp_fixf_210515_adam --train-batch 96 --test-batch 48 --batch-size 48 --lr 1e-5 # single_scale test
      # init nss 1.7751
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug_sa_sp_fixf --init_model sa_art_210514_adam --model_name sa_sp_fixf_210515_adam --train-batch 96 --test-batch 48 --batch-size 48 --lr 1e-5 # single_scale test
      # init nss 1.7771
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug_sa_sp_fixf --init_model sa_art_210514_adam --model_name sa_sp_fixf_210515_adam --train-batch 144 --test-batch 72 --batch-size 72 --lr 1e-5 # single_scale test
      # *** Done *** init nss 1.7342, train COCO_ALL
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp_fixf --init_model sa_art_210514_adam --model_name sa_sp_fixf_210529_all_adam --train-batch 144 --test-batch 72 --batch-size 72 --lr 1e-5 # single_scale test

      # *** Pending *** init nss 1.6912, train COCO_ALL, submit to 147987
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp_fixf --init_model sa_art_210607_all_adam --model_name sa_sp_fixf_210610_all_adam --train-batch 144 --test-batch 72 --batch-size 72 --lr 1e-5 # single_scale test


# (4) sa_sp *********************************
    # load model
CUDA_VISIBLE_DEVICES=0 python main.py --phase train_cw_aug_sa_sp --init_model basemodel_alt_210509_adam_2 --model_name tmp --train-batch 1 --test-batch 1 --batch-size 1 --lr 1e-5 # single_scale test
CUDA_VISIBLE_DEVICES=0 python main.py --phase train_cw_aug_sa_sp --init_model sa_sp_fixf_210511_adam --model_name sa_sp_210511_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

    # ** done *** init nss 1.7802 (cannot exceed 1.7802; gradually becomes worse)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_210516_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_210517_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
    # *** done *** init nss 1.7802
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_210517_adam_2 --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-4 # single_scale test
    # ** done *** init nss 1.7802 (worse than adam, nss drops significantly)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_210516_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-2 --n_epochs 90 --schedule 31 61 # single_scale test


CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp --init_model basemodel_alt_210509_adam_2 --model_name sa_sp_210511_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

    # *** Done *** init nss 1.7802; train COCO_ALL (not good)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_210528_all_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-4 # single_scale test
    # *** Pending *** init nss 1.7802; train COCO_ALL
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_210529_all_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

    # train from scratch
        # 210424_sgd
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.1 --n_epochs 90 --schedule 31 61 # single_scale test
        # 210424_sgd_2
        #     [27] Saving model with nss 1.4748 ...
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --model_name 210424_sgd_2 --resume --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test


# (5) sa_sp_alt *********************************
    # init model 210424_sgd_2 1.4748
CUDA_VISIBLE_DEVICES=1 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model 210424_sgd_2 --model_name tmp --train-batch 1 --test-batch 1 --batch-size 1 --lr 0.0001 # single_scale test
CUDA_VISIBLE_DEVICES=0 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210511_adam --model_name sa_sp_alt_210511_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.0001 # single_scale test

    # try this alt from scratch_trained sa_sp model ...
      # ** done; not converge *** (wrongly use MS_COCO val as train set)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model 210424_sgd_2 --model_name sa_sp_alt_210511_adam --train-batch 36 --test-batch 16 --batch-size 16 --lr 1e-5 # single_scale test
      # ** done ...  (correct MS_COCO train set) [not converge]
      # submit to 141734 for training
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model 210424_sgd_2 --model_name sa_sp_alt_210511_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

      # ** done *** (correct MS_COCO train set)
      # submit to 141466 for resume training [not good]
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model 210424_sgd_2 --model_name sa_sp_alt_210511_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210511_adam --model_name sa_sp_alt_210512_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

      # *** done *** init nss 1.7329; submit to 143496 resume; submit to 144119 resume; submit to 144233 resume
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210517_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
      # *** Done *** init nss 1.7329, init both (this is previous practice !!!)(not so good)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210522_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
      # ! *** done *** init nss 1.7329, aux_maps = aux_maps * ALT_RATIO, init both (this is previous practice !!!); resume 144944, 1.7359
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210523_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test

      # *** done *** init nss 1.7329; submit to 143928 resume (foreget to decay; rerun)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210517_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-2 # single_scale test
      # *** done *** init nss 1.7329;submit to 144120 (not so good)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210520_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-2 --n_epochs 90 --schedule 31 61 # single_scale test

      # *** done *** init nss 1.7802 (not so good?)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_alt_210518_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
      # *** done *** init nss 1.7802, init both, no ALT_RATIO (not good ...)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_alt_210524_adam --train-batch 40 --test-batch 20 --batch-size 20 --lr 1e-5 # single_scale test
      # *** done *** init nss 1.7802, init both, no ALT_RATIO (not good ...)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_fixf_210515_adam --model_name sa_sp_alt_210526_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-2 --n_epochs 90 --schedule 31 61 # single_scale test


# without A'
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_art --model_name sa_210428_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
