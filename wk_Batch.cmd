#!/bin/bash
#SBATCH --job-name=TRAIN
#SBATCH --mail-user=ashleylqx@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/research/dept2/qxlai/weakfixation4/basemodel_alt_210825_adam.txt ##Do not use "~" point to your home!
#SBATCH --gres=gpu:4
#SBATCH --constraint='highcpucount'
cd /research/dept2/qxlai/weakfixation4
source activate torch36
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --train-batch 48 --test-batch 24 --batch 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug --model_name basemodel_210428_sgd --train-batch 48 --test-batch 24 --batch 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug --model_name basemodel_210503_sgd --train-batch 48 --test-batch 24 --batch 24 --lr 0.01 --n_epochs 200 --gamma 0.5 --schedule 25 50 75 100 125 150 175 200  # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug --model_name basemodel_210504_sgd --train-batch 48 --test-batch 24 --batch 24 --lr 0.01 --n_epochs 90 --schedule 31 61  # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_alt_alpha --init_model basemodel_210504_sgd --model_name basemodel_alt_210509_adam_2 --train-batch 48 --test-batch 24 --batch 24 --lr 0.00001 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug_alt_alpha_sa_sp --init_model 210424_sgd_2 --model_name sa_sp_alt_210511_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model 210424_sgd_2 --model_name sa_sp_alt_210511_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210517_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
##CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210517_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-2 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210520_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-2 --n_epochs 90 --schedule 31 61 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug_alt_alpha_sa_sp --init_model sa_sp_210517_adam --model_name sa_sp_alt_210523_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug --model_name basemodel_210528_all_sgd --train-batch 48 --test-batch 24 --batch-size 24 --lr 0.01 --n_epochs 90 --schedule 31 61  # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_alt_alpha --init_model basemodel_210528_all_sgd --model_name basemodel_alt_210531_all_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_art --init_model basemodel_alt_210531_all_adam --model_name sa_art_210607_all_adam --train-batch 96 --test-batch 48 --batch-size 48 --lr 1e-5 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_alt_alpha --init_model basemodel_210528_all_sgd --model_name basemodel_alt_210531_all_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5 --n_epochs 300 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --phase train_cw_aug_sa_sp_fixf --init_model sa_art_210607_all_adam --model_name sa_sp_fixf_210610_all_adam --train-batch 144 --test-batch 72 --batch-size 72 --lr 1e-5 # single_scale test

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_old.py --resume --phase train_cw_alt_alpha --init_model basemodel_210504_sgd --model_name basemodel_alt_210825_adam --train-batch 48 --test-batch 24 --batch-size 24 --lr 1e-5
