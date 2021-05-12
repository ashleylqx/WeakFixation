#!/bin/bash
SBATCH --job-name=TRAIN
SBATCH --mail-user=qxlai@cse.cuhk.edu.hk
SBATCH --mail-type=ALL
SBATCH --output=/research/dept2/qxlai/weakfixation4/210423_sgd_2.txt ##Do not use "~" point to your home!
SBATCH --gres=gpu:4
SBATCH --constraint='ubuntu18'
cd /research/dept2/qxlai/weakfixation4
source activate torch36_wk
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --train-batch 48 --test-batch 24 --batch 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug --model_name basemodel_210428_sgd --train-batch 48 --test-batch 24 --batch 24 --lr 0.01 --n_epochs 90 --schedule 31 61 # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug --model_name basemodel_210503_sgd --train-batch 48 --test-batch 24 --batch 24 --lr 0.01 --n_epochs 200 --gamma 0.5 --schedule 25 50 75 100 125 150 175 200  # single_scale test
#CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_aug --model_name basemodel_210504_sgd --train-batch 48 --test-batch 24 --batch 24 --lr 0.01 --n_epochs 90 --schedule 31 61  # single_scale test
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --resume --phase train_cw_alt_alpha --init_model basemodel_210504_sgd --model_name basemodel_alt_210509_adam_2 --train-batch 48 --test-batch 24 --batch 24 --lr 0.00001 # single_scale test
