#!/bin/bash
SBATCH --job-name=TRAIN
SBATCH --mail-user=qxlai@cse.cuhk.edu.hk
SBATCH --mail-type=ALL
SBATCH --output=/research/dept2/qxlai/weakfixation4/0123.txt ##Do not use "~" point to your home!
SBATCH --gres=gpu:4
SBATCH --constraint='centos7'
cd /research/dept2/qxlai/weakfixation4/
source activate torch36_wk
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train-batch 208  --test-batch 96
