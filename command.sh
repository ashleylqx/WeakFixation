#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --test-batch 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train-batch 52*4 --test-batch 24*4 # alter training
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train-batch 208 --test-batch 96 # alter training; train 398, test 53, 10/5 log interval


sbatch -p batch_72h --gres=gpu:4 -c 40 --constraint='centos7' my_Batch_0123.cmd
sbatch -p gpu_24h --gres=gpu:4 -c 40 --constraint='centos7' my_Batch_0123.cmd # 113857



