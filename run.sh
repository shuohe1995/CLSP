#!/bin/bash
dataset=$2 # 'cifar100'/'voc'/'tiny-imagenet'
pr=$3 # 0.0/0.4/0.6/-1.0  0.0 -> LD / -1.0 -> ID
ir=$4 # 0.0/0.01/0.02
tau=$5 # 0.2 0.4 0.6
k=$6 # 5 50 150

model_name='blip2' # blip2/clip/ablef/resnet18_c/resnet18_i/resne18_s
model_type='pretrain' # blip2 -> pretrain; albef -> base; clip -> ViT-B-32
seed=7438

exp_name='test'

time=$(date +%F)
file_path="./output_log/${dataset}/${time}_pr=${pr}_ir=${ir}_tau=${tau}_k=${k}_model=${model_name}_seed=${seed}_${exp_name}.log"


CUDA_VISIBLE_DEVICES=$1 nohup python -u main.py --dataset ${dataset} --partial_rate ${pr} --imb_rate ${ir} --k ${k} --tau ${tau} --model_name ${model_name} --model_type ${model_type} --seed ${seed} --gpu 0 --save > ${file_path} 2>&1 &

