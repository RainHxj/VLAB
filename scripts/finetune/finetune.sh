#!/usr/bin/env bash
model_size=$1
task=$2

pretrain_dir=./output/pretrain/${model_size}
output_dir=./output/finetune/${model_size}/${task}
config_dir=./config/finetune/${model_size}/${task}.json

python3 -m torch.distributed.launch \
--nnodes 1 \
--nproc_per_node 8 \
--master_addr 127.0.0.1 \
--master_port 1145 \
./finetune/finetune_train.py \
--config $config_dir \
--output_dir $output_dir  \
--pretrain_dir $pretrain_dir
