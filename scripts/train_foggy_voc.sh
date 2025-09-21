#!/bin/bash

# 1. 定义实验输出目录
EXP_DIR=exps/ms_detr_foggy_voc

# 2. 定义您要使用的GPU总数
GPUS=1

# --- 以下是主执行命令 ---

# GPUS_PER_NODE=$GPUS: 告诉启动器每台机器用1个GPU
# ./tools/run_dist_launch.sh $GPUS: 告诉启动器总共只启动1个进程
GPUS_PER_NODE=$GPUS ./tools/run_dist_launch.sh $GPUS python -u main.py \
   --output_dir $EXP_DIR \
   --with_box_refine \
   --two_stage \
   --dim_feedforward 2048 \
   --epochs 50 \
   --lr_drop 40 \
   --dataset_file foggy_driving \
   --train_dataset VOC2007-FOG \
   --val_dataset VOCtest-FOG \
   --num_queries 300 \
   --use_ms_detr \
   --use_aux_ffn \
   --cls_loss_coef 1 \
   --o2m_cls_loss_coef 2 \
   --eval_every 10 \
   --batch_size 4 \
   --resume /home/hello/cuijiayu/yolo/MSDETR/ms_detr_300.pth