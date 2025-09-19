#!/bin/bash
EXP_DIR=exps/ms_detr_foggy_rainy
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 python -u main.py \
   --output_dir $EXP_DIR \
   --with_box_refine \
   --two_stage \
   --dim_feedforward 2048 \
   --epochs 12 \
   --lr_drop 11 \
   --dataset_file foggy_driving \
   --train_dataset RainyImages \
   --val_dataset RainyImages \
   --num_queries 300 \
   --use_ms_detr \
   --use_aux_ffn \
   --cls_loss_coef 1 \
   --o2m_cls_loss_coef 2\
   --resume