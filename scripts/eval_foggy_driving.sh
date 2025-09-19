#!/bin/bash
EXP_DIR=exps/ms_detr_foggy_rainy # Or snowy, or voc
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 python -u main.py \
    --dataset_file foggy_driving \
    --test_dataset Foggy_Driving_voc \
    --with_box_refine \
    --two_stage \
    --dim_feedforward 2048 \
    --num_queries 300 \
    --use_ms_detr \
    --use_aux_ffn \
    --resume $EXP_DIR/checkpoint.pth \
    --eval