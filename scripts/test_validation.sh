#!/bin/bash

# --- 您要测试的模型所在的实验目录 ---
EXP_DIR=exps/ms_detr_foggy_voc
# --- 您要测试的 checkpoint 文件名 ---
CHECKPOINT_FILE=checkpoint0004.pth

# --- 要测试的数据集 ---
TEST_DATASET=VOCtest-FOG

GPUS=1
MASTER_PORT=29503 # 使用一个和训练不同的端口，避免冲突

GPUS_PER_NODE=$GPUS MASTER_PORT=$MASTER_PORT ./tools/run_dist_launch.sh $GPUS python -u main.py \
   --output_dir $EXP_DIR/eval_on_$TEST_DATASET \
   --dataset_file foggy_driving \
   --val_dataset $TEST_DATASET \
   --resume $EXP_DIR/$CHECKPOINT_FILE \
   --eval \
   --with_box_refine \
   --two_stage \
   --dim_feedforward 2048 \
   --num_queries 300 \
   --use_ms_detr \
   --use_aux_ffn