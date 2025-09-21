# --- 1. 配置您的测试参数 ---

# 指定包含您想要用来测试的、已经训练好的模型的实验目录
# 例如，如果您想用在 RainyImages 上训练的模型，就设置为 exps/ms_detr_foggy_rainy
EXP_DIR=exps/ms_detr_foggy_voc

# 指定您要测试的具体模型文件名
# 例如 checkpoint.pth (最新的) 或 checkpoint0019.pth (第20轮的)
CHECKPOINT_FILE=checkpoint0019.pth

# 指定要测试的数据集名称
# 这个值会传递给 foggy_driving.py 来加载正确的路径
TEST_DATASET=RTTStest

# 指定您要使用的GPU数量
GPUS=1


# --- 2. 主执行命令 ---

echo "开始在数据集 [${TEST_DATASET}] 上测试模型 [${EXP_DIR}/${CHECKPOINT_FILE}]"

GPUS_PER_NODE=$GPUS  ./tools/run_dist_launch.sh $GPUS python -u main.py \
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