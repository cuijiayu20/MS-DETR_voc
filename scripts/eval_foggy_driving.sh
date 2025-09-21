
# 指定包含您训练好的模型的实验目录
# 这里我们指向您训练 voc 数据集的文件夹
EXP_DIR=exps/ms_detr_foggy_voc

# 指定您要测试的具体模型文件名
CHECKPOINT_FILE=checkpoint0019.pth

# 指定要测试的数据集名称
# 注意：在 --eval 模式下，程序会加载验证集(val_dataset)
TEST_DATASET=Foggy_Driving_voc

# 指定您要使用的GPU数量
GPUS=1

# 指定一个空闲的端口号，以防默认端口被占用

# --- 2. 主执行命令 ---

echo "开始在数据集 [${TEST_DATASET}] 上测试模型 [${EXP_DIR}/${CHECKPOINT_FILE}]"

GPUS_PER_NODE=$GPUS ./tools/run_dist_launch.sh $GPUS python -u main.py \
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