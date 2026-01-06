#!/bin/bash

# OpenSTL SEVIR 雷达预测训练脚本
# 预测任务：使用过去 7 帧预测未来 6 帧
# 支持多种模型：SimVP, ConvLSTM, PredRNN 等

echo "=========================================="
echo "OpenSTL SEVIR 雷达预测训练"
echo "=========================================="
echo ""

# 配置参数
DATA_ROOT="/root/autodl-tmp/earthformer-satellite-to-radar-main/data"
RESULT_DIR="/root/autodl-tmp/results/OpenSTL"
METHOD="${1:-SimVP}"  # 默认使用 SimVP，可通过参数指定其他模型
BATCH_SIZE=8  # 从 8 改为 4，避免 GPU OOM
EPOCHS=200
NUM_WORKERS=4

echo "配置信息:"
echo "  模型: $METHOD"
echo "  数据根目录: $DATA_ROOT"
echo "  结果保存: $RESULT_DIR"
echo "  输入帧数: 7"
echo "  输出帧数: 6"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo ""

# 开始训练
cd /root/autodl-tmp/Sevir/OpenSTL
export PYTHONPATH=/root/autodl-tmp/Sevir/OpenSTL:$PYTHONPATH

python tools/train.py \
    --dataname sevir_raw \
    --method $METHOD \
    --config_file configs/sevir_raw/${METHOD}.py \
    --data_root $DATA_ROOT \
    --res_dir $RESULT_DIR \
    --ex_name sevir_${METHOD}_7to6 \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --epoch $EPOCHS \
    --num_workers $NUM_WORKERS \
    --seed 42

echo ""
echo "=========================================="
echo "训练完成！"
echo "结果保存在: $RESULT_DIR/sevir_${METHOD}_7to6/"
echo "=========================================="

