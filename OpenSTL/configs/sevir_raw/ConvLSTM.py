# ConvLSTM 配置 - SEVIR 雷达预测 (7帧→6帧)
method = 'ConvLSTM'

# 模型参数
num_layers = 4  # LSTM 层数
num_hidden = [64, 64, 64, 64]  # 每层隐藏单元数
filter_size = 5  # 卷积核大小
stride = 1
layer_norm = True  # 使用 Layer Normalization

# 训练参数
lr = 1e-3  # 学习率
batch_size = 8  # 批次大小
sched = 'onecycle'  # 学习率调度器
warmup_epoch = 5  # 预热 epoch 数
epochs = 200  # 总训练轮数

# 数据参数
pre_seq_length = 7  # 输入帧数
aft_seq_length = 6  # 输出帧数（预测）
in_shape = [7, 1, 128, 128]  # [T, C, H, W]
data_name = 'sevir_raw'
input_frames = 7
output_frames = 6

