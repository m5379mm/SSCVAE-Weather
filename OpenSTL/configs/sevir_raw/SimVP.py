# SimVP 配置 - SEVIR 雷达预测 (7帧→6帧)
method = 'SimVP'

# 模型参数
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'  # 使用 Inception-UNet 结构
hid_S = 64  # 空间隐藏维度（与原始相同）
hid_T = 256  # 时间隐藏维度（参考原始 sevir 配置）
N_T = 4  # 时间块数量（参考原始）
N_S = 2  # 空间块数量（参考原始）

# 训练参数
lr = 5e-3  # 学习率（参考原始）
batch_size = 8  # 批次大小
drop_path = 0.1  # DropPath 概率
sched = 'onecycle'  # 学习率调度器
warmup_epoch = 0  # 预热 epoch 数（参考原始）
epochs = 200  # 总训练轮数

# 数据参数
pre_seq_length = 7  # 输入帧数
aft_seq_length = 6  # 输出帧数（预测）
in_shape = [7, 1, 128, 128]  # [T, C, H, W]
data_name = 'sevir_raw'
input_frames = 7
output_frames = 6

