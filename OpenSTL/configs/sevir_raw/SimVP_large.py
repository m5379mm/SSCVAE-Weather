# SimVP Large 配置 - SEVIR 雷达预测 (7帧→6帧)
# 更大的模型容量，可能效果更好但训练更慢
method = 'SimVP'

# 模型参数 - 大模型
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'
hid_S = 64
hid_T = 512  # 更大
N_T = 8      # 更深
N_S = 4      # 更多
groups = 4

# 训练参数
lr = 1e-3    # 保守的学习率
batch_size = 4  # 更小的批次（显存考虑）
drop_path = 0.1
sched = 'onecycle'
warmup_epoch = 5  # 预热
epochs = 200

# 数据参数
pre_seq_length = 7
aft_seq_length = 6
in_shape = [7, 1, 128, 128]
data_name = 'sevir_raw'
input_frames = 7
output_frames = 6

