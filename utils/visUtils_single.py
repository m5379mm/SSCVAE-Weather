import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 假设 sample_event 是 DataFrame，它包含了 SEVIR 事件数据
sample_event = pd.DataFrame({
    'img_type': ['lght', 'vis', 'ir069', 'ir107', 'vil'],
    'file_name': ['R18032123577327.h5', 'R18032123577327.h5', 'R18032123577327.h5', 'R18032123577327.h5', 'R18032123577327.h5'],
    'id': [0, 1, 2, 3, 4]
})

# 数据路径
DATA_PATH = '/root/autodl-tmp/earthformer-satellite-to-radar-main/data/train'

# 读取所有通道的数据
vis = read_data_from_h5(sample_event, DATA_PATH, 'vis')
ir069 = read_data_from_h5(sample_event, DATA_PATH, 'ir069')
ir107 = read_data_from_h5(sample_event, DATA_PATH, 'ir107')
vil = read_data_from_h5(sample_event, DATA_PATH, 'vil')
lght = read_data_from_h5(sample_event, DATA_PATH, 'lght')

# 创建一个图形，有5个子图
fig, axs = plt.subplots(1, 5, figsize=(20, 5))

# 假设我们有49帧数据
num_frames = 49  # 假设每个通道有49帧数据
for frame_idx in range(num_frames):
    axs[0].imshow(vis[:, :, frame_idx])
    axs[0].set_title('VIS')

    axs[1].imshow(ir069[:, :, frame_idx])
    axs[1].set_title('IR 6.9')

    axs[2].imshow(ir107[:, :, frame_idx])
    axs[2].set_title('IR 10.7')

    axs[3].imshow(vil[:, :, frame_idx])
    axs[3].set_title('VIL')

    axs[4].imshow(lght[:, :, frame_idx])
    axs[4].set_title('Lightning')

    # 显示当前帧
    plt.pause(0.1)  # 暂停以便更新图像

    # 清除图形，为下一帧准备
    if frame_idx < num_frames - 1:
        for ax in axs:
            ax.clear()

plt.show()
