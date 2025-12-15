import os
import h5py
import numpy as np
from utils.fixedValues import PREPROCESS_SCALE_SEVIR, PREPROCESS_OFFSET_SEVIR
def compute_lght_stats(data_dir):
    """
    统计 SEVIR 数据集中 lght 通道的最大值和最小值。
    
    Args:
        data_dir (str): HDF 文件目录路径。
    
    Returns:
        tuple: (global_max, global_min)
    """
    # 获取所有 HDF 文件
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
    print(f"Found {len(all_files)} HDF files in {data_dir}")

    global_max_lght = -np.inf  # 初始化为负无穷
    global_min_lght = np.inf   # 初始化为正无穷
    file_count = 0

    for h5_file in all_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'lght' in f:
                    ir107 = f['lght'][:]  # 加载 lght 数据 (H, W, T)
                    current_max = np.max(ir107 )
                    current_min = np.min(ir107 )
                    
                    global_max_lght = max(global_max_lght, current_max)
                    global_min_lght = min(global_min_lght, current_min)
                    
                    file_count += 1
                    print(f"Processed {h5_file}: max={current_max}, min={current_min}")
                else:
                    print(f"Skipping {h5_file}: no 'lght' key")
        except Exception as e:
            print(f"Error processing {h5_file}: {e}")

    print(f"\nGlobal stats across {file_count} files:")
    print(f"Max lght: {global_max_lght}")
    print(f"Min lght: {global_min_lght}")
    
    return global_max_lght, global_min_lght

# 使用示例（替换为您的实际路径）
if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/earthformer-satellite-to-radar-main/data/train_all"  # 替换为您的路径
    max_lght, min_lght = compute_lght_stats(data_dir)
# import h5py
# import numpy as np

# # 指定 h5 文件路径（替换为您的实际路径）
# h5_file_path = "/root/autodl-tmp/earthformer-satellite-to-radar-main/data/test"  # 示例路径，请替换

# # 打开并读取 h5 文件
# try:
#     with h5py.File(h5_file_path, 'r') as f:
#         # 打印文件结构
#         print("HDF5 File Structure:")
#         def print_h5_structure(name, obj):
#             if isinstance(obj, h5py.Dataset):
#                 print(f"Dataset: {name}, Shape: {obj.shape}, dtype: {obj.dtype}")
#                 # 修正: 确保括号匹配
#                 data_sample = obj[:min(5, obj.shape[0])]  # 取前 5 行或全部
#                 print(f"Sample data: {data_sample}")
#             elif isinstance(obj, h5py.Group):
#                 print(f"Group: {name}")

#         f.visititems(print_h5_structure)

#         # 打印特定键的数据
#         for key in ['ir069']:
#             if key in f:
#                 data = f[key][:]
#                 print(f"\nData for {key}:")
#                 print(f"Shape: {data.shape}")
#                 print(f"dtype: {data.dtype}")
#                 print(f"Max value: {np.max(data)}")
#                 print(f"Min value: {np.min(data)}")
#                 print(f"Sample data (first 5 elements): {data[:5]}")

# except FileNotFoundError:
#     print(f"Error: File {h5_file_path} not found. Please check the path.")
# except Exception as e:
#     print(f"Error: {e}")
        # lght = lght.permute(2, 0, 1)  # [49, 48, 48]
        # lght = F.interpolate(lght.unsqueeze(0), size=self.target_size, mode='nearest', align_corners=None)  # [1, 49, 128, 128]
        # # 后处理选择最大值（可选，按需调整窗口）
        # lght = torch.max_pool2d(lght, kernel_size=3, stride=1, padding=1)  # 保留局部最大值
        # output_file = os.path.join(self.root_dir, 'output.txt')  # 输出文件路径
        # print(output_file)
        # with open(output_file, 'a') as f:  # 'a' 表示追加模式
        #     f.write(f"File: {os.path.basename(h5_file)}\n")
        #     lght_flat = lght.reshape(-1).cpu().numpy()  # 展平为 1D 数组，移到 CPU
        #     for i in range(0, len(lght_flat), 128):
        #         row = lght_flat[i:i + 128]  # 每行 128 个值
        #         f.write(', '.join(map(str, row)) + '\n')  # 写入一行
        #     f.write("-" * 50 + "\n")  # 分隔符
        # input()