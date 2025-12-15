# 分辨率为384*384，切分为小patch训练
# 需要归一化[0,1]——统一归一化
# 确认是否需要逐帧输入
# 两个处理方案：1. 忽略闪电部分 2. 用文章方法处理闪电数据（缺点：专业知识；可能需要逐帧输入）
# 384*384，需要3-4个降采样块
import torch
from torch.utils.data import Dataset,random_split
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_mutual_information
import lpips
from torchvision import datasets, transforms
import os
from PIL import Image
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import os
import numpy as np
from PIL import Image
import random
from utils.fixedValues import PREPROCESS_SCALE_SEVIR, PREPROCESS_OFFSET_SEVIR
from matplotlib.colors import ListedColormap, BoundaryNorm

from torch.utils.data import Dataset
import h5py
import scipy
from utils.fixedValues import PREPROCESS_SCALE_SEVIR, PREPROCESS_OFFSET_SEVIR  # 归一化常量
import matplotlib.pyplot as plt

import os
import h5py
import torch
import numpy as np
from sklearn.model_selection import KFold

class SevirTimeTransDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 classes: list = ['ir069', 'ir107', 'vil', 'lght'],
                 seed: int = 42,
                 k_folds: int = 5,
                 fold_index: int = 0,
                 mode: str = 'train',  # train / val / test
                 transform=None,
                 target_size=(128, 128),
                 sequence_length=7):  # 添加 sequence_length 参数
        self.classes = classes
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.target_size = target_size
        self.sequence_length = sequence_length  # 时间序列的长度

        random.seed(seed)

        if mode == 'test':
            data_dir = os.path.join(root_dir, 'test_S')
            self.h5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        else:
            data_dir = os.path.join(root_dir, 'train_all')
            all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            splits = list(kf.split(all_files))
            train_idx, val_idx = splits[fold_index]

            if mode == 'train':
                self.h5_files = [all_files[i] for i in train_idx]
            elif mode == 'val':
                self.h5_files = [all_files[i] for i in val_idx]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        print(f"[{mode.upper()}] Fold {fold_index} → {len(self.h5_files)} files loaded from: {data_dir}")

    def __len__(self):
        # 返回拆分后样本的数量，每个文件被拆分为多个子序列（每个子序列有 sequence_length 帧）
        # 49帧可以切分为 49 // sequence_length 个完整序列
        return len(self.h5_files) * (49 // self.sequence_length)

    def __getitem__(self, index):
        # 计算当前文件和当前帧在子序列中的位置
        num_sequences_per_file = 49 // self.sequence_length
        file_index = index // num_sequences_per_file  # 计算对应文件
        sequence_index = index % num_sequences_per_file  # 计算对应子序列

        h5_file = self.h5_files[file_index]
        start_frame = sequence_index * self.sequence_length  # 计算当前子序列的起始帧
        end_frame = start_frame + self.sequence_length  # 固定长度
        # print(start_frame,end_frame)
        # input()
        with h5py.File(h5_file, 'r') as f:
            data = {}
            for cls in self.classes:
                if cls in f:
                    cls_data = f[cls][:]  # (H, W, 49)
                    data[cls] = cls_data[:, :, start_frame:end_frame]

        # 获取每个通道的数据，并应用归一化处理
        ir069 = data['ir069']
        ir107 = data['ir107']
        lght = data['lght']
        vil = data['vil']

        # 应用归一化
        ir069 = (ir069 + PREPROCESS_OFFSET_SEVIR['ir069']) * PREPROCESS_SCALE_SEVIR['ir069']
        ir107 = (ir107 + PREPROCESS_OFFSET_SEVIR['ir107']) * PREPROCESS_SCALE_SEVIR['ir107']
        lght = (lght + PREPROCESS_OFFSET_SEVIR['lght']) * PREPROCESS_SCALE_SEVIR['lght']
        vil = (vil + PREPROCESS_OFFSET_SEVIR['vil']) * PREPROCESS_SCALE_SEVIR['vil']

        # 执行归一化操作
        ir069 = (ir069 +24.76) / (5.08 +24.76)
        ir107 = (ir107 - (-2.99)) / (2.80+2.99)
        # 特殊归一化处理 lght：非 0 值除以 (281 + 0.9)
        vil = (vil +0.7035) / (4.6395+0.7035)

        lght_mask = lght != 0  # 创建非 0 值的掩码
        lght_nonzero = lght[lght_mask]  # 提取非 0 值
        if len(lght_nonzero) > 0:
            # print(123789)
            lght[lght_mask] = (lght_nonzero / 281.0)*0.1 + 0.9  # 非 0 值归一化


        # 转换为 Tensor
        ir069 = torch.from_numpy(np.array(ir069).astype(np.float32))
        ir107 = torch.from_numpy(np.array(ir107).astype(np.float32))
        lght = torch.from_numpy(np.array(lght).astype(np.float32))
        vil = torch.from_numpy(np.array(vil).astype(np.float32))

        ir069 = ir069.permute(2, 0, 1).unsqueeze(0)  # [1, 49, 192, 192]

        # 插值操作
        ir069 = F.interpolate(ir069, size=self.target_size, mode='bilinear', align_corners=False)  # [1, 49, 128, 128]

        # 转换回原来的形状
        ir069 = ir069.squeeze(0).permute(1, 2, 0)  # [128, 128, 49]

        ir107 = ir107.permute(2, 0, 1).unsqueeze(0)  # [1, 49, 192, 192]

        # 插值操作
        ir107 = F.interpolate(ir107, size=self.target_size, mode='bilinear', align_corners=False)  # [1, 49, 128, 128]

        # 转换回原来的形状
        ir107 = ir107.squeeze(0).permute(1, 2, 0)  # [128, 128, 49]
        # print(ir107.size())

        lght= lght.permute(2, 0, 1).unsqueeze(0)  # [1, 49, 192, 192]

        # 插值操作
        #lght = F.interpolate(lght, size=self.target_size, mode='bilinear', align_corners=False)  # [1, 49, 128, 128]
        lght = F.interpolate(lght, size=self.target_size, mode='nearest', align_corners=None)  # [1, 49, 128, 128]
        # 后处理选择最大值（可选，按需调整窗口）
        # print(lght.min(),lght.max())
        lght = torch.max_pool2d(lght, kernel_size=3, stride=1, padding=1)  # 保留局部最大值

        # 转换回原来的形状
        lght= lght.squeeze(0).permute(1, 2, 0)  # [128, 128, 49]

        vil = vil.permute(2, 0, 1).unsqueeze(0)  # [1, 49, 192, 192]

        # 插值操作
        vil = F.interpolate(vil, size=self.target_size, mode='bilinear', align_corners=False)  # [1, 49, 128, 128]

        # 转换回原来的形状
        vil = vil.squeeze(0).permute(1, 2, 0)  # [128, 128, 49]

        # 合并三个通道
        input_data = torch.stack([ir069, ir107, lght], dim=0)  # (3, 49, H, W)
        vil = vil.unsqueeze(0)

        return input_data, vil  # 返回堆叠后的输入数据和 vil 通道用于重建损失

class SevirTransDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 classes: list = ['ir069', 'ir107', 'vil', 'lght'],
                 seed: int = 42,
                 k_folds: int = 5,
                 fold_index: int = 0,
                 mode: str = 'train',  # train / val / test
                 transform=None,
                 target_size=(128, 128)):

        self.classes = classes
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.target_size = target_size

        random.seed(seed)

        if mode == 'test':
            #data_dir = os.path.join(root_dir, 'test_subset_test')
            data_dir = os.path.join(root_dir, 'test_s')
            self.h5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        else:
            data_dir = os.path.join(root_dir, 'train_all_subsubset')
            # data_dir = os.path.join(root_dir, 'ceshi')
            all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            splits = list(kf.split(all_files))
            train_idx, val_idx = splits[fold_index]

            if mode == 'train':
                self.h5_files = [all_files[i] for i in train_idx]
            elif mode == 'val':
                self.h5_files = [all_files[i] for i in val_idx]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        print(f"[{mode.upper()}] Fold {fold_index} → {len(self.h5_files)} files loaded from: {data_dir}")

    def __len__(self):
        return len(self.h5_files) * 49  # 每个文件 49 帧

    def __getitem__(self, index):
        file_index = index // 49
        frame_index = index % 49
        h5_file = self.h5_files[file_index]

        with h5py.File(h5_file, 'r') as f:
            data = {}
            for cls in self.classes:
                if cls in f:
                    cls_data = f[cls][:]  # (H, W, 49)
                    data[cls] = cls_data[:, :, frame_index]

        # 获取每个通道的数据，并应用归一化处理
        ir069 = data['ir069']
        ir107 = data['ir107']
        lght = data['lght']
        vil = data['vil']

        # 应用归一化
        ir069 = (ir069 + PREPROCESS_OFFSET_SEVIR['ir069']) * PREPROCESS_SCALE_SEVIR['ir069']
        ir107 = (ir107 + PREPROCESS_OFFSET_SEVIR['ir107']) * PREPROCESS_SCALE_SEVIR['ir107']
        lght = (lght + PREPROCESS_OFFSET_SEVIR['lght']) * PREPROCESS_SCALE_SEVIR['lght']
        vil = (vil + PREPROCESS_OFFSET_SEVIR['vil']) * PREPROCESS_SCALE_SEVIR['vil']
        # 归一化到 [0, 1] 范围
        #print(ir069.min(),ir069.max() - ir069.min())
        #print(ir107.min(),ir107.max() - ir107.min())
        ir069 = (ir069 - ir069.min()) / (ir069.max() - ir069.min())
        ir107 = (ir107 - ir107.min()) / (ir107.max() - ir107.min())
        vil = vil/255
        # 假设 lght 是一个 numpy 数组

        # 执行归一化操作，先计算最大值和最小值
        lght_min = lght.min()
        lght_max = lght.max()

        # 检查最大值是否等于最小值，防止除以零
        if lght_min == lght_max:
            # 如果最大值和最小值相同（意味着数据中所有值相等），可以直接将其设置为零或其他值
            lght = np.zeros_like(lght)  # 或者选择其他默认值
            #print(lght_min)
            #print("Warning: lght data has no variation, setting all values to 0.")
        else:
            # 正常归一化
            lght = (lght - lght_min) / (lght_max - lght_min)

        ir069 = torch.from_numpy(np.array(ir069).astype(np.float32))
        ir107 = torch.from_numpy(np.array(ir107).astype(np.float32))
        lght = torch.from_numpy(np.array(lght).astype(np.float32))
        vil = torch.from_numpy(np.array(vil).astype(np.float32))

        # 使用最邻近插值进行尺寸调整
        ir069 = F.interpolate(ir069.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        ir107 = F.interpolate(ir107.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        lght = F.interpolate(lght.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        vil = F.interpolate(vil.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)

        # 合并三个通道
        input_data = torch.stack([ir069, ir107, lght], dim=0)  # (3, H, W)

        # 如果有 transform（如归一化），应用它
        if self.transform:
            input_data = self.transform(input_data)

        return input_data, vil  # 返回堆叠后的输入数据和 vil 通道用于重建损失 reconstruction loss

class SevirTrans1Dataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 classes: list = ['ir069', 'ir107', 'vil', 'lght'],
                 seed: int = 42,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.2,
                 mode: str = 'train_subset',
                 transform=None,
                 target_size=(192, 192)):
        """
        初始化 SevirDataset 类
        """

        self.classes = classes
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.target_size = target_size

        # 设置随机种子
        random.seed(seed)

        # 获取所有 h5 文件
        path = 'train_subset' if mode in ['train', 'val'] else 'test_subset_test'
        data_dir = os.path.join(root_dir, path)

        # 获取文件列表
        self.h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.h5_files = [os.path.join(data_dir, f) for f in self.h5_files]

        # 根据文件名中的 'R' 或 'S' 分类
        r_files = [f for f in self.h5_files if 'R' in f]  # 假设文件名中有'R'表示R类
        s_files = [f for f in self.h5_files if 'S' in f]  # 假设文件名中有'S'表示S类

        # 计算R类和S类的训练集和验证集大小
        r_train_n = int(len(r_files) * train_ratio)
        r_val_n = len(r_files) - r_train_n
        s_train_n = int(len(s_files) * train_ratio)
        s_val_n = len(s_files) - s_train_n

        # 根据数据类别划分训练集和验证集
        if mode == 'train':
            # R类和S类分别按比例取训练集文件
            self.h5_files = r_files[:r_train_n] + s_files[:s_train_n]
        elif mode == 'val':
            # R类和S类分别按比例取验证集文件
            self.h5_files = r_files[r_train_n:] + s_files[s_train_n:]
        elif mode == 'test':
            self.h5_files = self.h5_files  # 测试集使用所有文件
        else:
            raise ValueError(f"mode: {mode} isn't allowed.")

        
        #print(f"[DEBUG] Found {len(self.h5_files)} files for mode '{mode}'")

    def __len__(self):
        return len(self.h5_files) * 49  # 每个文件有49帧

    def __getitem__(self, index):
        # 获取对应的 h5 文件
        file_index = index // 49  # 每个文件有49帧
        frame_index = index % 49  # 每个文件的具体帧
        
        h5_file = self.h5_files[file_index]
        
        # Debugging file path
        file_path = h5_file
        #print(f"[DEBUG] Attempting to open file: {file_path}")

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        # 打开 .h5 文件
        with h5py.File(file_path, 'r') as f:
            data = {}
            for cls in self.classes:
                if cls in f:
                    cls_data = f[cls][:]  # 获取整个通道数据
                    # 获取每帧的单独数据
                    frame_data = cls_data[:, :, frame_index]
                    data[cls] = frame_data  # 不进行额外的归一化
                # 获取每个通道的数据，并应用归一化处理

        ir069 = data['ir069']
        ir107 = data['ir107']
        lght = data['lght']
        vil = data['vil']

        # 应用归一化
        ir069 = (ir069 + PREPROCESS_OFFSET_SEVIR['ir069']) * PREPROCESS_SCALE_SEVIR['ir069']
        ir107 = (ir107 + PREPROCESS_OFFSET_SEVIR['ir107']) * PREPROCESS_SCALE_SEVIR['ir107']
        lght = (lght + PREPROCESS_OFFSET_SEVIR['lght']) * PREPROCESS_SCALE_SEVIR['lght']
        vil = (vil + PREPROCESS_OFFSET_SEVIR['vil']) * PREPROCESS_SCALE_SEVIR['vil']
        # 归一化到 [0, 1] 范围
        #print(ir069.min(),ir069.max() - ir069.min())
        #print(ir107.min(),ir107.max() - ir107.min())
        ir069 = (ir069 - ir069.min()) / (ir069.max() - ir069.min())
        ir107 = (ir107 - ir107.min()) / (ir107.max() - ir107.min())
        vil = (vil-vil.min())/(vil.max()-vil.min())
        # 假设 lght 是一个 numpy 数组

        # 执行归一化操作，先计算最大值和最小值
        lght_min = lght.min()
        lght_max = lght.max()

        # 检查最大值是否等于最小值，防止除以零
        if lght_min == lght_max:
            # 如果最大值和最小值相同（意味着数据中所有值相等），可以直接将其设置为零或其他值
            lght = np.zeros_like(lght)  # 或者选择其他默认值
            #print(lght_min)
            #print("Warning: lght data has no variation, setting all values to 0.")
        else:
            # 正常归一化
            lght = (lght - lght_min) / (lght_max - lght_min)


        # 输出最大值和最小值
        # print(f"ir069 min: {ir069.min().item()}, ir069 max: {ir069.max().item()}")
        # print(f"ir107 min: {ir107.min().item()}, ir107 max: {ir107.max().item()}")
        #print(f"lght min: {lght.min().item()}, lght max: {lght.max().item()}")
        # print(f"vil min: {vil.min().item()}, vil max: {vil.max().item()}")
        # Resize images and stack them
        ir069 = torch.from_numpy(np.array(ir069).astype(np.float32))
        ir107 = torch.from_numpy(np.array(ir107).astype(np.float32))
        lght = torch.from_numpy(np.array(lght).astype(np.float32))
        vil = torch.from_numpy(np.array(vil).astype(np.float32))

        # 使用最邻近插值进行尺寸调整
        ir069 = F.interpolate(ir069.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        ir107 = F.interpolate(ir107.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        lght = F.interpolate(lght.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        vil = F.interpolate(vil.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)

        # 合并三个通道
        input_data = torch.stack([ir069, ir107, lght], dim=0)  # (3, H, W)

        # 如果有 transform（如归一化），应用它
        if self.transform:
            input_data = self.transform(input_data)

        return input_data, vil  # 返回堆叠后的输入数据和 vil 通道用于重建损失 reconstruction loss

class SevirTransDataset_comfirm(Dataset):
    def __init__(self,
                 root_dir: str,
                 classes: list = ['ir069', 'ir107', 'vil', 'lght'],
                 seed: int = 42,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.2,
                 mode: str = 'train_subset',
                 transform=None,
                 target_size=(192, 192)):
        """
        初始化 SevirDataset 类
        """

        self.classes = classes
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.target_size = target_size

        # 设置随机种子
        random.seed(seed)

        # 获取所有 h5 文件
        path = 'train_subset_large' if mode in ['train', 'val'] else 'test_subset_test'
        data_dir = os.path.join(root_dir,path)
        self.h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.h5_files = [os.path.join(data_dir, f) for f in self.h5_files]
        print(f"[DEBUG] Found {len(self.h5_files)} .h5 files in {data_dir}")
        # 打印文件列表以调试
        #print(f"Total .h5 files found: {len(self.h5_files)}")
        #print(f"Files: {self.h5_files}")
        # 划分数据集
        n = len(self.h5_files)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)

        if mode == 'train':
            self.h5_files = self.h5_files[:train_n]
        elif mode == 'val':
            self.h5_files = self.h5_files[train_n:]
        elif mode == 'test':
            self.h5_files = self.h5_files
        else:
            raise ValueError(f"mode: {mode} isn't allowed.")

    def __len__(self):
        return len(self.h5_files) * 49  # 每个文件有49帧

    def __getitem__(self, index):
        # 获取对应的 h5 文件
        file_index = index // 49  # 每个文件有49帧
        frame_index = index % 49  # 每个文件的具体帧
        
        h5_file = self.h5_files[file_index]
        file_path = h5_file

        # 打开 .h5 文件
        with h5py.File(file_path, 'r') as f:
            data = {}
            for cls in self.classes:
                # 获取每个类别的数据
                if cls in f:
                    cls_data = f[cls][:]  # 获取整个通道数据
                    # 获取每帧的单独数据
                    frame_data = cls_data[:, :, frame_index]
                    data[cls] = frame_data  # 不进行额外的归一化

        # 获取每个通道的数据，并应用归一化处理
        ir069 = data['ir069']
        ir107 = data['ir107']
        lght = data['lght']
        vil = data['vil']

        # 应用归一化
        ir069 = (ir069 + PREPROCESS_OFFSET_SEVIR['ir069']) * PREPROCESS_SCALE_SEVIR['ir069']
        ir107 = (ir107 + PREPROCESS_OFFSET_SEVIR['ir107']) * PREPROCESS_SCALE_SEVIR['ir107']
        lght = (lght + PREPROCESS_OFFSET_SEVIR['lght']) * PREPROCESS_SCALE_SEVIR['lght']
        vil = (vil + PREPROCESS_OFFSET_SEVIR['vil']) * PREPROCESS_SCALE_SEVIR['vil']
        # 归一化到 [0, 1] 范围
        #print(ir069.min(),ir069.max() - ir069.min())
        #print(ir107.min(),ir107.max() - ir107.min())
        ir069 = (ir069 - ir069.min()) / (ir069.max() - ir069.min())
        ir107 = (ir107 - ir107.min()) / (ir107.max() - ir107.min())
        vil = (vil-vil.min())/(vil.max()-vil.min())
        # 假设 lght 是一个 numpy 数组

        # 执行归一化操作，先计算最大值和最小值
        lght_min = lght.min()
        lght_max = lght.max()

        # 检查最大值是否等于最小值，防止除以零
        if lght_min == lght_max:
            # 如果最大值和最小值相同（意味着数据中所有值相等），可以直接将其设置为零或其他值
            lght = np.zeros_like(lght)  # 或者选择其他默认值
            #print(lght_min)
            #print("Warning: lght data has no variation, setting all values to 0.")
        else:
            # 正常归一化
            lght = (lght - lght_min) / (lght_max - lght_min)


        # 输出最大值和最小值
        # print(f"ir069 min: {ir069.min().item()}, ir069 max: {ir069.max().item()}")
        # print(f"ir107 min: {ir107.min().item()}, ir107 max: {ir107.max().item()}")
        #print(f"lght min: {lght.min().item()}, lght max: {lght.max().item()}")
        # print(f"vil min: {vil.min().item()}, vil max: {vil.max().item()}")
        # Resize images and stack them
        ir069 = torch.from_numpy(np.array(ir069).astype(np.float32))
        ir107 = torch.from_numpy(np.array(ir107).astype(np.float32))
        lght = torch.from_numpy(np.array(lght).astype(np.float32))
        vil = torch.from_numpy(np.array(vil).astype(np.float32))

        # 使用最邻近插值进行尺寸调整
        ir069 = F.interpolate(ir069.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        ir107 = F.interpolate(ir107.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        lght = F.interpolate(lght.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0).squeeze(0)
        vil = F.interpolate(vil.unsqueeze(0).unsqueeze(0), size=self.target_size, mode='bilinear').squeeze(0)

        # 合并三个通道
        input_data = torch.stack([ir069, ir107, lght], dim=0)  # (3, H, W)

        # 如果有 transform（如归一化），应用它
        if self.transform:
            input_data = self.transform(input_data)

        return input_data, vil  # 返回堆叠后的输入数据和 vil 通道用于重建损失 reconstruction loss

if __name__ == "__main__":
    # 设置数据集路径和加载参数
    root_dir = '/root/autodl-tmp/earthformer-satellite-to-radar-main/data'
    dataset = SevirTransDataset(root_dir=root_dir, mode='train', target_size=(192, 192))
    
    # 获取第一个样本
    input_data, vil = dataset[0]  # input_data: [3, H, W]，vil: [H, W]

    # 可视化三个输入通道：ir069, ir107, lght
    titles = ['ir069', 'ir107', 'lght']
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 4, i + 1)
        plt.imshow(input_data[i].cpu().numpy(), cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    levels = np.array([0.0, 16.0/255, 31.0/255, 59.0/255, 74.0/255, 100.0/255, 133.0/255, 160.0/255, 181.0/255, 219.0/255, 255.0/255])
    
    VIL_COLORS = [
        [0, 0, 0],
        [0.30196, 0.30196, 0.30196],
        [0.15686, 0.74509, 0.15686],
        [0.09803, 0.58823, 0.09803],
        [0.03921, 0.41176, 0.03921],
        [0.03921, 0.29411, 0.03921],
        [0.96078, 0.96078, 0.0],
        [0.92941, 0.67451, 0.0],
        [0.94117, 0.43137, 0.0],
        [0.62745, 0.0, 0.0],
        [0.90588, 0.0, 1.0]
    ]
    cmap = ListedColormap(VIL_COLORS)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # 可视化目标 VIL 图像，使用真实雷达色阶
    plt.subplot(1, 4, 4)
    vil_np = vil.squeeze().cpu().numpy()
    plt.imshow(vil_np, cmap=cmap, norm=norm)

    plt.axis('off')

    plt.tight_layout()
    plt.savefig("1.png")
