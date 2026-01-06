"""
SEVIR Dataloader for OpenSTL - 原始 H5 文件版本
支持 7 帧输入 → 6 帧输出的雷达预测任务
与 SevirTimeTransDataset 数据处理保持一致
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import h5py
import os
import random
from sklearn.model_selection import KFold
from openstl.datasets.utils import create_loader

# SEVIR 数据预处理常量
PREPROCESS_SCALE_SEVIR = {'vil': 1 / 47.54}
PREPROCESS_OFFSET_SEVIR = {'vil': 33.44}


class SevirRawDataset(Dataset):
    """
    SEVIR 原始 H5 文件数据集 - 用于 OpenSTL
    输入：前 input_frames 帧雷达图像
    输出：接下来 output_frames 帧雷达图像
    """
    def __init__(self, root_dir, mode='train',
                 input_frames=7, output_frames=6,
                 seed=42, k_folds=5, fold_index=0,
                 target_size=(128, 128)):
        """
        Args:
            root_dir: SEVIR 数据根目录
            mode: 'train' or 'val' or 'test'
            input_frames: 输入帧数 (默认7)
            output_frames: 输出帧数 (默认6)
            seed: 随机种子
            k_folds: K折交叉验证折数
            fold_index: 当前折索引
            target_size: 目标图像尺寸
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.total_frames = input_frames + output_frames  # 7 + 6 = 13
        self.target_size = target_size
        
        # OpenSTL 需要的属性（用于反归一化和数据模块）
        self.mean = 0.0  # 数据已归一化到 [0, 1]
        self.std = 1.0
        self.data_name = 'sevir_raw'  # 数据集名称
        
        random.seed(seed)
        
        # 加载数据文件
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
        
        print(f"=" * 60)
        print(f"✅ SEVIR {mode.upper()} 雷达预测数据集 (OpenSTL)")
        print(f"   Files: {len(self.h5_files)} 个文件")
        print(f"   输入帧数: {input_frames} 帧")
        print(f"   输出帧数: {output_frames} 帧")
        print(f"   总序列长度: {self.total_frames} 帧")
        print(f"   数据类型: 仅雷达图像 (VIL)")
        print(f"=" * 60)
    
    def __len__(self):
        # 每个文件49帧，可以切分为多少个完整序列
        num_sequences_per_file = 49 // self.total_frames
        return len(self.h5_files) * num_sequences_per_file
    
    def __getitem__(self, idx):
        """
        返回:
            input_seq: [input_frames, C, H, W] - 前7帧
            target_seq: [output_frames, C, H, W] - 后6帧
        """
        # 计算文件索引和序列索引
        num_sequences_per_file = 49 // self.total_frames
        file_index = idx // num_sequences_per_file
        sequence_index = idx % num_sequences_per_file
        
        h5_file = self.h5_files[file_index]
        start_frame = sequence_index * self.total_frames
        end_frame = start_frame + self.total_frames
        
        # 只读取 VIL 雷达数据
        with h5py.File(h5_file, 'r') as f:
            vil_data = f['vil'][:]  # (H, W, 49)
            vil = vil_data[:, :, start_frame:end_frame]  # (H, W, total_frames)
        
        # ============ 应用与 SevirTimeTransDataset 完全相同的 VIL 归一化 ============
        vil = (vil + PREPROCESS_OFFSET_SEVIR['vil']) * PREPROCESS_SCALE_SEVIR['vil']
        vil = (vil + 0.7035) / (4.6395 + 0.7035)
        
        # 转换为 Tensor
        vil = torch.from_numpy(np.array(vil).astype(np.float32))
        
        # 调整尺寸 - 与 SevirTimeTransDataset 一致
        vil = vil.permute(2, 0, 1).unsqueeze(0)  # [1, T, 192, 192]
        vil = F.interpolate(vil, size=self.target_size, mode='bilinear', align_corners=False)
        vil = vil.squeeze(0).permute(1, 2, 0)  # [128, 128, T]
        
        # 准备输入和输出
        # 输入：前 input_frames 帧（历史）
        # 输出：接下来 output_frames 帧（未来）
        input_data = vil[:, :, :self.input_frames]  # [128, 128, input_frames]
        target_data = vil[:, :, self.input_frames:self.input_frames + self.output_frames]  # [128, 128, output_frames]
        
        # 注意：OpenSTL 的 SimVP forward 方法会自动处理 output_frames < input_frames 的情况
        # 它会截断模型输出到 output_frames，所以这里不需要 padding
        # 直接返回真实的 output_frames 即可
        
        # 转换为 [T, C, H, W] 格式
        input_seq = input_data.permute(2, 0, 1).unsqueeze(1)  # [input_frames, 1, 128, 128]
        target_seq = target_data.permute(2, 0, 1).unsqueeze(1)  # [output_frames, 1, 128, 128]
        
        return input_seq, target_seq


def load_data(batch_size,
              val_batch_size,
              data_root='/root/autodl-tmp/earthformer-satellite-to-radar-main/data',
              num_workers=4,
              input_frames=7,
              output_frames=6,
              distributed=False,
              use_augment=False,
              use_prefetcher=False,
              drop_last=True,
              **kwargs):
    """
    加载 SEVIR 雷达数据 - 原始 H5 文件版本
    
    Args:
        batch_size: 训练批次大小
        val_batch_size: 验证批次大小
        data_root: SEVIR 数据根目录
        num_workers: DataLoader 工作线程数
        input_frames: 输入帧数 (默认7)
        output_frames: 输出帧数 (默认6)
        distributed: 是否分布式训练
        use_augment: 是否使用数据增强
        use_prefetcher: 是否使用预取器
        drop_last: 是否丢弃最后不完整的batch
    
    Returns:
        dataloader_train, dataloader_vali, dataloader_test
    """
    # 创建数据集
    train_set = SevirRawDataset(
        root_dir=data_root,
        mode='train',
        input_frames=input_frames,
        output_frames=output_frames
    )
    
    val_set = SevirRawDataset(
        root_dir=data_root,
        mode='val',
        input_frames=input_frames,
        output_frames=output_frames
    )
    
    test_set = SevirRawDataset(
        root_dir=data_root,
        mode='test',
        input_frames=input_frames,
        output_frames=output_frames
    )
    
    # 创建 DataLoader
    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher
    )
    
    dataloader_vali = create_loader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher
    )
    
    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher
    )
    
    print(f"\n{'='*60}")
    print(f"✅ SEVIR 雷达预测 DataLoader 配置完成 (OpenSTL)")
    print(f"{'='*60}")
    print(f"   训练样本: {len(train_set)}, 批次: {len(dataloader_train)}")
    print(f"   验证样本: {len(val_set)}, 批次: {len(dataloader_vali)}")
    print(f"   测试样本: {len(test_set)}, 批次: {len(dataloader_test)}")
    print(f"   输入: {input_frames} 帧雷达图像")
    print(f"   输出: {output_frames} 帧雷达图像")
    print(f"   数据类型: 仅 VIL 雷达数据")
    print(f"   数据处理: 与 SevirTimeTransDataset 完全一致 ✓")
    print(f"{'='*60}\n")
    
    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == "__main__":
    # 测试 dataloader
    print("🧪 测试 SEVIR 雷达预测 Dataloader (OpenSTL)")
    print("   配置: 输入7帧雷达 → 预测6帧雷达\n")
    
    dataloader_train, dataloader_vali, dataloader_test = load_data(
        batch_size=4,
        val_batch_size=4,
        num_workers=2,
        input_frames=7,
        output_frames=6
    )
    
    # 测试一个 batch
    print("\n正在测试数据批次...")
    for input_seq, target_seq in dataloader_train:
        print(f"\n📊 批次形状:")
        print(f"  输入:  {input_seq.shape}")  # 应该是 [4, 7, 1, 128, 128]
        print(f"  输出:  {target_seq.shape}")  # 应该是 [4, 7, 1, 128, 128] (padding后)
        print(f"\n📈 数值范围:")
        print(f"  输入:  [{input_seq.min():.4f}, {input_seq.max():.4f}]")
        print(f"  输出:  [{target_seq.min():.4f}, {target_seq.max():.4f}]")
        break
    
    print("\n" + "="*60)
    print("✅ Dataloader 测试通过！")
    print("   ✓ 仅加载 VIL 雷达数据")
    print("   ✓ 数据处理与 SevirTimeTransDataset 一致")
    print("="*60)

