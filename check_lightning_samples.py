#!/usr/bin/env python3
"""检查多个样本的闪电数据"""

import h5py
import numpy as np
import os

data_dir = '/root/autodl-tmp/earthformer-satellite-to-radar-main/data/test'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])[:20]  # 检查前20个

print(f"检查 {len(files)} 个文件的闪电数据...\n")

有闪电_count = 0
for i, fname in enumerate(files):
    file_path = os.path.join(data_dir, fname)
    with h5py.File(file_path, 'r') as f:
        lght = f['lght'][:]  # [H, W, T]
        
        # 统计所有帧
        max_val = lght.max()
        nonzero_pct = 100 * np.sum(lght > 0) / lght.size
        
        if max_val > 0:
            有闪电_count += 1
            marker = "⚡"
        else:
            marker = "  "
        
        print(f"{marker} 文件 {i+1:2d}: max={max_val:4.0f}, 非零={nonzero_pct:5.2f}% - {fname}")

print(f"\n总结: {有闪电_count}/{len(files)} 个文件有闪电数据")





