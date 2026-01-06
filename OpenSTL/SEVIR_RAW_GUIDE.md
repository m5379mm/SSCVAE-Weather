# OpenSTL SEVIR 雷达预测 - 快速开始指南

## 📋 任务说明

使用 OpenSTL 框架进行 SEVIR 雷达图像预测：
- **输入**：前 7 帧 VIL 雷达图像
- **输出**：接下来 6 帧 VIL 雷达图像
- **数据类型**：仅使用雷达数据，不加载卫星数据
- **数据处理**：与 `SevirTimeTransDataset` 的 VIL 处理完全一致
- **支持模型**：SimVP, ConvLSTM, PredRNN, PhyDNet 等

## 🚀 快速开始

### 安装依赖

```bash
cd /root/autodl-tmp/Sevir/OpenSTL
pip install -r requirements.txt
pip install scikit-learn  # 如果未安装
```

### 训练模型

#### 方法 1: 使用训练脚本（推荐）

```bash
cd /root/autodl-tmp/Sevir/OpenSTL
chmod +x train_sevir_raw.sh

# 训练 SimVP 模型（默认）
./train_sevir_raw.sh

# 训练 ConvLSTM 模型
./train_sevir_raw.sh ConvLSTM

# 训练 PredRNN 模型
./train_sevir_raw.sh PredRNN
```

#### 方法 2: 直接运行 Python

```bash
cd /root/autodl-tmp/Sevir/OpenSTL

# SimVP
python tools/train.py \
    --dataname sevir_raw \
    --method SimVP \
    --config_file configs/sevir_raw/SimVP.py \
    --data_root /root/autodl-tmp/earthformer-satellite-to-radar-main/data \
    --res_dir /root/autodl-tmp/results/OpenSTL \
    --ex_name sevir_SimVP_7to6 \
    --batch_size 8 \
    --epochs 200

# ConvLSTM
python tools/train.py \
    --dataname sevir_raw \
    --method ConvLSTM \
    --config_file configs/sevir_raw/ConvLSTM.py \
    --data_root /root/autodl-tmp/earthformer-satellite-to-radar-main/data \
    --res_dir /root/autodl-tmp/results/OpenSTL \
    --ex_name sevir_ConvLSTM_7to6 \
    --batch_size 8 \
    --epochs 200
```

## 📊 数据配置

### 数据加载
- **数据集类**：`SevirRawDataset` (在 `openstl/datasets/dataloader_sevir_raw.py`)
- **数据源**：与 `SevirTimeTransDataset` 相同的原始 H5 文件
- **归一化**：完全复刻 `SevirTimeTransDataset` 的处理逻辑

### 数据形状
```
输入: [Batch, 7, 1, 128, 128]  # 7帧 VIL 雷达数据
输出: [Batch, 7, 1, 128, 128]  # 7帧（6帧真实 + 1帧padding）
```

### 序列划分
- 每个 H5 文件包含 49 帧
- 每 13 帧（7输入 + 6输出）为一个完整序列
- 每个文件可产生 3 个完整序列（49 // 13 = 3）

## 🎯 支持的模型

OpenSTL 支持多种时空预测模型：

| 模型 | 类型 | 特点 | 配置文件 |
|------|------|------|----------|
| **SimVP** | CNN | 快速、高效 | `configs/sevir_raw/SimVP.py` |
| **ConvLSTM** | RNN | 经典循环网络 | `configs/sevir_raw/ConvLSTM.py` |
| **PredRNN** | RNN | 改进的RNN | `configs/sevir_raw/PredRNN.py` |

### 模型对比

```bash
# 快速但效果好
./train_sevir_raw.sh SimVP

# 更强的时序建模
./train_sevir_raw.sh ConvLSTM

# 适合长序列预测
./train_sevir_raw.sh PredRNN
```

## ⚙️ 配置参数

### 关键参数

```python
# 数据参数
pre_seq_length = 7      # 输入帧数
aft_seq_length = 6      # 输出帧数
in_shape = [7, 1, 128, 128]  # [T, C, H, W]
batch_size = 8          # 批次大小

# 训练参数
lr = 1e-3               # 学习率
epochs = 200            # 训练轮数
warmup_epoch = 5        # 预热轮数

# SimVP 特定
hid_S = 64              # 空间隐藏维度
hid_T = 512             # 时间隐藏维度
N_S = 4                 # 空间块数
N_T = 8                 # 时间块数

# ConvLSTM 特定
num_layers = 4          # LSTM 层数
num_hidden = [64, 64, 64, 64]  # 隐藏单元数
filter_size = 5         # 卷积核大小
```

### 自定义配置

修改配置文件 `configs/sevir_raw/<Model>.py`：

```python
# 减小批次（GPU内存不足）
batch_size = 4

# 增加模型容量
hid_S = 128
hid_T = 1024

# 调整学习率
lr = 5e-4
```

## 🧪 测试 Dataloader

在训练之前测试数据加载：

```bash
cd /root/autodl-tmp/Sevir/OpenSTL
python openstl/datasets/dataloader_sevir_raw.py
```

预期输出：
```
✅ SEVIR TRAIN 雷达预测数据集 (OpenSTL)
   Files: 4544 个文件
   输入帧数: 7 帧
   输出帧数: 6 帧

📊 批次形状:
  输入:  torch.Size([4, 7, 1, 128, 128])
  输出:  torch.Size([4, 7, 1, 128, 128])
```

## 📁 目录结构

```
OpenSTL/
├── configs/
│   └── sevir_raw/              # SEVIR 配置文件
│       ├── SimVP.py            # SimVP 配置
│       ├── ConvLSTM.py         # ConvLSTM 配置
│       └── PredRNN.py          # PredRNN 配置
├── openstl/
│   ├── datasets/
│   │   ├── dataloader_sevir_raw.py  # SEVIR 原始数据加载器 ⭐
│   │   └── dataloader.py       # 数据加载器入口
│   └── ...
├── tools/
│   ├── train.py                # 训练入口
│   └── test.py                 # 测试入口
├── train_sevir_raw.sh          # 快速训练脚本 ⭐
└── SEVIR_RAW_GUIDE.md          # 本指南
```

## 📈 训练监控

### 查看训练日志

```bash
# 实时查看日志
tail -f /root/autodl-tmp/results/OpenSTL/sevir_SimVP_7to6/log.log

# 查看 TensorBoard（如果启用）
tensorboard --logdir /root/autodl-tmp/results/OpenSTL/sevir_SimVP_7to6
```

### 训练输出

```
Epoch 1/200:
  Train Loss: 0.0123
  Val Loss: 0.0145
  Val MAE: 0.0234
  Val SSIM: 0.8123
  Time: 4.5min
```

## 🔧 常见问题

### Q1: GPU 内存不足？
**A**: 
```bash
# 减小批次大小
./train_sevir_raw.sh SimVP
# 编辑配置文件，修改 batch_size = 4
```

### Q2: 数据加载慢？
**A**: 增加 `--num_workers` 参数（默认 4，可增至 8）

### Q3: 训练时间估算？
**A**: 
- SimVP: ~4分钟/epoch → 200 epochs ≈ 13-15小时
- ConvLSTM: ~6分钟/epoch → 200 epochs ≈ 20小时
- PredRNN: ~5分钟/epoch → 200 epochs ≈ 16-18小时

### Q4: 如何修改输入/输出帧数？
**A**: 
```bash
python tools/train.py \
    --dataname sevir_raw \
    --input_frames 10 \
    --output_frames 10 \
    --in_shape 10 1 128 128
```
**注意**：确保 `input_frames + output_frames <= 49`

### Q5: 如何恢复训练？
**A**: OpenSTL 支持自动 checkpoint，在训练中断后重新运行相同命令即可恢复

## 💡 7→6 帧的实现细节

### Padding 策略
由于大多数模型要求输入输出帧数相同，我们采用 padding 策略：

1. **数据加载**：目标6帧 + padding 1帧 → 变成7帧
2. **Loss计算**：只计算前6帧的loss，忽略padding帧
3. **评估**：只在前6帧上计算指标

```python
# 在 dataloader_sevir_raw.py 中
if self.output_frames < self.input_frames:
    padding_frames = self.input_frames - self.output_frames
    last_frame = target_data[:, :, -1:]
    padding = last_frame.repeat(1, 1, padding_frames)
    target_data = torch.cat([target_data, padding], dim=2)
```

## 📚 相关文档

- [OpenSTL 官方文档](https://openstl.readthedocs.io/)
- [OpenSTL GitHub](https://github.com/chengtan9907/OpenSTL)
- `Sevir/data.py` - 原始数据处理（参考）

## 🎉 开始训练！

选择您喜欢的模型并开始训练：

```bash
# SimVP - 推荐首选
./train_sevir_raw.sh SimVP

# ConvLSTM - 经典模型
./train_sevir_raw.sh ConvLSTM

# PredRNN - 改进的RNN
./train_sevir_raw.sh PredRNN
```

训练完成后，模型和结果将保存在 `/root/autodl-tmp/results/OpenSTL/` 目录下。

---

**✅ 配置完成！现在可以开始训练雷达预测模型了！** 🚀

