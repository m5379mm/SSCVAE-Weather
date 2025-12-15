import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math
from geomloss import SamplesLoss

# 注意力机制的编码器-解码器，并引入了LISTA用于稀疏编码的解码部分

# 通道注意力机制：通过全局平均池化和最大池化来提取每个通道的全局信息，然后通过一系列全连接层生成通道注意力权重，最终，输入特征图与这些权重相乘，以强调重要的通道特征。
class ChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction):
        super(ChannelAttention, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        self._max_pool = nn.AdaptiveMaxPool2d(1) # 最大池化
        self._fc = nn.Sequential( # 两层的全连接层，用于生成通道注意力权重
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self._sigmoid = nn.Sigmoid() # 激活函数，映射到[0,1]
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avgOut = self._fc(self._avg_pool(x).view(b, c))  # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        maxOut = self._fc(self._max_pool(x).view(b, c))  # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self._sigmoid(avgOut + maxOut).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
        out = x * y.expand_as(x)  # [B, C, H, W]
        return out

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self,
                 kernel_size):
        super(SpatialAttention, self).__init__()
        self._conv = nn.Conv2d(in_channels=2,
                               out_channels=1,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=(kernel_size - 1) // 2,
                               bias=False)
        self._sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, _, h, w = x.size()
        avgOut = torch.mean(x, dim=1, keepdim=True)  # [B, C, H, W] -> [B, 1, H, W]
        maxOut, _ = torch.max(x, dim=1, keepdim=True)  # [B, C, H, W] -> [B, 1, H, W]
        y = torch.cat([avgOut, maxOut], dim=1)  # [B, 2, H, W]
        y = self._sigmoid(self._conv(y))  # [B, 2, H, W] -> [B, 1, H, W]
        out = x * y.expand_as(x)  # [B, C, H, W]
        return out


class CBAM(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction,
                 kernel_size):
        super(CBAM, self).__init__()
        self.ChannelAtt = ChannelAttention(in_channels, reduction)
        self.SpatialAtt = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ChannelAtt(x)  # [B, C, H, W]
        x = self.SpatialAtt(x)  # [B, C, H, W]
        return x

# 残差块 使用了两个卷积层，并通过跳跃连接将输入直接加到输出上。这样可以缓解深层神经网络中的梯度消失问题，并加速训练。
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels):
        super(ResidualBlock, self).__init__()
        
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hid_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=hid_channels,
                      out_channels=in_channels,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x) # 将输入直接加到经过一系列卷积和激活函数处理后的输出上

# 下采样操作
class DownSampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DownSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels), # 批量归一化？？？
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self._block(x)

#上采样操作 
class UpSampleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UpSampleBlock, self).__init__()

        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2, stride=2),# 反卷积？？？
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self._block(x)

# 激活函数
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

# 非局部注意力块：捕捉全局依赖关系。通过计算输入特征的自相关矩阵来生成全局上下文信息，并将其应用到原始输入特征上。
class NonLocalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels):
        super(NonLocalBlock, self).__init__()

        self.hid_channels = hid_channels
        self._conv_theta = nn.Conv2d(in_channels=in_channels,
                                     out_channels=hid_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self._conv_phi = nn.Conv2d(in_channels=in_channels,
                                   out_channels=hid_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self._conv_g = nn.Conv2d(in_channels=in_channels,
                                 out_channels=hid_channels,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self._soft_max = nn.Softmax(dim=1)
        self._conv_mask = nn.Conv2d(in_channels=hid_channels,
                                    out_channels=in_channels,
                                    kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # [B, C, H, W] -> [B, C', HW] -> [B, HW, C']
        theta = self._conv_theta(x).view(b, self.hid_channels, -1).permute(0, 2, 1).contiguous()
        # [B, C, H, W] -> [B, C', HW]
        phi = self._conv_phi(x).view(b, self.hid_channels, -1)
        # [B, C, H, W] -> [B, C', HW] -> [B, HW, C']
        g = self._conv_g(x).view(b, self.hid_channels, -1).permute(0, 2, 1).contiguous()
        # [B, HW, C'] * [B, C', HW] = [B, HW, HW]
        mul_theta_phi = self._soft_max(torch.matmul(theta, phi))
        # [B, HW, HW] * [B, HW, C'] = [B, HW, C']
        mul_theta_phi_g = torch.matmul(mul_theta_phi, g)
        # [B, HW, C'] -> [B, C', HW] -> [B, C', H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.hid_channels, h, w)
        # [B, C', H, W] -> [B, C, H, W]
        mask = self._conv_mask(mul_theta_phi_g)

        return x + mask


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups):
        super(Encoder, self).__init__()

        # [B, T, C, H, W] -> [B, T, C', H, W]
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=hid_channels_1,
                                 kernel_size=3, stride=1, padding=1)
        
        # 下采样模块: [B, T, C', H, W] -> [B, T, C'', h, w]
        self._down_samples = nn.ModuleList()
        for i in range(down_samples):
            cur_in_channels = hid_channels_1 if i == 0 else hid_channels_2
            self._down_samples.append(
                ResidualBlock(in_channels=cur_in_channels,
                              hid_channels=cur_in_channels // 2)
            )
            self._down_samples.append(
                DownSampleBlock(in_channels=cur_in_channels,
                                out_channels=hid_channels_2)
            )

        # [B, T, C'', H, W] -> [B, T, C'', H, W]
        self._res_1 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock(in_channels=hid_channels_2,
                                        hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        
        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_2)
        self._swish = Swish()

        # [B, T, C'', H, W] -> [B, T, n, H, W]
        self._conv_2 = nn.Conv2d(in_channels=hid_channels_2,
                                 out_channels=out_channels,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, T, C, H, W = x.size()

        # 处理每个时间步：输入 [B*T, C, H, W]
        x = x.view(-1, C, H, W)
        x = self._conv_1(x)


        for layer in self._down_samples:
            x = layer(x)

        x = self._res_1(x)

        x = self._non_local(x)
    
        x = self._res_2(x)

        x = self._group_norm(x)
        x = self._swish(x)
        x = self._conv_2(x)

        _,D,h,w=x.size()
        x = x.view(B, T, D, h, w)  # 恢复时间维度: [B, T, C'', H, W]
        return x  # [B, T, out_channels, H, W]

class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 up_samples,
                 num_groups):
        super(Decoder, self).__init__()

        # [B, T, n, H, W] -> [B, T, C'', H, W]
        self._conv_1 = nn.Conv2d(in_channels=out_channels,
                                 out_channels=hid_channels_2,
                                 kernel_size=3, stride=1, padding=1)

        # [B, T, C'', H, W] -> [B, T, C'', H, W]
        self._res_1 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock(in_channels=hid_channels_2,
                                        hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        
        # 上采样模块: [B, T, C'', H, W] -> [B, T, C', H, W]
        self._up_samples = nn.ModuleList()
        for i in range(up_samples):
            cur_in_channels = hid_channels_2 if i == 0 else hid_channels_1
            self._up_samples.append(
                ResidualBlock(in_channels=cur_in_channels,
                              hid_channels=cur_in_channels // 2)
            )
            self._up_samples.append(
                UpSampleBlock(in_channels=cur_in_channels,
                              out_channels=hid_channels_1)
            )
        
        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_1)
        self._swish = Swish()

        # [B, T, C', H, W] -> [B, T, C, H, W]
        self._conv_2 = nn.Conv2d(in_channels=hid_channels_1,
                                 out_channels=in_channels,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, T, D, h, w = x.size()
        x = x.view(-1, D, h, w)
        # 处理每个时间步
        x = self._conv_1(x)
        x = self._res_1(x)
        x = self._non_local(x)
        x = self._res_2(x)

        for layer in self._up_samples:
            x = layer(x)

        x = self._group_norm(x)
        x = self._swish(x)
        x = self._conv_2(x)
        _,C,H,W=x.size()
        x = x.view(B, T, C, H, W)
        return x  # [B, T, C, H, W]

# DCT字典初始化：建立一个超完备的DCT字典
def init_dct(n, m):  # n, m
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    oc_dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)  
        if k > 0:
            V = V - np.mean(V)
        oc_dictionary[:, k] = V / np.linalg.norm(V)  
    oc_dictionary = np.kron(oc_dictionary, oc_dictionary)   
    oc_dictionary = oc_dictionary.dot(np.diag(1 / np.sqrt(np.sum(oc_dictionary ** 2, axis=0))))  
    idx = np.arange(0, n ** 2)  
    idx = idx.reshape(n, n, order="F")  
    idx = idx.reshape(n ** 2, order="C") 
    oc_dictionary = oc_dictionary[idx, :]
    oc_dictionary = torch.from_numpy(oc_dictionary).float()
    return oc_dictionary  # n**2, m**2

class TemporalAttention(nn.Module):
    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        # 用一个可训练的向量表示每个时间步的重要性
        self.attn = nn.Parameter(torch.randn(1, num_steps, 1, 1, 1))  # [1, T, 1, 1, 1]
        self.softmax = nn.Softmax(dim=1)  # softmax 在时间维度

    def forward(self, x):  # x: [B, T, C, H, W]
        # 1. 复制权重到 batch 维度
        attn_scores = self.attn.expand(x.size(0), -1, x.size(2), x.size(3), x.size(4))  # [B, T, C, H, W]
        # 2. 在时间维度 softmax
        attn_scores = self.softmax(attn_scores)  # [B, T, C, H, W]
        # 3. 加权
        x_attended = x * attn_scores  # [B, T, C, H, W]
        return x_attended


class LISTA(nn.Module):
    def __init__(self,
                 num_atoms,
                 num_dims,
                 num_iters,
                 h,
                 w,
                 device):
        super(LISTA, self).__init__()

        self._num_atoms = num_atoms
        self._num_dims = num_dims
        self._device = device

        self._Dict = nn.Parameter(self.initialize_dct_weights())  # [D, K]
        self._L = nn.Parameter((torch.norm(self._Dict, p=2)) ** 2)  # scalar
        one = torch.ones(h, w)
        one = torch.unsqueeze(one, 0)
        one = torch.unsqueeze(one, -1)  # [1, h, w, 1]
        self._alpha = nn.Parameter(one)

        self._Zero = torch.zeros(num_atoms).to(device)  # [K]
        self._Identity = torch.eye(num_atoms).to(device)  # [K, K]

        self._num_iters = num_iters
    
    def initialize_dct_weights(self):
        n = math.ceil(math.sqrt(self._num_dims))
        m = math.ceil(math.sqrt(self._num_atoms))
        weights = init_dct(n, m)[:, :self._num_atoms]  # [D, K]
        return weights

    def soft_thresh(self, x, theta):
        return torch.sign(x) * torch.max(torch.abs(x) - theta, self._Zero)

    def generation(self, input_z):
        input_z = input_z.permute(0, 2, 3, 1).contiguous()  # [B, K, H, W] -> [B, H, W, K]
        x_recon = torch.matmul(input_z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]
        return x_recon

    def forward(self, x):
        l = self._alpha / self._L  # scalar

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, D, H, W] -> [B, H, W, D]
        ## print(x.size(),'999')

        S = self._Identity - (1 / self._L) * self._Dict.t().mm(self._Dict)  # [K, K]
        S = S.t()  # [K, K]

        y = torch.matmul(x, self._Dict)  # [B, H, W, D] * [D, K] -> [B, H, W, K]

        z = self.soft_thresh(y, l)  # [B, H, W, K]
        for t in range(self._num_iters):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self._L) * y, l)

        x_recon = torch.matmul(z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]

        z = z.permute(0, 3, 1, 2).contiguous()  # [B, H, W, K] -> [B, K, H, W]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]

        return z, x_recon, self._Dict

class AttentiveLISTA_Time(nn.Module):
    def __init__(self,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device):
        super(AttentiveLISTA_Time, self).__init__()
        self._num_atoms = num_atoms
        self._num_dims = num_dims
        self._device = device

        self._Dict = nn.Parameter(self.initialize_dct_weights())  # [D, K]
        self._L = nn.Parameter((torch.norm(self._Dict, p=2)) ** 2)  # scalar
        self._conv = nn.Conv2d(in_channels=num_dims,
                               out_channels=num_atoms,
                               kernel_size=3, stride=1, padding=1)
        self._res1 = ResidualBlock(in_channels=num_atoms,
                                   hid_channels=num_atoms//2)
        self._res2 = ResidualBlock(in_channels=num_atoms,
                                   hid_channels=num_atoms//2)
        self._cbam = CBAM(in_channels=num_atoms,
                          reduction=16,
                          kernel_size=3)

        self._Zero = torch.zeros(num_atoms).to(device)  # [K]
        self._Identity = torch.eye(num_atoms).to(device)  # [K, K]

        # 改进：时序注意力应用在特征上，而不是阈值参数
        # 移除对阈值参数的时序注意力，改为在特征上进行时序建模
        self._temporal_conv = nn.Conv1d(num_atoms, num_atoms, kernel_size=3, padding=1, groups=num_atoms)
        self._temporal_norm = nn.BatchNorm1d(num_atoms)

        self._num_iters = num_iters
        self._lista_model = LISTA(num_atoms=num_atoms,
                                  num_dims=num_dims,
                                  num_iters=num_iters,
                                  h=10,  # 你需要根据实际尺寸传递正确的 h 和 w
                                  w=10,
                                  device=device)

    def initialize_dct_weights(self):
        n = math.ceil(math.sqrt(self._num_dims))
        m = math.ceil(math.sqrt(self._num_atoms))
        weights = init_dct(n, m)[:, :self._num_atoms]  # [D, K]
        
        return weights
    def soft_thresh(self, x, theta):
        return torch.sign(x) * torch.max(torch.abs(x) - theta, self._Zero)
    
    def generation(self, input_z):
        """
        input_z: [B, T, K, H, W]  输入的稀疏编码序列
        返回:
            x_recon_seq: [B, T, D, H, W]  重建的卫星降水序列
        """
        B, T, K, H, W = input_z.size()
        x_recon_seq = []

        # 逐帧生成每个时间步的重建图像
        for t in range(T):
            z_t = input_z[:, t]  # 选择第 t 个时间步的稀疏编码 [B, K, H, W]
            z_t = z_t.permute(0,2,3,1).contiguous()
            x_recon_t = torch.matmul(z_t, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]
            x_recon_t = x_recon_t.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]
            x_recon_seq.append(x_recon_t)

        # 堆叠时间维度
        x_recon_seq = torch.stack(x_recon_seq, dim=1)  # [B, T, D, H, W]

        return x_recon_seq

    def forward(self, x):
        """
        x: [B, T, C, H, W]  输入的卫星降水序列
        返回:
            z_seq: [B, T, K, H, W]  稀疏编码表示
            x_recon_seq: [B, T, D, H, W]  重建的卫星降水序列
            dictionary: LISTA 字典 [D, K]
        """
        B, T, C, H, W = x.size()

        # 1. 展平时间维度用于 CBAM
        x_ = x.view(B * T, C, H, W)  # [B*T, C, H, W]

        # 2. CBAM + 前置卷积 + 残差块
        l = self._conv(x_)           # [B*T, K, H, W]
        l = self._res1(l)
        l = self._res2(l)
        l = self._cbam(l)            # [B*T, K, H, W]
        # print(l.size(), "After CBAM")  # 查看CBAM输出形状

        # 3. 恢复时间维度
        l = l.view(B, T, -1, H, W)   # [B, T, K, H, W]
        
        # 4. 改进：时序建模应用在特征上，而不是阈值参数
        # 先对特征进行时序建模，然后再计算阈值
        # 对每个空间位置，在时间维度上进行1D卷积
        l_reshaped = l.permute(0, 3, 4, 2, 1).contiguous()  # [B, H, W, K, T]
        l_reshaped = l_reshaped.view(B * H * W, -1, T)  # [B*H*W, K, T]
        l_temporal = self._temporal_conv(l_reshaped)  # [B*H*W, K, T]
        l_temporal = self._temporal_norm(l_temporal)
        l_temporal = l_temporal.view(B, H, W, -1, T).permute(0, 4, 3, 1, 2).contiguous()  # [B, T, K, H, W]
        
        # 残差连接：原始特征 + 时序建模特征（小权重）
        l = l + 0.1 * l_temporal
        
        # 5. 迭代每个时间步输入 LISTA
        z_seq = []
        x_recon_seq = []
        for t in range(T):
            # 处理当前时间步
            l_t = l[:, t]  # [B, K, H, W]  选择第 t 个时间步
            l_t = l_t / self._L  # 正则化（阈值参数）

            # 输入到 LISTA 模型
            l_t = l_t.permute(0, 2, 3, 1).contiguous()  # [B, K, H, W] -> [B, H, W, K]

            x_t = x[:, t].permute(0, 2, 3, 1).contiguous()  # [B, D, H, W] -> [B, H, W, D]

            # 计算 S 和 y
            S = self._Identity - (1 / self._L) * self._Dict.t().mm(self._Dict)  # [K, K]
            S = S.t()  # [K, K]

            y = torch.matmul(x_t, self._Dict)  # [B, H, W, D] * [D, K] -> [B, H, W, K]

            # soft thresholding
            z = self.soft_thresh(y, l_t)  # [B, H, W, K]
            for iter in range(self._num_iters):
                z = self.soft_thresh(torch.matmul(z, S) + (1 / self._L) * y, l_t)

            # 重建
            x_recon = torch.matmul(z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]

            # 恢复维度顺序
            z = z.permute(0, 3, 1, 2).contiguous()  # [B, H, W, K] -> [B, K, H, W]
            x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]

            z_seq.append(z)         # [B, K, H, W]
            x_recon_seq.append(x_recon)  # [B, D, H, W]

        # 6. 堆叠时间维度
        z_seq = torch.stack(z_seq, dim=1)           # [B, T, K, H, W]
        x_recon_seq = torch.stack(x_recon_seq, dim=1) # [B, T, D, H, W]

        return z_seq, x_recon_seq, self._Dict




class AttentiveLISTA(nn.Module):
    def __init__(self,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device):
        super(AttentiveLISTA, self).__init__()

        self._num_atoms = num_atoms
        self._num_dims = num_dims
        self._device = device

        self._Dict = nn.Parameter(self.initialize_dct_weights())  # [D, K]
        self._L = nn.Parameter((torch.norm(self._Dict, p=2)) ** 2)  # scalar
        self._conv = nn.Conv2d(in_channels=num_dims,
                               out_channels=num_atoms,
                               kernel_size=3, stride=1, padding=1)
        self._res1 = ResidualBlock(in_channels=num_atoms,
                                   hid_channels=num_atoms//2)
        self._res2 = ResidualBlock(in_channels=num_atoms,
                                   hid_channels=num_atoms//2)
        self._cbam = CBAM(in_channels=num_atoms,
                          reduction=16,
                          kernel_size=3)

        self._Zero = torch.zeros(num_atoms).to(device)  # [K]
        self._Identity = torch.eye(num_atoms).to(device)  # [K, K]

        self._num_iters = num_iters
    
    # def initialize_dct_weights(self):
    #     weights = torch.zeros(self._num_atoms, self._num_dims).to(self._device)  # [K, D]
    #     for i in range(self._num_atoms):
    #         atom = torch.cos((2 * torch.arange(self._num_dims) + 1) * i * (3.141592653589793 / (2 * self._num_dims)))# * math.sqrt(2 / self._num_dims)
    #         weights[i, :] = atom / torch.norm(atom, p=2)
    #     return weights.t()  # [D, K]
    
    def initialize_dct_weights(self):
        n = math.ceil(math.sqrt(self._num_dims))
        m = math.ceil(math.sqrt(self._num_atoms))
        weights = init_dct(n, m)[:, :self._num_atoms]  # [D, K]
        
        return weights
    
    def get_dict(self):
        return self._Dict
    
    def set_dict(self, dictionary):
        self._Dict= nn.Parameter(dictionary)
        # print("Dictionary initialized with pretrained values.")

    def soft_thresh(self, x, theta):
        return torch.sign(x) * torch.max(torch.abs(x) - theta, self._Zero)

    def generation(self, input_z):
        input_z = input_z.permute(0, 2, 3, 1).contiguous()  # [B, K, H, W] -> [B, H, W, K]
        x_recon = torch.matmul(input_z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]
        return x_recon

    def forward(self, x):
        
        l = self._conv(x)  # [B, D, H, W] -> [B, K, H, W]
        l = self._res1(l)
        l = self._res2(l)
        
        l = self._cbam(l) / self._L
        # l = l / self._L

        l = l.permute(0, 2, 3, 1).contiguous()  # [B, K, H, W] -> [B, H, W, K]

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, D, H, W] -> [B, H, W, D]
        
        S = self._Identity - (1 / self._L) * self._Dict.t().mm(self._Dict)  # [K, K]
        S = S.t()  # [K, K]

        y = torch.matmul(x, self._Dict)  # [B, H, W, D] * [D, K] -> [B, H, W, K]

        z = self.soft_thresh(y, l)  # [B, H, W, K]
        for t in range(self._num_iters):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self._L) * y, l)

        x_recon = torch.matmul(z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]

        z = z.permute(0, 3, 1, 2).contiguous()  # [B, H, W, K] -> [B, K, H, W]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]

        return z, x_recon, self._Dict

# class SSCVAE(nn.Module):
#     def __init__(self,
#                  in_channels,# 输入图像的通道数，灰度为1，RGB为3 ## 已经是数据了——应该是3，通道1为069；通道2为107;通道3为lght
#                  hid_channels_1,
#                  hid_channels_2,# 这两部分根据实验结果调整
#                  out_channels,  # 编码器的输出通道数，它定义了在下采样后的特征图的深度。这个值决定了从输入到潜在空间的维度。通常比 hid_channels_2 更大，例如 128 或 256。
#                  down_samples,
#                  num_groups,
#                  num_atoms,# 字典的原子个数，与num_dims相匹配
#                  num_dims, # 每个原子的维度
#                  num_iters,# 决定稀疏编码的收敛速度和精度
#                  device):
#         super(SSCVAE, self).__init__()

#         self._encoder = Encoder(in_channels=in_channels,
#                                 hid_channels_1=hid_channels_1,
#                                 hid_channels_2=hid_channels_2,
#                                 out_channels=out_channels,
#                                 down_samples=down_samples,
#                                 num_groups=num_groups)

#         self._decoder = Decoder(in_channels=in_channels,
#                                 hid_channels_1=hid_channels_1,
#                                 hid_channels_2=hid_channels_2,
#                                 out_channels=out_channels,
#                                 up_samples=down_samples,
#                                 num_groups=num_groups)

#         self._decoder_trans = Decoder(in_channels=in_channels,
#                                 hid_channels_1=hid_channels_1,
#                                 hid_channels_2=hid_channels_2,
#                                 out_channels=out_channels,
#                                 up_samples=down_samples,
#                                 num_groups=num_groups)

#         self._LISTA = AttentiveLISTA(num_atoms=num_atoms,
#                                      num_dims=num_dims,
#                                      num_iters=num_iters,
#                                      device=device)

#         self._mlp=MLPforTrans(in_channels,
#                  hid_channels_1,
#                  hid_channels_2,
#                  out_channels)

#     def generation(self, input_z):
#         ex = self._LISTA.generation(input_z)  # [B, K, h, w] -> [B, D, h, w]
#         x_generation = self._decoder(ex)  # [B, D, h, w] -> [B, C, H, W]
#         x_generation = torch.sigmoid(x_generation)
#         return x_generation
    
#     def get_dict(self):
#         return self._LISTA.get_dict()

#     def forward(self, x):
#         ex = self._encoder(x)  # [B, C, H, W] -> [B, D, h, w]
#         z, ex_recon, dictionary = self._LISTA(ex)  # [B, D, h, w] -> [B, K, h, w], [B, D, h, w]
        
#         z_trans = self._mlp()
#         ex_recon_trans = slef.generation(z_trans)
#         x_recon_trans = self._decoder_trans(ex_recon_trans)# 得到的转换后的潜在表示
#         x_recon_trans = torch.sigmoid(x_recon_trans)

#         x_recon = self._decoder(ex_recon)  # [B, D, h, w] -> [B, C, H, W]
#         x_recon = torch.sigmoid(x_recon)

#         latent_loss = torch.sum((ex_recon - ex).pow(2), dim=1).mean() #KL损失吗

#         return x_recon, z, latent_loss, dictionary
class SSCVAEDouble(nn.Module):
    def __init__(self,
                 in_channels_sate,# 输入通道数为3：卫星
                 in_channels_radar,# 通道数为1：雷达图像
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,  # 编码器的输出通道数，它定义了在下采样后的特征图的深度。这个值决定了从输入到潜在空间的维度。通常比 hid_channels_2 更大，例如 128 或 256。
                 down_samples,
                 num_groups,
                 num_atoms,# 字典的原子个数，与num_dims相匹配
                 num_dims, # 每个原子的维度
                 num_iters,# 决定稀疏编码的收敛速度和精度
                 device):
        super(SSCVAEDouble, self).__init__()
        self._encoder_sate_r = Encoder(in_channels=1, 
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)
        self._encoder_sate_g = Encoder(in_channels=1, 
                        hid_channels_1=hid_channels_1,
                        hid_channels_2=hid_channels_2,
                        out_channels=out_channels,
                        down_samples=down_samples,
                        num_groups=num_groups)
        self._encoder_sate_b = Encoder(in_channels=1, 
                        hid_channels_1=hid_channels_1,
                        hid_channels_2=hid_channels_2,
                        out_channels=out_channels,
                        down_samples=down_samples,
                        num_groups=num_groups)
        self._encoder_sate = Encoder(in_channels=in_channels_sate,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)
        
        self._decoder_sate_r = Decoder(in_channels=1, 
                                       hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)
        self._decoder_sate_g = Decoder(in_channels=1, 
                                hid_channels_1=hid_channels_1,
                        hid_channels_2=hid_channels_2,
                        out_channels=out_channels,
                        up_samples=down_samples,
                        num_groups=num_groups)
        self._decoder_sate_b = Decoder(in_channels=1, 
                                hid_channels_1=hid_channels_1,
                        hid_channels_2=hid_channels_2,
                        out_channels=out_channels,
                        up_samples=down_samples,
                        num_groups=num_groups)

        self._decoder_sate = Decoder(in_channels=in_channels_sate,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)# 解码为三通道

        self._encoder_radar = Encoder(in_channels=in_channels_radar,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)

        self._decoder_radar= Decoder(in_channels=in_channels_radar,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)# 解码为三通道

        self._LISTA = AttentiveLISTA_Time(num_atoms=num_atoms,
                                     num_dims=num_dims,
                                     num_iters=num_iters,
                                     device=device,)
    def generation(self, input_z, is_sate=True):
        if is_sate:
            # 对卫星图像的每个通道分别应用LISTA生成
            ex_r = self._LISTA.generation(input_z)  # 对红色通道
            ex_g = self._LISTA.generation(input_z)  # 对绿色通道
            ex_b = self._LISTA.generation(input_z)  # 对蓝色通道

            # 对每个通道解码
            x_r = self._decoder_sate_r(ex_r)
            x_g = self._decoder_sate_g(ex_g)
            x_b = self._decoder_sate_b(ex_b)

            # 合并三个通道
            x_generation = torch.cat((x_r, x_g, x_b), dim=1)  # 合并为三通道的输出
        else:
            ex = self._LISTA.generation(input_z)
            x_generation = self._decoder_radar(ex)

        x_generation = torch.sigmoid(x_generation)
        return x_generation
        

    def forward(self, satellite_seq, radar_seq):
        # 0) [B,C,H,W,T] -> [B,T,C,H,W]
        sat_btchw = self.to_BTCHW(satellite_seq)  # 转换为 [B, T, C, H, W]
        radar_btchw = self.to_BTCHW(radar_seq)

        # 1) 编码每个通道
        ex_sate_r = self._encoder_sate_r(sat_btchw[:,:,0,:,:].unsqueeze(2))   # [B,T,D,h,w]
        ex_sate_g = self._encoder_sate_g(sat_btchw[:,:,1,:,:].unsqueeze(2))   # [B,T,D,h,w]
        ex_sate_b = self._encoder_sate_b(sat_btchw[:,:,2,:,:].unsqueeze(2))   # [B,T,D,h,w]
        ex_radar_seq = self._encoder_radar(radar_btchw)  # [B,T,D,h,w]

        # 2) LISTA 稀疏编码
        z_sate_r, exrec_sate_r, dictionary = self._LISTA(ex_sate_r)  # [B,T,K,h,w], [B,T,D,h,w]
        z_sate_g, exrec_sate_g, _ = self._LISTA(ex_sate_g)  # [B,T,K,h,w], [B,T,D,h,w]
        z_sate_b, exrec_sate_b, _ = self._LISTA(ex_sate_b)  # [B,T,K,h,w], [B,T,D,h,w]
        z_radar_seq, exrec_radar_seq, _ = self._LISTA(ex_radar_seq)  # [B,T,K,h,w], [B,T,D,h,w]

        # 3) 解码
        x_recon_sate_r = self._decoder_sate_r(exrec_sate_r)
        x_recon_sate_r = torch.sigmoid(x_recon_sate_r)

        x_recon_sate_g = self._decoder_sate_g(exrec_sate_g)
        x_recon_sate_g = torch.sigmoid(x_recon_sate_g)

        x_recon_sate_b = self._decoder_sate_b(exrec_sate_b)
        x_recon_sate_b = torch.sigmoid(x_recon_sate_b)

        # 合并卫星图像的三个通道
        x_recon_sate = torch.cat((x_recon_sate_r, x_recon_sate_g, x_recon_sate_b), dim=2)  # [B, 3, H, W]
        # print(x_recon_sate_b.size(),x_recon_sate.size())
        # 计算卫星的潜在损失
        latent_loss_sate = torch.sum((exrec_sate_r - ex_sate_r).pow(2), dim=1).mean() + \
                            torch.sum((exrec_sate_g - ex_sate_g).pow(2), dim=1).mean() + \
                            torch.sum((exrec_sate_b - ex_sate_b).pow(2), dim=1).mean()

        # 解码雷达图像
        x_recon_radar = self._decoder_radar(exrec_radar_seq)
        x_recon_radar = torch.sigmoid(x_recon_radar)

        latent_loss_radar = torch.sum((exrec_radar_seq - ex_radar_seq).pow(2), dim=1).mean()

        # 将卫星和雷达的潜在损失相加（或者加权）
        total_latent_loss = latent_loss_sate + latent_loss_radar

        # 将输出转换回原始形状
        x_recon_sate = self.to_BCHWT(x_recon_sate)
        x_recon_radar = self.to_BCHWT(x_recon_radar)

        # 手动释放不再需要的显存
        del ex_sate_r, ex_sate_g, ex_sate_b, ex_radar_seq, exrec_sate_r, exrec_sate_g, exrec_sate_b, exrec_radar_seq
        torch.cuda.empty_cache()  # 清理缓存

        return x_recon_sate, x_recon_radar, z_sate_r, z_radar_seq, total_latent_loss, dictionary

    def get_dict(self):
        return self._LISTA.get_dict()
    # --- 小工具：把 [B,C,H,W,T] <-> [B,T,C,H,W] 互转 ---

    def to_BTCHW(self,x_bchwt):
        # [B,C,H,W,T] -> [B,T,C,H,W]
        # print(x_bchwt.size())
        return x_bchwt.permute(0, 4, 1, 2, 3).contiguous()

    def to_BCHWT(self,x_btchw):
        # [B,T,C,H,W] -> [B,C,H,W,T]
        return x_btchw.permute(0, 2, 3, 4, 1).contiguous()

    # def forward(self, satellite_seq, radar_seq):
    #     # 0) [B,C,H,W,T] -> [B,T,C,H,W]
    #     sat_btchw   = self.to_BTCHW(satellite_seq)
    #     # print(sat_btchw.size())
    #     radar_btchw = self.to_BTCHW(radar_seq)
    #     # print(radar_btchw.size())

    #     # 1) 编码到中间表征
    #     ex_sate_seq  = self._encoder_sate(sat_btchw[:,:,0,:,:])   # [B,T,D,h,w]
    #     ex_radar_seq = self._encoder_radar(radar_btchw)  # [B,T,D,h,w]

    #     # 2) LISTA 稀疏编码
    #     # print(ex_sate_seq.size())
    #     z_sate_seq,  exrec_sate_seq,  dictionary = self._LISTA(ex_sate_seq)   # [B,T,K,h,w], [B,T,D,h,w]
    #     z_radar_seq, exrec_radar_seq, _         = self._LISTA(ex_radar_seq)   # [B,T,K,h,w], [B,T,D,h,w]

    #     x_recon_sate = self._decoder_sate(exrec_sate_seq)
    #     x_recon_sate = torch.sigmoid(x_recon_sate)

    #     # 计算卫星的潜在损失
    #     latent_loss_sate = torch.sum((exrec_sate_seq - ex_sate_seq).pow(2), dim=1).mean()

    #     x_recon_radar = self._decoder_radar(exrec_radar_seq)
    #     x_recon_radar = torch.sigmoid(x_recon_radar)

    #     latent_loss_radar = torch.sum((exrec_radar_seq - ex_radar_seq).pow(2), dim=1).mean()

    #     # 将卫星和雷达的潜在损失相加（或者加权）
    #     total_latent_loss = latent_loss_sate + latent_loss_radar

    #     x_recon_sate = self.to_BCHWT(x_recon_sate)
    #     x_recon_radar = self.to_BCHWT(x_recon_radar)

    #     return x_recon_sate,x_recon_radar, z_sate_seq,z_radar_seq, total_latent_loss, dictionary

# def segmented_weighted_loss(x_recon, x_target):
#     """
#     分段加权 MSE loss (标准化到 0-1)
#     x_recon, x_target: [B, C, T, H, W]
#     """
#     # ---- 统一到 0–1 ----
#     if x_target.max() > 1:
#         x_target = x_target / 255.0
#     if x_recon.max() > 1:
#         x_recon = x_recon / 255.0

#     # ---- 分段权重 (基于反射率 dBZ 范围) ----
#     # 注意：这里用归一化后的值来划分阈值，需要换算
#     # 例如：74/255 ≈ 0.29, 181/255 ≈ 0.71, 219/255 ≈ 0.86
#     x_target_01 = x_target
#     weights = torch.ones_like(x_target_01)

#     weights[(x_target_01 >= 16/254) & (x_target_01 < 181/254)] = 1.0
#     weights[(x_target_01 >= 181/254) & (x_target_01 < 219/254)] = 5.0
#     weights[(x_target_01 >= 219/254)] = 10.0

#     abs_diff = torch.abs(x_recon - x_target)

#     # 帧级平均（可视为 temporal attention）
#     frame_losses = (weights * abs_diff).flatten(1, -1).mean(dim=1)  # [B]
#     loss = frame_losses.mean()

    # return loss

def segmented_weighted_loss(x_recon, x_target):
    """
    分段加权 MSE loss (标准化到 0-1)
    x_recon, x_target: [B, C, T, H, W]
    """
    # ---- 统一到 0–1 ----
    if x_target.max() > 1:
        x_target = x_target / 255.0
    if x_recon.max() > 1:
        x_recon = x_recon / 255.0

    # ---- 分段权重 (基于反射率 dBZ 范围) ----
    # 注意：这里用归一化后的值来划分阈值，需要换算
    # 例如：74/255 ≈ 0.29, 181/255 ≈ 0.71, 219/255 ≈ 0.86
    x_target_01 = x_target
    weights = torch.ones_like(x_target_01)

    weights[(x_target_01 >= 16/254) & (x_target_01 < 181/254)] = 1.0
    weights[(x_target_01 >= 181/254) & (x_target_01 < 219/254)] = 5.0
    weights[(x_target_01 >= 219/254)] = 10.0

    abs_diff = torch.abs(x_recon - x_target)

    # ---- 逐帧计算损失 ----
    total_frame_loss = 0.0
    B = x_recon.size(0)
    T = x_recon.size(2)

    for t in range(T):
        # 选择当前时间步 t 的所有数据
        temporal_abs_diff = abs_diff[:, :, t, :, :]  # [B, C, H, W]
        temporal_weights = weights[:, :, t, :, :]  # [B, C, H, W]

        # 计算当前帧的加权损失 (按像素进行加权)
        frame_loss = (temporal_weights * temporal_abs_diff)  # [B, C, H, W]
        
        # 累加到总损失
        total_frame_loss += frame_loss

    # ---- 在 B * T 上进行平均 ----
    # 将损失展开并在 B * T 维度上进行平均
    frame_losses = total_frame_loss.flatten(0, 1).mean()  # [1]

    return frame_losses



import torch
import torch.nn as nn
import torch.nn.functional as F
# class WeightedMultiScaleFeatureFusion(nn.Module): 
#     def __init__(self, in_channels, out_channels): 
#         super().__init__() 
#         self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1) 
#         self.conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1) 
#         self.conv5x5 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2) 
#         self.weights = nn.Parameter(torch.ones(3)) 
        
#     def forward(self, x): 
#         scale1 = self.conv1x1(x) 
#         scale2 = self.conv3x3(x) 
#         scale3 = self.conv5x5(x) 
#         w = torch.softmax(self.weights, dim=0) 
#         return w[0] * scale1 + w[1] * scale2 + w[2] * scale3
    
# # === 多尺度特征融合 ===
# class WeightedMultiScaleFeatureFusion(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(WeightedMultiScaleFeatureFusion, self).__init__()
#         # 使用2D卷积
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
#         self.weights = nn.Parameter(torch.ones(3))  # 用于加权不同尺度的特征

#     def forward(self, x):
#         # 计算不同尺度的特征
#         scale1 = self.conv1x1(x)  # [B, out_channels, H, W]
#         scale2 = self.conv3x3(x)  # [B, out_channels, H, W]
#         scale3 = self.conv5x5(x)  # [B, out_channels, H, W]
        
#         # 使用softmax对不同尺度的权重进行加权
#         w = torch.softmax(self.weights, dim=0)
#         return w[0] * scale1 + w[1] * scale2 + w[2] * scale3


# === 残差块 ===
# class ResidualBlock3D(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock3D, self).__init__()
#         # 使用2D卷积代替3D卷积
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))  # [B, channels, H, W]
#         out = self.bn2(self.conv2(out))  # [B, channels, H, W]
#         return self.relu(out + x)  # 残差连接

class WeightedMultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        scale1 = self.conv1x1(x)
        scale2 = self.conv3x3(x)
        scale3 = self.conv5x5(x)
        w = torch.softmax(self.weights, dim=0)
        return w[0] * scale1 + w[1] * scale2 + w[2] * scale3


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)
# 10.1普通版
# class SparseU_Net(nn.Module): 
#     def __init__(self, in_channels, out_channels, num_res_blocks=3): 
#         super(SparseU_Net, self).__init__() # ===== 编码器部分 ===== 
#         self.enc1 = nn.Sequential( nn.ConvTranspose3d(in_channels, 64, kernel_size=2, stride=(2, 1, 1)), nn.BatchNorm3d(64), nn.ReLU() ) 
#         self.enc2 = nn.Sequential( nn.ConvTranspose3d(64, 128, kernel_size=2, stride=(2, 2, 2)), nn.BatchNorm3d(128), nn.ReLU() ) 
#         self.enc3 = nn.Sequential( nn.Conv3d(128, 256, kernel_size=2, stride=(1, 2, 2)), # 升到 256 通道 
#                                   nn.BatchNorm3d(256), nn.ReLU() ) # ===== Bottleneck: 多尺度融合 + 多个残差块 ===== 
#         self.multi_scale_fusion = WeightedMultiScaleFeatureFusion(256, 256) 
#         self.res_blocks = nn.Sequential(*[ResidualBlock3D(256) for _ in range(num_res_blocks)]) # ===== 解码器部分 ===== 
#         self.dec3 = nn.Sequential( nn.ConvTranspose3d(256, 128, kernel_size=2, stride=(1, 2, 2)), nn.BatchNorm3d(128), nn.ReLU() ) 
#         self.dec2 = nn.Sequential( nn.Conv3d(128, 64, kernel_size=2, stride=(2, 2, 2)), nn.BatchNorm3d(64), nn.ReLU() ) 
#         self.dec1 = nn.Conv3d(64, out_channels, kernel_size=2, stride=(2, 1, 1)) 
#     def forward(self, x): 
#         """ 输入: [B, T, K, H, W] 输出: [B, T, K, H, W] """ 
#         x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, K, T, H, W] # 编码 
#         e1 = self.enc1(x) # [B, 64, 2T, H, W] 
#         e2 = self.enc2(e1) # [B, 128, 4T, H/2, W/2] 
#         e3 = self.enc3(e2) # [B, 512, 4T, H/4, W/4] # Bottleneck 
#         e3_fused = self.multi_scale_fusion(e3) 
#         e3_fused = self.res_blocks(e3_fused) # 解码 + skip 
#         d3 = self.dec3(e3_fused) + e2 
#         d2 = self.dec2(d3) + e1 
#         out = self.dec1(d2) # [B, out, T, H, W] 
#         return out.permute(0, 2, 1, 3, 4).contiguous()

# class WeightedMultiScaleFeatureFusion(nn.Module):
#     def __init__(self, in_channels_list, out_channels):
#         super(WeightedMultiScaleFeatureFusion, self).__init__()
#         self.convs = nn.ModuleList([
#             nn.Conv3d(in_channels, out_channels, kernel_size=1)
#             for in_channels in in_channels_list
#         ])
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, features):
#         assert len(features) == len(self.convs), "Number of features must match conv layers"
#         conv_features = [conv(f) for conv, f in zip(self.convs, features)]
        
#         # 确定最大时间和空间维度
#         max_t = max(f.size(2) for f in features)  # 最大时间维度
#         max_h = max(f.size(3) for f in features)  # 最大高度
#         max_w = max(f.size(4) for f in features)  # 最大宽度
#         target_size = (max_t, max_h, max_w)
        
#         # 对齐所有特征图到最大分辨率
#         conv_features = [F.interpolate(f, size=target_size, mode='trilinear', align_corners=False)
#                         if f.size()[2:] != target_size else f
#                         for f in conv_features]
        
#         weights = self.softmax(torch.stack(conv_features, dim=1))
#         fused = torch.sum(weights * torch.stack(conv_features, dim=1), dim=1)
#         return fused

class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)

# class SparseU_Net(nn.Module):
#     def __init__(self, in_channels, out_channels, num_res_blocks=3):
#         super(SparseU_Net, self).__init__()
#         # ===== 编码器部分 =====
#         self.enc1 = nn.Sequential(
#             nn.Conv3d(in_channels, 64, kernel_size=1, stride=1, padding=0),  
#             nn.BatchNorm3d(64),
#             nn.ReLU()
#         )
#         self.enc2 = nn.Sequential(
#             nn.ConvTranspose3d(64, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),  # T 不变
#             nn.BatchNorm3d(128),
#             nn.ReLU()
#         )
#         self.enc3 = nn.Sequential(
#             nn.ConvTranspose3d(128, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),  # T 不变
#             nn.BatchNorm3d(256),
#             nn.ReLU()
#         )
#         # ===== Bottleneck: 多尺度融合 + 多个残差块 =====
#         self.multi_scale_fusion = WeightedMultiScaleFeatureFusion(256,256)
#         self.res_blocks = nn.Sequential(*[ResidualBlock3D(256) for _ in range(num_res_blocks)])
#         # ===== 解码器部分 =====
#         self.dec3 = nn.Sequential(
#             nn.Conv3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),  # T 不变
#             nn.BatchNorm3d(128),
#             nn.ReLU()
#         )
#         self.dec2 = nn.Sequential(
#             nn.Conv3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),  # T 不变
#             nn.BatchNorm3d(64),
#             nn.ReLU()
#         )
#         self.dec1 = nn.Conv3d(64, out_channels, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x = x.permute(0, 2, 1, 3, 4).contiguous()
#         e1 = self.enc1(x)  # [B, 64, T, H, W]
#         e2 = self.enc2(e1)  # [B, 128, 2T, 2H, 2W]
#         e3 = self.enc3(e2)  # [B, 256, 4T, 4H, 4W]

#         e3_fused = self.multi_scale_fusion(e3)
#         bottleneck = self.res_blocks(e3_fused)

#         d3 = self.dec3(bottleneck) + e2
#         d2 = self.dec2(d3) + e1
#         out = self.dec1(d2)
#         return out.permute(0, 2, 1, 3, 4).contiguous()

class WeightedMultiScaleFeatureFusion2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        scale1 = self.conv1x1(x)
        scale2 = self.conv3x3(x)
        scale3 = self.conv5x5(x)
        w = torch.softmax(self.weights, dim=0)
        return w[0] * scale1 + w[1] * scale2 + w[2] * scale3

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class SparseU_Net0(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=3):
        super(SparseU_Net0, self).__init__()

        # ===== 编码器部分 =====
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0),  
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=0),  
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=0),  
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # ===== Bottleneck: 多尺度融合 + 多个残差块 =====
        self.multi_scale_fusion = WeightedMultiScaleFeatureFusion(128, 128)
        self.res_blocks = nn.Sequential(*[ResidualBlock2D(128) for _ in range(num_res_blocks)])

        # ===== 解码器部分 =====
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=2, padding=0), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=2, stride=2, padding=0),  
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dec1 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        输入: [B, T, K, H, W]
        输出: [B, T, K, H, W]
        """
        batch_size, num_frames, num_channels, height, width = x.size()
        outputs = []
        # print(x.size())
        # input()
        # 逐帧处理
        for t in range(num_frames):
            frame = x[:, t, :, :, :]  # [B, K, H, W]
            # 编码
            e1 = self.enc1(frame)  # [B, 32, H, W]
            e2 = self.enc2(e1)    # [B, 64, H/2, W/2]
            e3 = self.enc3(e2)    # [B, 128, H/4, W/4]
            
            # Bottleneck
            e3_fused = self.multi_scale_fusion(e3)

            # 解码 + skip (需保存 e1, e2)
            d3 = self.dec3(e3_fused) + e2  # [B, 64, H/2, W/2]
            d2 = self.dec2(d3) + e1        # [B, 32, H, W]
            out = self.dec1(d2)            # [B, out_channels, H, W]
            outputs.append(out)
            # print(f"Output size: {out.size()}")

        # 重组输出
        out = torch.stack(outputs, dim=2)  # [B, out_channels, T, H, W]
        # print(f"Output size: {out.size()}")
        # input()
        return out.permute(0, 2, 1, 3, 4).contiguous()  # 恢复为 [B, T, K, H, W]  
    
# 最初版  
# 改进版本：保持时间维度不变，避免膨胀
class SparseU_Net_Fixed(nn.Module):
    """
    修复版本：保持时间维度不变，使用2D卷积逐帧处理 + 轻量级时序融合
    避免时间维度膨胀导致的信息丢失
    """
    def __init__(self, in_channels, out_channels, num_res_blocks=3):
        super(SparseU_Net_Fixed, self).__init__()

        # 逐帧处理的2D卷积模块（类似 TranslatorWithResidual）
        # 注意：这里需要导入 TranslatorWithResidual，如果导入失败，使用下面的内联版本
        try:
            from models import TranslatorWithResidual
            self.frame_translator = TranslatorWithResidual(
                C_in=in_channels,
                C_hid=196,
                C_out=out_channels,
                incep_ker=[1, 3, 5],
                groups=4,
                dropout_p=0.1
            )
        except ImportError:
            # 如果导入失败，使用简化的版本
            self.frame_translator = self._create_simple_translator(in_channels, out_channels)
        
        # 轻量级时序融合（可选，在最后融合时序信息）
        self.temporal_fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
    
    def _create_simple_translator(self, C_in, C_out):
        """创建简化的逐帧转换器"""
        return nn.Sequential(
            nn.Conv2d(C_in, 196, kernel_size=3, padding=1),
            nn.GroupNorm(4, 196),
            nn.GELU(),
            nn.Conv2d(196, C_out, kernel_size=3, padding=1),
            nn.GroupNorm(4, C_out)
        )

    def forward(self, x):
        """
        输入: [B, T, K, H, W]
        输出: [B, T, K, H, W]
        """
        B, T, K, H, W = x.size()
        
        # 逐帧处理
        frame_outputs = []
        for t in range(T):
            frame_t = x[:, t]  # [B, K, H, W]
            frame_out = self.frame_translator(frame_t)  # [B, K, H, W]
            frame_outputs.append(frame_out)
        
        # 堆叠时间维度
        x_out = torch.stack(frame_outputs, dim=1)  # [B, T, K, H, W]
        
        # 轻量级时序融合（可选）
        # 对每个空间位置，在时间维度上进行1D卷积
        x_out_reshaped = x_out.permute(0, 3, 4, 2, 1).contiguous()  # [B, H, W, K, T]
        x_out_reshaped = x_out_reshaped.view(B * H * W, K, T)  # [B*H*W, K, T]
        x_fused = self.temporal_fusion(x_out_reshaped)  # [B*H*W, K, T]
        x_fused = x_fused.view(B, H, W, K, T).permute(0, 4, 3, 1, 2).contiguous()  # [B, T, K, H, W]
        
        # 残差连接：融合后的特征 + 原始逐帧特征
        x_out = x_out + 0.1 * x_fused  # 小权重融合，避免破坏逐帧特征
        
        return x_out

# 保留原版本作为对比
class SparseU_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=3):
        super(SparseU_Net, self).__init__()

        # ===== 编码器部分 =====
        # 修复：保持时间维度不变，只对空间维度下采样
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1)),  # T不变
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)),  # T不变，空间下采样
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1)),  # T不变，空间下采样
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        # ===== Bottleneck: 多尺度融合 + 多个残差块 =====
        self.multi_scale_fusion = WeightedMultiScaleFeatureFusion(256, 256)
        self.res_blocks = nn.Sequential(*[ResidualBlock3D(256) for _ in range(num_res_blocks)])
        
        # ===== 解码器部分 =====
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),  # T不变
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),  # T不变
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.dec1 = nn.Conv3d(64, out_channels, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1))  # T不变

    def forward(self, x):
        """
        输入: [B, T, K, H, W]
        输出: [B, T, K, H, W]
        """
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, K, T, H, W]

        # 编码（时间维度保持不变）
        e1 = self.enc1(x)  # [B, 64, T, H, W]
        e2 = self.enc2(e1)  # [B, 128, T, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 256, T, H/4, W/4]
        
        # Bottleneck
        e3_fused = self.multi_scale_fusion(e3)
        bottleneck = self.res_blocks(e3_fused)

        # 解码 + skip（时间维度匹配）
        d3 = self.dec3(bottleneck)  # [B, 128, T, H/2, W/2]
        # 注意：skip connection需要上采样e2到相同空间尺寸
        if d3.shape[3:] != e2.shape[3:]:
            e2_up = F.interpolate(e2, size=d3.shape[3:], mode='trilinear', align_corners=False)
            d3 = d3 + e2_up
        else:
            d3 = d3 + e2
            
        d2 = self.dec2(d3)  # [B, 64, T, H, W]
        if d2.shape[3:] != e1.shape[3:]:
            e1_up = F.interpolate(e1, size=d2.shape[3:], mode='trilinear', align_corners=False)
            d2 = d2 + e1_up
        else:
            d2 = d2 + e1
            
        out = self.dec1(d2)  # [B, out, T, H, W]
        
        return out.permute(0, 2, 1, 3, 4).contiguous()  # 恢复为 [B, T, K, H, W]

# T:50版
class SparseU_Net2(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=3):
        super(SparseU_Net, self).__init__()

        # ===== 编码器部分 =====
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=2, stride=1, padding=1),  # 保持 T 和 H, W
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),  # 保持 T 和 H, W
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),  # 保持 T 和 H, W
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        # ===== Bottleneck: 多尺度融合 + 多个残差块 =====
        self.multi_scale_fusion = WeightedMultiScaleFeatureFusion(256, 256)
        self.res_blocks = nn.Sequential(*[ResidualBlock3D(256) for _ in range(num_res_blocks)])

        # ===== 解码器部分 =====
        self.dec3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),  # 保持 T 和 H, W
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),  # 保持 T 和 H, W
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.dec1 = nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1)  # 保持 T 和 H, W

    def forward(self, x):
        """
        输入: [B, T, K, H, W]
        输出: [B, T, K, H, W]
        """
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, K, T, H, W]

        # 编码
        e1 = self.enc1(x)   # [B, 64, T, H, W]
        e2 = self.enc2(e1)  # [B, 128, T, H, W]
        e3 = self.enc3(e2)  # [B, 256, T, H, W]

        # Bottleneck
        e3_fused = self.multi_scale_fusion(e3)
        e3_fused = self.res_blocks(e3_fused)

        # 解码 + skip
        d3 = self.dec3(e3_fused) + e2  # [B, 128, T, H, W]
        d2 = self.dec2(d3) + e1  # [B, 64, T, H, W]
        out = self.dec1(d2)  # [B, out, T, H, W]
        print(e1.size(),e2.size(),e3.size())
        input()
        return out.permute(0, 2, 1, 3, 4).contiguous()  # 保持 T 维度不变
# 时序膨胀压缩
class SparseU_Net0(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=3):
        super(SparseU_Net0, self).__init__()

        # ===== 编码器部分 =====
        self.enc1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 32, kernel_size=3, stride=(1, 1, 1), padding=1),  # 对 H 和 W 下采样
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.ConvTranspose3d(32, 64, kernel_size=2, stride=(2, 2, 2)),  # H 和 W 下采样，T -> 4T
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.ConvTranspose3d(64, 128, kernel_size=2, stride=(2, 2, 2)),  # H 和 W 下采样，T -> 8T
            nn.BatchNorm3d(128),
            nn.ReLU()
        )

        # ===== Bottleneck: 多尺度融合 + 多个残差块 =====
        self.multi_scale_fusion = WeightedMultiScaleFeatureFusion(128, 128)
        self.res_blocks = nn.Sequential(*[ResidualBlock3D(128) for _ in range(num_res_blocks)])

        # ===== 解码器部分 =====
        self.dec3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=2, stride=(2, 2, 2), padding=0),  # T 和 H/W 都下采样
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=2, stride=(2, 2, 2), padding=0),  # T 和 H/W 都下采样
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.dec1 = nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        输入: [B, T, K, H, W]
        输出: [B, T, K, H, W]
        """
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, K, T, H, W]
        # print("Input size: ", x.size())

        # 编码
        e1 = self.enc1(x)   # [B, 64, 2T, H, W]
        e2 = self.enc2(e1)  # [B, 128, 4T, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 256, 8T, H/4, W/4]
        
        # Bottleneck
        e3_fused = self.multi_scale_fusion(e3)
        e3_fused = self.res_blocks(e3_fused)

        # 解码 + skip
        d3 = self.dec3(e3_fused) + e2  # [B, 128, 4T, H/2, W/2] (T -> 4T)

        d2 = self.dec2(d3) + e1  # [B, 64, 2T, H, W] (T -> 2T)
        # print(e1.size(),e2.size(),e3.size(),d3.size(),d2.size())
        # input()
        out = self.dec1(d2)  # [B, out, T, H, W] (恢复 T)

        # print(e1.size(),e2.size(),e3.size(),d3.size(),d2.size(),out.size())
        # input()
        return out.permute(0, 2, 1, 3, 4).contiguous()  # 恢复为 [B, T, K, H, W]

class SparseU_Net1(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=3):
        super(SparseU_Net1, self).__init__()

        # ===== 编码器部分 =====
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  # 对 H 和 W 进行处理
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H 和 W 下采样
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # H 和 W 下采样
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # ===== Bottleneck: 多尺度融合 + 多个残差块 =====
        self.multi_scale_fusion = WeightedMultiScaleFeatureFusion(256, 256)
        self.res_blocks = nn.Sequential(*[ResidualBlock3D(256) for _ in range(num_res_blocks)])

        # ===== 解码器部分 =====
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # H 和 W 上采样
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # H 和 W 上采样
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        输入: [B, T, K, H, W]
        输出: [B, T, K, H, W]
        """
        # 输入的维度是 [B, T, K, H, W]，我们需要逐T处理
        B, T, K, H, W = x.shape
        x = rearrange(x, 'b t k h w -> (b t) k h w')  # 将 T 展开到批次维度

        # 编码
        e1 = self.enc1(x)   # [B*T, 64, H, W]
        e2 = self.enc2(e1)  # [B*T, 128, H/2, W/2]
        e3 = self.enc3(e2)  # [B*T, 256, H/4, W/4]

        # Bottleneck
        e3_fused = self.multi_scale_fusion(e3)
        e3_fused = self.res_blocks(e3_fused)

        # 解码
        d3 = self.dec3(e3_fused)  # [B*T, 128, H/2, W/2]
        d2 = self.dec2(d3)  # [B*T, 64, H, W]
        out = self.dec1(d2)  # [B*T, out, H, W]

        # 恢复为 [B, T, K, H, W]
        out = rearrange(out, '(b t) k h w -> b t k h w', b=B, t=T)  
        # print(e1.size(),e2.size(),e3.size(),d3.size(),d2.size(),out.size())
        # input()
        return out
# 有bug
class SparseU_Net8(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=3):
        super(SparseU_Net, self).__init__()

        # ===== 编码器部分 =====
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),  # T 保持不变
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.ConvTranspose3d(64, 128, kernel_size=2, stride=(2, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=2, stride=(1, 2, 2)),  
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        # ===== Bottleneck: 多尺度融合 + 多个残差块 =====
        self.multi_scale_fusion = WeightedMultiScaleFeatureFusion(256, 256)
        self.res_blocks = nn.Sequential(*[ResidualBlock3D(256) for _ in range(num_res_blocks)])

        # ===== 解码器部分 =====
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=2, stride=(2, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.dec1 = nn.Conv3d(64, out_channels, kernel_size=2, stride=(2, 1, 1))

    def forward(self, x):
        """
        输入: [B, T, K, H, W]
        输出: [B, T, K, H, W]
        """
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, K, T, H, W]
        print(x.size())

        # 编码
        e1 = self.enc1(x)   # [B, 64, 2T, H, W]
        e2 = self.enc2(e1)  # [B, 128, 4T, H/2, W/2]
        e3 = self.enc3(e2)  # [B, 512, 4T, H/4, W/4]

        # Bottleneck
        e3_fused = self.multi_scale_fusion(e3)
        e3_fused = self.res_blocks(e3_fused)

        # 解码 + skip
        d3 = self.dec3(e3_fused) + e2
        d2 = self.dec2(d3) + e1
        out = self.dec1(d2)  # [B, out, T, H, W]
        print(e1.size(),e2.size(),e3.size(),d3.size(),d2.size(),out.size())
        input()

        return out.permute(0, 2, 1, 3, 4).contiguous()
from math import exp
from torch.cuda.amp import autocast
# ===== SSIM =====
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=1):
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=1) - mu1_mu2
    C1 = (0.01*val_range)**2
    C2 = (0.03*val_range)**2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# ===== Focal Loss =====
def focal_loss(x_recon, x_target, alpha=1.0, gamma=2.0, threshold=0.71):
    x_target_01 = x_target if x_target.max() <= 1 else x_target / 255.0
    x_recon = x_recon if x_recon.max() <= 1 else x_recon / 255.0
    diff = (x_recon - x_target_01) ** 2
    p_t = torch.abs(x_recon - x_target_01)
    focal_factor = (1 - p_t) ** gamma
    weight_mask = (x_target_01 >= threshold).float() * alpha + (x_target_01 < threshold).float()
    return (weight_mask * focal_factor * diff).mean()

# ===== Edge Loss (Sobel) =====
def sobel_edges(x):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, focal_alpha=1.0, focal_gamma=2.5, threshold=0.71):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        self.threshold = threshold

    def forward(self, pred, target):
        with autocast():
            # 1) Focal Loss (逐帧计算)
            focal_loss_val = 0
            mse_loss_val = segmented_weighted_loss(pred, target)
            for t in range(pred.shape[2]):
                pred_t = pred[:, :, t, :, :]  # [B, C, H, W]
                target_t = target[:, :, t, :, :]  # [B, C, H, W]
                focal_loss_val += focal_loss(pred_t, target_t, alpha=self.focal_alpha, gamma=self.focal_gamma, threshold=self.threshold)
            focal_loss_val /= pred.shape[2]

            # 2) Segmented Weighted Loss (MSE)


            # 3) SSIM Loss (逐帧计算)
            ssim_loss_total = 0
            window = create_window(11, pred.shape[1]).to(pred.device)
            for t in range(pred.shape[2]):
                pred_t = pred[:, :, t, :, :]  # [B, C, H, W]
                target_t = target[:, :, t, :, :]  # [B, C, H, W]
                ssim_val = ssim(pred_t, target_t, window=window, val_range=1)
                ssim_loss_total += 1 - ssim_val
            ssim_loss = ssim_loss_total / pred.shape[2]

            # 4) Edge Loss (逐帧计算)
            edge_loss_total = 0
            for t in range(pred.shape[2]):
                pred_t = pred[:, :, t, :, :].mean(1, keepdim=True)  # [B, 1, H, W]
                target_t = target[:, :, t, :, :].mean(1, keepdim=True)  # [B, 1, H, W]
                pred_edge = sobel_edges(pred_t)
                target_edge = sobel_edges(target_t)
                edge_loss_total += F.mse_loss(pred_edge, target_edge)
            edge_loss = edge_loss_total / pred.shape[2]

            # 总损失
            total_loss = (self.alpha * mse_loss_val +
                          self.beta * ssim_loss +
                          self.gamma * edge_loss +
                          self.focal_alpha * focal_loss_val)

        return total_loss, {
            "mse": mse_loss_val.item(),
            "focal_loss": focal_loss_val.item(),
            "ssim": ssim_loss.item(),
            "edge": edge_loss.item()
        }
class Channelattention(nn.Module):
    def __init__(self, reduction=16):
        super(Channelattention, self).__init__()
        # 我们只在 3 个 RGB 通道上做注意力，所以 in_channels=3
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 只对 K,H,W 做 pooling，保留 RGB 维
        self.fc = nn.Sequential(
            nn.Linear(3, 3 // reduction if 3 // reduction > 0 else 1, bias=False),
            nn.ReLU(),
            nn.Linear(3 // reduction if 3 // reduction > 0 else 1, 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B,T,3,K,H,W]
        B,T,C,K,H,W = x.shape  # C=3
        # 对 K,H,W 做池化，得到每个通道的全局特征
        y = self.avg_pool(x.view(B*T, C, K, H, W)).view(B,T,C)  # [B,T,3]
        # FC 生成注意力权重
        y = self.fc(y)  # [B,T,3]
        y = y.view(B,T,C,1,1,1)  # broadcast
        return (x * y).sum(dim=2)  # 融合 RGB → [B,T,K,H,W]
class SSCVAE(nn.Module):
    def __init__(self,
                 in_channels_sate,# 输入通道数为3：卫星
                 in_channels_radar,# 通道数为1：雷达图像
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,  # 编码器的输出通道数，它定义了在下采样后的特征图的深度。这个值决定了从输入到潜在空间的维度。通常比 hid_channels_2 更大，例如 128 或 256。
                 down_samples,
                 num_groups,
                 num_atoms,# 字典的原子个数，与num_dims相匹配
                 num_dims, # 每个原子的维度
                 num_iters,# 决定稀疏编码的收敛速度和精度
                 device):
        super(SSCVAE, self).__init__()
        # self._encoder_sate_r = Encoder(in_channels=1, 
        #                         hid_channels_1=hid_channels_1,
        #                         hid_channels_2=hid_channels_2,
        #                         out_channels=out_channels,
        #                         down_samples=down_samples,
        #                         num_groups=num_groups)
        # self._encoder_sate_g = Encoder(in_channels=1, 
        #                 hid_channels_1=hid_channels_1,
        #                 hid_channels_2=hid_channels_2,
        #                 out_channels=out_channels,
        #                 down_samples=down_samples,
        #                 num_groups=num_groups)
        # self._encoder_sate_b = Encoder(in_channels=1, 
        #                 hid_channels_1=hid_channels_1,
        #                 hid_channels_2=hid_channels_2,
        #                 out_channels=out_channels,
        #                 down_samples=down_samples,
        #                 num_groups=num_groups)
        self._encoder_sate = Encoder(in_channels=in_channels_sate,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)
        
        # self._decoder_sate_r = Decoder(in_channels=1, 
        #                                hid_channels_1=hid_channels_1,
        #                         hid_channels_2=hid_channels_2,
        #                         out_channels=out_channels,
        #                         up_samples=down_samples,
        #                         num_groups=num_groups)
        # self._decoder_sate_g = Decoder(in_channels=1, 
        #                         hid_channels_1=hid_channels_1,
        #                 hid_channels_2=hid_channels_2,
        #                 out_channels=out_channels,
        #                 up_samples=down_samples,
        #                 num_groups=num_groups)
        # self._decoder_sate_b = Decoder(in_channels=1, 
        #                         hid_channels_1=hid_channels_1,
        #                 hid_channels_2=hid_channels_2,
        #                 out_channels=out_channels,
        #                 up_samples=down_samples,
        #                 num_groups=num_groups)

        self._decoder_sate = Decoder(in_channels=in_channels_sate,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)# 解码为三通道

        self._encoder_radar = Encoder(in_channels=in_channels_radar,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)

        self._decoder_radar= Decoder(in_channels=in_channels_radar,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)# 解码为三通道

        self._LISTA = AttentiveLISTA_Time(num_atoms=num_atoms,
                                     num_dims=num_dims,
                                     num_iters=num_iters,
                                     device=device,)
        # 使用改进后的 Translator
        # 修复版本：保持时间维度不变，避免膨胀
        # self._mlp = SparseU_Net(in_channels=128, out_channels=128)
        # 或者使用逐帧处理版本（推荐，性能更好）
        self._mlp = SparseU_Net_Fixed(in_channels=128, out_channels=128)
        #self._mlp = WeightedMultiScaleFeatureFusion(128, 128)
        self._channel_attention = Channelattention(reduction=16)
        # 组合损失
        self.criterion = CombinedLoss()

    def generation(self, input_z):
        ex = self._LISTA.generation(input_z)
        x_generation = self._decoder_radar(ex)

        x_generation = torch.sigmoid(x_generation)
        return ex,x_generation
    
    def get_dict(self):
        return self._LISTA.get_dict()

    def to_BTCHW(self,x_bchwt):
        # [B,C,H,W,T] -> [B,T,C,H,W]
        return x_bchwt.permute(0, 4, 1, 2, 3).contiguous()

    def to_BCHWT(self,x_btchw):
        # [B,T,C,H,W] -> [B,C,H,W,T]
        return x_btchw.permute(0, 2, 3, 4, 1).contiguous() 
       
    def forward(self, satellite_seq, radar_seq):
        # 0) [B,C,H,W,T] -> [B,T,C,H,W]
        sat_btchw = self.to_BTCHW(satellite_seq)  # 转换为 [B, T, C, H, W]
        radar_btchw = self.to_BTCHW(radar_seq)

        # 1) 编码每个通道
        ex_sate_seq = self._encoder_sate(sat_btchw)   # [B,T,D,h,w]
        ex_radar_seq = self._encoder_radar(radar_btchw)  # [B,T,D,h,w]

        # 2) LISTA 稀疏编码
        z_sate_seq, exrec_sate_seq, dictionary = self._LISTA(ex_sate_seq)  # [B,T,K,h,w], [B,T,D,h,w]
        z_radar_seq, exrec_radar_seq, _ = self._LISTA(ex_radar_seq)  # [B,T,K,h,w], [B,T,D,h,w]

        z_mlp_seq  = self._mlp(z_sate_seq)                             # 再进入 SparseU_Net

        # 4) 逐帧生成潜在特征
        ex_trans_seq,_ = self.generation(z_mlp_seq)   # ex_t:[B,D,h,w], 只需要潜在特征，不需要重建结果
        # 6) 使用解码器生成每一帧的重建结果
        x_recon_list = self._decoder_radar(ex_trans_seq)
        x_recon_list = torch.sigmoid(x_recon_list)

        # 7) 转换成 [B, Cr, H, W, T] 格式
        x_recon_trans_bchwt = self.to_BCHWT(x_recon_list)  # [B, Cr, H, W, T]

        # 8) 重建损失（雷达像素级）
        reconstruction_loss = segmented_weighted_loss(x_recon_trans_bchwt, radar_seq)
        #reconstruction_loss, loss_dict = self.criterion(x_recon_trans_bchwt, radar_seq)
        latent_trans_loss = F.mse_loss(ex_trans_seq, ex_radar_seq.detach()) 
        sparsity_loss=1
        z_student = z_mlp_seq # 学生（目标域、经时序） 
        z_teacher = z_radar_seq # 老师（真实雷达稀疏编码，停止梯度） 
        mse = F.mse_loss(z_student, z_teacher) 

        latent_dist_loss = mse

        # 返回时序处理后的重建结果
        return x_recon_trans_bchwt, ex_trans_seq, latent_dist_loss, latent_trans_loss, reconstruction_loss, dictionary, sparsity_loss
    
    # def forward(self, satellite_seq, radar_seq):
    #     # 0) [B,C,H,W,T] -> [B,T,C,H,W]
    #     sat_btchw = self.to_BTCHW(satellite_seq)  # 转换为 [B, T, C, H, W]
    #     radar_btchw = self.to_BTCHW(radar_seq)

    #     # 1) 编码每个通道
    #     ex_sate_r = self._encoder_sate_r(sat_btchw[:,:,0,:,:].unsqueeze(2))   # [B,T,D,h,w]
    #     ex_sate_g = self._encoder_sate_g(sat_btchw[:,:,1,:,:].unsqueeze(2))   # [B,T,D,h,w]
    #     ex_sate_b = self._encoder_sate_b(sat_btchw[:,:,2,:,:].unsqueeze(2))   # [B,T,D,h,w]
    #     ex_radar_seq = self._encoder_radar(radar_btchw)  # [B,T,D,h,w]

    #     # 2) LISTA 稀疏编码
    #     z_sate_r, exrec_sate_r, dictionary = self._LISTA(ex_sate_r)  # [B,T,K,h,w], [B,T,D,h,w]
    #     z_sate_g, exrec_sate_g, _ = self._LISTA(ex_sate_g)  # [B,T,K,h,w], [B,T,D,h,w]
    #     z_sate_b, exrec_sate_b, _ = self._LISTA(ex_sate_b)  # [B,T,K,h,w], [B,T,D,h,w]
    #     z_radar_seq, exrec_radar_seq, _ = self._LISTA(ex_radar_seq)  # [B,T,K,h,w], [B,T,D,h,w]

    #     z_sate_seq = torch.stack((z_sate_r, z_sate_g, z_sate_b), dim=2)  # [B,T,3,K,H,W]

    #     z_sate_seq = self._channel_attention(z_sate_seq)               # 在跨模态前 re-weight
    #     z_mlp_seq  = self._mlp(z_sate_seq)                             # 再进入 SparseU_Net

    #     # 4) 逐帧生成潜在特征
    #     ex_trans_seq,_ = self.generation(z_mlp_seq)   # ex_t:[B,D,h,w], 只需要潜在特征，不需要重建结果
    #     # 6) 使用解码器生成每一帧的重建结果
    #     x_recon_list = self._decoder_radar(ex_trans_seq)
    #     x_recon_list = torch.sigmoid(x_recon_list)

    #     # 7) 转换成 [B, Cr, H, W, T] 格式
    #     x_recon_trans_bchwt = self.to_BCHWT(x_recon_list)  # [B, Cr, H, W, T]

    #     # 8) 重建损失（雷达像素级）
    #     reconstruction_loss = segmented_weighted_loss(x_recon_trans_bchwt, radar_seq)
    #     #reconstruction_loss, loss_dict = self.criterion(x_recon_trans_bchwt, radar_seq)
    #     latent_trans_loss = F.mse_loss(ex_trans_seq, ex_radar_seq.detach()) 
    #     sparsity_loss=1
    #     z_student = z_mlp_seq # 学生（目标域、经时序） 
    #     z_teacher = z_radar_seq # 老师（真实雷达稀疏编码，停止梯度） 
    #     mse = F.mse_loss(z_student, z_teacher) 

    #     latent_dist_loss = mse

    #     # 返回时序处理后的重建结果
    #     return x_recon_trans_bchwt, ex_trans_seq, latent_dist_loss, latent_trans_loss, reconstruction_loss, dictionary, sparsity_loss