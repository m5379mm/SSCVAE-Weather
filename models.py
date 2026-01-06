import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math
from geomloss import SamplesLoss
from enhanced_losses import EnhancedReconstructionLoss
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

        # [B, C, H, W] -> [B, C', H, W]
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=hid_channels_1,
                                 kernel_size=3, stride=1, padding=1)
        
        # [B, C', H, W] -> [B, C'', h, w]
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

        # [B, C'', h, w] -> [B, C'', h, w]
        self._res_1 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock(in_channels=hid_channels_2,
                                        hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        
        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_2)
        self._swish = Swish()

        # [B, C'', h, w] -> [B, n, h, w]
        self._conv_2 = nn.Conv2d(in_channels=hid_channels_2,
                                 out_channels=out_channels,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self._conv_1(x)

        for layer in self._down_samples:
            x = layer(x)

        x = self._res_1(x)
        x = self._non_local(x)
        x = self._res_2(x)

        x = self._group_norm(x)
        x = self._swish(x)
        x = self._conv_2(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 up_samples,
                 num_groups):
        super(Decoder, self).__init__()

        # [B, n, h, w] -> [B, C'', h, w]
        self._conv_1 = nn.Conv2d(in_channels=out_channels,
                                 out_channels=hid_channels_2,
                                 kernel_size=3, stride=1, padding=1)

        # [B, C'', h, w] -> [B, C'', h, w]
        self._res_1 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock(in_channels=hid_channels_2,
                                        hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock(in_channels=hid_channels_2,
                                    hid_channels=hid_channels_2 // 2)
        
        # [B, C'', h, w] -> [B, C', H, W]
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
        
        # [B, C', H, W] -> [B, C', H, W]
        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_1)
        self._swish = Swish()

        # [B, C', H, W] -> [B, C, H, W]
        self._conv_2 = nn.Conv2d(in_channels=hid_channels_1,
                                 out_channels=in_channels,
                                 kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        #print('*')
        x = self._conv_1(x)
        #print('/')
        x = self._res_1(x)
        x = self._non_local(x)
        x = self._res_2(x)

        for layer in self._up_samples:
            x = layer(x)

        x = self._group_norm(x)
        x = self._swish(x)
        x = self._conv_2(x)

        return x

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

#稀疏编码的迭代算法
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


# 时间注意力模块：对多帧序列在时间维度上进行加权
class TemporalAttention(nn.Module):
    def __init__(self, num_steps, channels=None):
        super().__init__()
        self.num_steps = num_steps
        
        # 增强版：使用卷积网络学习时序依赖
        if channels is None:
            # 简化版：仅使用可学习的权重向量
            self.attn = nn.Parameter(torch.randn(1, num_steps, 1, 1, 1))  # [1, T, 1, 1, 1]
            self.use_conv = False
        else:
            # 增强版：使用1D卷积学习时序模式
            self.use_conv = True
            self.temporal_conv = nn.Sequential(
                nn.Conv3d(channels, channels // 4, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=1),
                nn.BatchNorm3d(channels // 4),
                nn.ReLU(),
                nn.Conv3d(channels // 4, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            )
        
        self.softmax = nn.Softmax(dim=1)  # softmax 在时间维度

    def forward(self, x):  # x: [B, T, C, H, W]
        if self.use_conv:
            # 增强版：通过卷积学习时序注意力
            B, T, C, H, W = x.size()
            x_perm = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            attn_logits = self.temporal_conv(x_perm)  # [B, 1, T, H, W]
            attn_logits = attn_logits.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
            attn_scores = self.softmax(attn_logits)  # [B, T, 1, H, W]
            x_attended = x * attn_scores.expand_as(x)  # [B, T, C, H, W]
        else:
            # 简化版：使用可学习权重
            attn_scores = self.attn.expand(x.size(0), -1, x.size(2), x.size(3), x.size(4))  # [B, T, C, H, W]
            attn_scores = self.softmax(attn_scores)  # [B, T, C, H, W]
            x_attended = x * attn_scores  # [B, T, C, H, W]
        
        return x_attended


class AttentiveLISTA(nn.Module):
    def __init__(self,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device,
                 use_time_attention=False):  # 添加参数控制是否使用时间注意力
        super(AttentiveLISTA, self).__init__()

        self._num_atoms = num_atoms
        self._num_dims = num_dims
        self._device = device
        self.use_time_attention = use_time_attention

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

        # 添加时间注意力模块
        if self.use_time_attention:
            # ✅ 使用增强版：传入 channels 参数以启用卷积注意力
            # 这里 num_dims 是潜在空间的通道数
            self._time_attention = TemporalAttention(num_steps=1, channels=num_atoms)

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
        print("Dictionary initialized with pretrained values.")

    def soft_thresh(self, x, theta):
        return torch.sign(x) * torch.max(torch.abs(x) - theta, self._Zero)

    def generation(self, input_z):
        input_z = input_z.permute(0, 2, 3, 1).contiguous()  # [B, K, H, W] -> [B, H, W, K]
        x_recon = torch.matmul(input_z, self._Dict.t())  # [B, H, W, K] * [K, D] -> [B, H, W, D]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B, H, W, D] -> [B, D, H, W]
        return x_recon

    def forward(self, x):
        # 输入保证是 [B, T, D, H, W]
        B, T, D, H, W = x.size()
        x = x.view(B * T, D, H, W)  # 展平时间维度
        
        l = self._conv(x)  # [B*T, D, H, W] -> [B*T, K, H, W]
        l = self._res1(l)
        l = self._res2(l)
        
        # 时间注意力：恢复时间维度后应用
        if self.use_time_attention:
            _, K, _, _ = l.size()
            l = l.view(B, T, K, H, W)  # [B, T, K, H, W]
            self._time_attention.num_steps = T
            l = self._time_attention(l)  # [B, T, K, H, W]
            l = l.view(B * T, K, H, W)  # 再展平
        
        l = self._cbam(l) / self._L

        l = l.permute(0, 2, 3, 1).contiguous()  # [B*T, K, H, W] -> [B*T, H, W, K]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B*T, D, H, W] -> [B*T, H, W, D]
        
        S = self._Identity - (1 / self._L) * self._Dict.t().mm(self._Dict)  # [K, K]
        S = S.t()  # [K, K]

        y = torch.matmul(x, self._Dict)  # [B*T, H, W, D] * [D, K] -> [B*T, H, W, K]

        z = self.soft_thresh(y, l)  # [B*T, H, W, K]
        for t in range(self._num_iters):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self._L) * y, l)

        x_recon = torch.matmul(z, self._Dict.t())  # [B*T, H, W, K] * [K, D] -> [B*T, H, W, D]

        z = z.permute(0, 3, 1, 2).contiguous()  # [B*T, H, W, K] -> [B*T, K, H, W]
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # [B*T, H, W, D] -> [B*T, D, H, W]
        
        # 恢复时间维度
        z = z.view(B, T, -1, H, W)  # [B, T, K, H, W]
        x_recon = x_recon.view(B, T, -1, H, W)  # [B, T, D, H, W]

        return z, x_recon, self._Dict

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y
# class Residual(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
#                  dropout_p=0.0, norm='bn', groups=8, padding_mode='reflect'):
#         super().__init__()
#         Norm = (lambda c: nn.GroupNorm(groups, c)) if norm=='gn' else (lambda c: nn.BatchNorm2d(c))
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
#                                bias=False, padding_mode=padding_mode)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding,
#                                bias=False, padding_mode=padding_mode)
#         self.n1 = Norm(out_channels)
#         self.n2 = Norm(out_channels)
#         self.act = nn.GELU()
#         self.drop_p = dropout_p

#         self.match = (nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             Norm(out_channels)
#         ) if in_channels != out_channels else nn.Identity())

#     def forward(self, x):
#         r = x
#         x = self.act(self.n1(self.conv1(x)))
#         if self.drop_p > 0: x = F.dropout2d(x, p=self.drop_p, training=self.training)
#         x = self.n2(self.conv2(x))
#         if self.drop_p > 0: x = F.dropout2d(x, p=self.drop_p, training=self.training)
#         x = self.act(x + self.match(r))
#         return x


class TranslatorWithResidual0(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7],
                 norm='bn', groups=8, dropout_p=0.0, padding_mode='reflect'):
        super().__init__()
        Norm = (lambda c: nn.GroupNorm(groups, c)) if norm=='gn' else (lambda c: nn.BatchNorm2d(c))

        self.initial = Residual(C_in, C_hid, 3, 1, 1, dropout_p=dropout_p,
                                norm=norm, groups=groups, padding_mode=padding_mode)

        self.branches = nn.ModuleList([
            Residual(C_hid, C_out, kernel_size=k, stride=1, padding=k//2,
                     dropout_p=dropout_p, norm=norm, groups=groups, padding_mode=padding_mode)
            for k in incep_ker
        ])

        self.merge = nn.Sequential(
            nn.Conv2d(len(incep_ker)*C_out, C_out, kernel_size=1, bias=False),
            Norm(C_out),
            nn.GELU(),                    # <- 新增激活
            # 可选：nn.Dropout2d(p=0.1)
        )

        # 全局残差（通道对齐+标定+门控）
        self.skip = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            Norm(C_out)
        )
        self.skip_gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        x0 = self.skip(x)
        h  = self.initial(x)
        h  = torch.cat([b(h) for b in self.branches], dim=1)
        h  = self.merge(h)
        y  = h + self.skip_gate * x0

        # 掩膜（如果有缺测/海面等）——可显著减少背景“花”
        if mask is not None:
            y = y.masked_fill(mask, float('nan'))
        return y

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_p=0.1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_p)
        
        # If input and output channels don't match, add a 1x1 convolution to match the dimensions
        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        # Save the input for the residual connection
        residual = x

        # First convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.dropout(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        #x = self.dropout(x) 

        # If the input and output channels don't match, adjust the residual
        if self.match_channels:
            residual = self.match_channels(residual)

        # Add the residual connection
        x += residual
        x = self.relu(x)
        return x

# 目前最优
class TranslatorWithResidual(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7], groups=4, dropout_p=0.1):
        super(TranslatorWithResidual, self).__init__()

        # Initial convolution block with residual
        self.initial = Residual(C_in, C_hid, kernel_size=3, stride=1, padding=1, dropout_p=dropout_p)

        # Branches for multi-scale features (inception-style)
        self.branches = nn.ModuleList([
            Residual(C_hid, C_out, kernel_size=k, stride=1, padding=k//2, dropout_p=dropout_p)
            for k in incep_ker
        ])

        # Merge layers for final output
        self.merge = nn.Sequential(
            nn.Conv2d(len(incep_ker) * C_out, C_out, kernel_size=1),
            nn.BatchNorm2d(C_out),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        residual = self.initial(x)  # 保存初步特征
        
        branch_outputs = [branch(residual) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.merge(x)

        return x

def segmented_weighted_loss(x_recon, x_target):
    """
    x_recon, x_target: shape [B, C, H, W, T]
    """
    # 归一化处理（支持 0-1 或 0-255 输入）
    x_target_255 = x_target * 255 if x_target.max() <= 1 else x_target
    x_recon_255 = x_recon * 255 if x_recon.max() <= 1 else x_recon

    weights = torch.ones_like(x_target_255)

    # 色阶分段权重（可以自定义）
    weights[(x_target_255 >= 16) & (x_target_255 < 181)] = 1.0
    weights[(x_target_255 >= 181) & (x_target_255 < 219)] = 5.0
    weights[(x_target_255 >= 219) & (x_target_255 <= 255)] = 10.0

    abs_diff = torch.abs(x_recon - x_target)

    # 帧级平均（可视为 temporal attention）
    frame_losses = (weights * abs_diff).flatten(1, -1).mean(dim=1)  # [B]
    loss = frame_losses.mean()
    return loss

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
                 device,
                 use_time_attention=False):  # 添加参数控制是否使用时间注意力
        super(SSCVAE, self).__init__()

        self._encoder_sate = Encoder(in_channels=in_channels_sate,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
                                num_groups=num_groups)

        self._decoder_sate = Decoder(in_channels=in_channels_sate,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                up_samples=down_samples,
                                num_groups=num_groups)

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
                                num_groups=num_groups)

        self._LISTA = AttentiveLISTA(num_atoms=num_atoms,
                                     num_dims=num_dims,
                                     num_iters=num_iters,
                                     device=device,
                                     use_time_attention=use_time_attention)

        self._mlp = TranslatorWithResidual(C_in=128, C_hid=196, C_out=128, incep_ker=[1,3,5])

    def generation(self, input_z):
        ex = self._LISTA.generation(input_z)  # [B, K, h, w] -> [B, D, h, w]
        x_generation = self._decoder_radar(ex)  # [B, D, h, w] -> [B, C, H, W]
        x_generation = torch.sigmoid(x_generation)
        return ex,x_generation
    
    def get_dict(self):
        return self._LISTA.get_dict()

    def forward(self, satellite, vil):
        # 支持单帧或多帧输入 satellite: [B, T, C, H, W] 或 [B, C, H, W]
        # vil: [B, T, C, H, W] 或 [B, C, H, W]
        is_multi_frame = (satellite.dim() == 5)
        
        if is_multi_frame:
            satellite = satellite.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, H, W, T] -> [B, T, C, H, W]
            vil = vil.permute(0, 4, 1, 2, 3).contiguous()
            
            B, T, C_sat, H, W = satellite.size()
            _, _, C_vil, _, _ = vil.size()
            
            # 展平时间维度进行编码
            satellite_flat = satellite.view(B * T, C_sat, H, W)
            vil_flat = vil.view(B * T, C_vil, H, W)
            
            ex = self._encoder_sate(satellite_flat)  # [B*T, D, h, w]
            ex_radar = self._encoder_radar(vil_flat)  # [B*T, D, h, w]
            
            # 恢复时间维度后送入 LISTA（LISTA内部会根据use_time_attention决定是否用时间注意力）
            _, D, h, w = ex.size()
            ex = ex.view(B, T, D, h, w)
            ex_radar = ex_radar.view(B, T, D, h, w)
        else:
            # 单帧输入直接编码
            ex = self._encoder_sate(satellite)  # [B, D, h, w]
            ex_radar = self._encoder_radar(vil)

        # LISTA 处理（多帧或单帧）
        z, ex_recon, dictionary = self._LISTA(ex)  # 输出可能是 [B, T, K, h, w] 或 [B, K, h, w]
        z_radar, ex_recon_radar, dictionary_radar = self._LISTA(ex_radar)

        # TranslatorWithResidual 2D逐帧转换
        if is_multi_frame:
            # 展平时间维度逐帧转换
            z_flat = z.view(B * T, -1, h, w)  # [B*T, K, h, w]
            z_trans_flat = self._mlp(z_flat)  # [B*T, K, h, w]
            z_trans = z_trans_flat.view(B, T, -1, h, w)  # [B, T, K, h, w]
        else:
            z_trans = self._mlp(z)  # [B, K, h, w]

        # 生成和解码
        if is_multi_frame:
            # 逐帧生成
            ex_trans_list = []
            x_recon_trans_list = []
            for t in range(T):
                ex_t, x_t = self.generation(z_trans[:, t])  # [B, D, h, w], [B, C, H, W]
                ex_trans_list.append(ex_t)
                x_recon_trans_list.append(x_t)
            ex_trans = torch.stack(ex_trans_list, dim=1)  # [B, T, D, h, w]
            x_recon_trans = torch.stack(x_recon_trans_list, dim=1)  # [B, T, C, H, W]
        else:
            ex_trans, x_recon_trans = self.generation(z_trans)

        reconstruction_loss = segmented_weighted_loss(x_recon_trans, vil)

        z_diff_loss = F.mse_loss(z_trans, z_radar)
        latent_dist_loss = z_diff_loss

        latent_trans_loss = torch.sum((ex_trans - ex_radar).pow(2), dim=1).mean()

        return x_recon_trans, z, latent_dist_loss, latent_trans_loss, reconstruction_loss, dictionary

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
                 device,
                 use_time_attention=False):  # 添加参数控制是否使用时间注意力
        super(SSCVAEDouble, self).__init__()

        self._encoder_sate = Encoder(in_channels=in_channels_sate,
                                hid_channels_1=hid_channels_1,
                                hid_channels_2=hid_channels_2,
                                out_channels=out_channels,
                                down_samples=down_samples,
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

        self._LISTA = AttentiveLISTA(num_atoms=num_atoms,
                                     num_dims=num_dims,
                                     num_iters=num_iters,
                                     device=device,
                                     use_time_attention=use_time_attention)

    def generation(self, input_z, is_sate=True):
        if is_sate:
            ex = self._LISTA.generation(input_z)  # [B, K, h, w] -> [B, D, h, w]
            x_generation = self._decoder_sate(ex)  # [B, D, h, w] -> [B, C, H, W]
        else:
            ex = self._LISTA.generation(input_z)
            x_generation = self._decoder_radar(ex)
        
        x_generation = torch.sigmoid(x_generation)
        return x_generation

    
    def get_dict(self):
        return self._LISTA.get_dict()

    def forward(self, satellite, vil):
        # 支持单帧或多帧输入
        is_multi_frame = (satellite.dim() == 5)
        
        if is_multi_frame:
            satellite = satellite.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, H, W, T] -> [B, T, C, H, W]
            vil = vil.permute(0, 4, 1, 2, 3).contiguous()
            
            B, T, C_sat, H, W = satellite.size()
            _, _, C_vil, _, _ = vil.size()
            
            # 展平时间维度进行编码
            satellite_flat = satellite.view(B * T, C_sat, H, W)
            vil_flat = vil.view(B * T, C_vil, H, W)
            
            ex_sate = self._encoder_sate(satellite_flat)  # [B*T, D, h, w]
            ex_radar = self._encoder_radar(vil_flat)  # [B*T, D, h, w]
            
            # 恢复时间维度
            _, D, h, w = ex_sate.size()
            ex_sate = ex_sate.view(B, T, D, h, w)
            ex_radar = ex_radar.view(B, T, D, h, w)
        else:
            # 单帧输入
            ex_sate = self._encoder_sate(satellite)  # [B, D, h, w]
            ex_radar = self._encoder_radar(vil)

        # LISTA 处理
        z_sate, ex_recon_sate, dictionary = self._LISTA(ex_sate)
        z_radar, ex_recon_radar, _ = self._LISTA(ex_radar)
        
        # 解码
        if is_multi_frame:
            # 展平解码
            ex_recon_sate_flat = ex_recon_sate.view(B * T, D, h, w)
            ex_recon_radar_flat = ex_recon_radar.view(B * T, D, h, w)
            
            x_recon_sate_flat = self._decoder_sate(ex_recon_sate_flat)
            x_recon_radar_flat = self._decoder_radar(ex_recon_radar_flat)
            
            # 恢复时间维度
            x_recon_sate = torch.sigmoid(x_recon_sate_flat.view(B, T, C_sat, H, W))
            x_recon_radar = torch.sigmoid(x_recon_radar_flat.view(B, T, C_vil, H, W))
        else:
            x_recon_sate = torch.sigmoid(self._decoder_sate(ex_recon_sate))
            x_recon_radar = torch.sigmoid(self._decoder_radar(ex_recon_radar))

        # 计算损失
        latent_loss_sate = torch.sum((ex_recon_sate - ex_sate).pow(2), dim=1).mean()
        latent_loss_radar = torch.sum((ex_recon_radar - ex_radar).pow(2), dim=1).mean()
        total_latent_loss = latent_loss_sate + latent_loss_radar

        return x_recon_sate, x_recon_radar, z_sate, z_radar, total_latent_loss, dictionary


# ============================================================================
# 🎬 3D 时空卷积模块 (用于真正的多帧时序建模)
# ============================================================================

# 3D 残差块
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super(ResidualBlock3D, self).__init__()
        
        self._block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_channels,
                      out_channels=hid_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=hid_channels,
                      out_channels=in_channels,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


# 3D 下采样（只在空间维度下采样，时间维度保持）
class DownSampleBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock3D, self).__init__()

        self._block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 只在 H,W 下采样
        )
    
    def forward(self, x):
        return self._block(x)


# 3D 上采样（只在空间维度上采样，时间维度保持）
class UpSampleBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock3D, self).__init__()

        self._block = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 只在 H,W 上采样
            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self._block(x)


# 3D 非局部注意力块
class NonLocalBlock3D(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super(NonLocalBlock3D, self).__init__()

        self.hid_channels = hid_channels
        self._conv_theta = nn.Conv3d(in_channels=in_channels,
                                     out_channels=hid_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self._conv_phi = nn.Conv3d(in_channels=in_channels,
                                   out_channels=hid_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self._conv_g = nn.Conv3d(in_channels=in_channels,
                                 out_channels=hid_channels,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self._soft_max = nn.Softmax(dim=1)
        self._conv_mask = nn.Conv3d(in_channels=hid_channels,
                                    out_channels=in_channels,
                                    kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, t, h, w = x.size()

        # [B, C, T, H, W] -> [B, C', THW] -> [B, THW, C']
        theta = self._conv_theta(x).view(b, self.hid_channels, -1).permute(0, 2, 1).contiguous()
        # [B, C, T, H, W] -> [B, C', THW]
        phi = self._conv_phi(x).view(b, self.hid_channels, -1)
        # [B, C, T, H, W] -> [B, C', THW] -> [B, THW, C']
        g = self._conv_g(x).view(b, self.hid_channels, -1).permute(0, 2, 1).contiguous()
        # [B, THW, C'] * [B, C', THW] = [B, THW, THW]
        mul_theta_phi = self._soft_max(torch.matmul(theta, phi))
        # [B, THW, THW] * [B, THW, C'] = [B, THW, C']
        mul_theta_phi_g = torch.matmul(mul_theta_phi, g)
        # [B, THW, C'] -> [B, C', THW] -> [B, C', T, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.hid_channels, t, h, w)
        # [B, C', T, H, W] -> [B, C, T, H, W]
        mask = self._conv_mask(mul_theta_phi_g)

        return x + mask


# 3D 通道注意力
class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction):
        super(ChannelAttention3D, self).__init__()
        self._avg_pool = nn.AdaptiveAvgPool3d(1)  # 全局平均池化
        self._max_pool = nn.AdaptiveMaxPool3d(1)  # 最大池化
        self._fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self._sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        avgOut = self._fc(self._avg_pool(x).view(b, c))
        maxOut = self._fc(self._max_pool(x).view(b, c))
        y = self._sigmoid(avgOut + maxOut).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


# 3D 空间注意力
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention3D, self).__init__()
        self._conv = nn.Conv3d(in_channels=2,
                               out_channels=1,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=(kernel_size - 1) // 2,
                               bias=False)
        self._sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avgOut = torch.mean(x, dim=1, keepdim=True)
        maxOut, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avgOut, maxOut], dim=1)
        y = self._sigmoid(self._conv(y))
        return x * y.expand_as(x)


# 3D CBAM
class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction, kernel_size):
        super(CBAM3D, self).__init__()
        self.ChannelAtt = ChannelAttention3D(in_channels, reduction)
        self.SpatialAtt = SpatialAttention3D(kernel_size)
    
    def forward(self, x):
        x = self.ChannelAtt(x)
        x = self.SpatialAtt(x)
        return x


# 3D 编码器
class Encoder3D(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups):
        super(Encoder3D, self).__init__()

        # [B, C, T, H, W] -> [B, C', T, H, W]
        self._conv_1 = nn.Conv3d(in_channels=in_channels,
                                 out_channels=hid_channels_1,
                                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        # [B, C', T, H, W] -> [B, C'', T, h, w]
        self._down_samples = nn.ModuleList()
        for i in range(down_samples):
            cur_in_channels = hid_channels_1 if i == 0 else hid_channels_2
            self._down_samples.append(
                ResidualBlock3D(in_channels=cur_in_channels,
                                hid_channels=cur_in_channels // 2)
            )
            self._down_samples.append(
                DownSampleBlock3D(in_channels=cur_in_channels,
                                  out_channels=hid_channels_2)
            )

        # [B, C'', T, h, w] -> [B, C'', T, h, w]
        self._res_1 = ResidualBlock3D(in_channels=hid_channels_2,
                                      hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock3D(in_channels=hid_channels_2,
                                          hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock3D(in_channels=hid_channels_2,
                                      hid_channels=hid_channels_2 // 2)
        
        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_2)
        self._swish = Swish()

        # [B, C'', T, h, w] -> [B, D, T, h, w]
        self._conv_2 = nn.Conv3d(in_channels=hid_channels_2,
                                 out_channels=out_channels,
                                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):  # [B, C, T, H, W]
        x = self._conv_1(x)

        for layer in self._down_samples:
            x = layer(x)

        x = self._res_1(x)
        x = self._non_local(x)
        x = self._res_2(x)

        # GroupNorm 需要特殊处理：[B, C, T, H, W] -> [B, C, T*H, W]
        b, c, t, h, w = x.size()
        x = x.view(b, c, t*h, w)
        x = self._group_norm(x)
        x = x.view(b, c, t, h, w)
        
        x = self._swish(x)
        x = self._conv_2(x)

        return x  # [B, D, T, h, w]


# 3D 解码器
class Decoder3D(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 up_samples,
                 num_groups):
        super(Decoder3D, self).__init__()

        # [B, D, T, h, w] -> [B, C'', T, h, w]
        self._conv_1 = nn.Conv3d(in_channels=out_channels,
                                 out_channels=hid_channels_2,
                                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        # [B, C'', T, h, w] -> [B, C'', T, h, w]
        self._res_1 = ResidualBlock3D(in_channels=hid_channels_2,
                                      hid_channels=hid_channels_2 // 2)
        self._non_local = NonLocalBlock3D(in_channels=hid_channels_2,
                                          hid_channels=hid_channels_2 // 2)
        self._res_2 = ResidualBlock3D(in_channels=hid_channels_2,
                                      hid_channels=hid_channels_2 // 2)
        
        # [B, C'', T, h, w] -> [B, C', T, H, W]
        self._up_samples = nn.ModuleList()
        for i in range(up_samples):
            cur_in_channels = hid_channels_2 if i == 0 else hid_channels_1
            self._up_samples.append(
                ResidualBlock3D(in_channels=cur_in_channels,
                                hid_channels=cur_in_channels // 2)
            )
            self._up_samples.append(
                UpSampleBlock3D(in_channels=cur_in_channels,
                                out_channels=hid_channels_1)
            )
        
        # [B, C', T, H, W] -> [B, C', T, H, W]
        self._group_norm = nn.GroupNorm(num_groups=num_groups,
                                        num_channels=hid_channels_1)
        self._swish = Swish()

        # [B, C', T, H, W] -> [B, C, T, H, W]
        self._conv_2 = nn.Conv3d(in_channels=hid_channels_1,
                                 out_channels=in_channels,
                                 kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
    def forward(self, x):  # [B, D, T, h, w]
        x = self._conv_1(x)
        
        x = self._res_1(x)
        x = self._non_local(x)
        x = self._res_2(x)

        for layer in self._up_samples:
            x = layer(x)

        # GroupNorm 需要特殊处理
        b, c, t, h, w = x.size()
        x = x.view(b, c, t*h, w)
        x = self._group_norm(x)
        x = x.view(b, c, t, h, w)
        
        x = self._swish(x)
        x = self._conv_2(x)

        return x  # [B, C, T, H, W]


# 3D AttentiveLISTA (保持时间维度的稀疏编码)
class AttentiveLISTA3D(nn.Module):
    def __init__(self,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device):
        super(AttentiveLISTA3D, self).__init__()

        self._num_atoms = num_atoms
        self._num_dims = num_dims
        self._device = device

        self._Dict = nn.Parameter(self.initialize_dct_weights())  # [D, K]
        self._L = nn.Parameter((torch.norm(self._Dict, p=2)) ** 2)  # scalar
        
        # 3D 卷积处理时空特征
        self._conv = nn.Conv3d(in_channels=num_dims,
                               out_channels=num_atoms,
                               kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self._res1 = ResidualBlock3D(in_channels=num_atoms,
                                     hid_channels=num_atoms//2)
        self._res2 = ResidualBlock3D(in_channels=num_atoms,
                                     hid_channels=num_atoms//2)
        self._cbam = CBAM3D(in_channels=num_atoms,
                            reduction=16,
                            kernel_size=3)

        self._Zero = torch.zeros(num_atoms).to(device)  # [K]
        self._Identity = torch.eye(num_atoms).to(device)  # [K, K]

        self._num_iters = num_iters
    
    def initialize_dct_weights(self):
        n = math.ceil(math.sqrt(self._num_dims))
        m = math.ceil(math.sqrt(self._num_atoms))
        weights = init_dct(n, m)[:, :self._num_atoms]  # [D, K]
        return weights
    
    def get_dict(self):
        return self._Dict
    
    def set_dict(self, dictionary):
        self._Dict = nn.Parameter(dictionary)
        print("Dictionary initialized with pretrained values.")

    def soft_thresh(self, x, theta):
        return torch.sign(x) * torch.max(torch.abs(x) - theta, self._Zero)

    def generation(self, input_z):  # [B, K, T, h, w]
        # 转换维度用于矩阵乘法
        b, k, t, h, w = input_z.size()
        input_z = input_z.permute(0, 2, 3, 4, 1).contiguous()  # [B, T, h, w, K]
        input_z = input_z.view(b * t * h * w, k)  # [B*T*h*w, K]
        
        x_recon = torch.matmul(input_z, self._Dict.t())  # [B*T*h*w, K] * [K, D] -> [B*T*h*w, D]
        x_recon = x_recon.view(b, t, h, w, self._num_dims).permute(0, 4, 1, 2, 3).contiguous()  # [B, D, T, h, w]
        return x_recon

    def forward(self, x):  # [B, D, T, H, W]
        B, D, T, H, W = x.size()
        
        # 3D 卷积提取时空特征
        l = self._conv(x)  # [B, D, T, H, W] -> [B, K, T, H, W]
        l = self._res1(l)
        l = self._res2(l)
        l = self._cbam(l) / self._L

        # 稀疏编码（在每个时空点独立进行）
        b, k, t, h, w = l.size()
        l = l.permute(0, 2, 3, 4, 1).contiguous()  # [B, K, T, H, W] -> [B, T, H, W, K]
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, T, H, W] -> [B, T, H, W, D]
        
        # 展平用于矩阵乘法
        l = l.view(b * t * h * w, k)  # [B*T*H*W, K]
        x = x.view(b * t * h * w, D)  # [B*T*H*W, D]
        
        S = self._Identity - (1 / self._L) * self._Dict.t().mm(self._Dict)  # [K, K]
        S = S.t()  # [K, K]

        y = torch.matmul(x, self._Dict)  # [B*T*H*W, D] * [D, K] -> [B*T*H*W, K]

        z = self.soft_thresh(y, l)  # [B*T*H*W, K]
        for iter_t in range(self._num_iters):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self._L) * y, l)

        x_recon = torch.matmul(z, self._Dict.t())  # [B*T*H*W, K] * [K, D] -> [B*T*H*W, D]

        # 恢复维度
        z = z.view(b, t, h, w, k).permute(0, 4, 1, 2, 3).contiguous()  # [B, K, T, h, w]
        x_recon = x_recon.view(b, t, h, w, D).permute(0, 4, 1, 2, 3).contiguous()  # [B, D, T, h, w]

        return z, x_recon, self._Dict


# 3D Residual 块（用于 TranslatorWithResidual3D）
class Residual3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_p=0.1):
        super(Residual3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout3d(dropout_p)
        
        # If input and output channels don't match, add a 1x1 convolution
        self.match_channels = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.match_channels:
            residual = self.match_channels(residual)

        x += residual
        x = self.relu(x)
        return x


# 3D TranslatorWithResidual
class TranslatorWithResidual3D(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7], dropout_p=0.1):
        super(TranslatorWithResidual3D, self).__init__()

        # Initial convolution block with residual
        self.initial = Residual3D(C_in, C_hid, kernel_size=3, stride=1, padding=1, dropout_p=dropout_p)

        # Branches for multi-scale features (inception-style)
        self.branches = nn.ModuleList([
            Residual3D(C_hid, C_out, kernel_size=k, stride=1, padding=k//2, dropout_p=dropout_p)
            for k in incep_ker
        ])

        # Merge layers for final output
        self.merge = nn.Sequential(
            nn.Conv3d(len(incep_ker) * C_out, C_out, kernel_size=1),
            nn.BatchNorm3d(C_out),
            nn.Dropout3d(p=0.2)
        )

    def forward(self, x):  # [B, C_in, T, h, w]
        residual = self.initial(x)  # [B, C_hid, T, h, w]
        
        branch_outputs = [branch(residual) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)  # [B, len(incep_ker)*C_out, T, h, w]
        x = self.merge(x)  # [B, C_out, T, h, w]

        return x


# ============================================================================
# 🎬 3D SSCVAE (真正的多帧时序模型)
# ============================================================================

class SSCVAE3D(nn.Module):
    def __init__(self,
                 in_channels_sate,  # 卫星图像通道数 (3)
                 in_channels_radar,  # 雷达图像通道数 (1)
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device):
        super(SSCVAE3D, self).__init__()

        self._encoder_sate = Encoder3D(in_channels=in_channels_sate,
                                       hid_channels_1=hid_channels_1,
                                       hid_channels_2=hid_channels_2,
                                       out_channels=out_channels,
                                       down_samples=down_samples,
                                       num_groups=num_groups)

        self._decoder_sate = Decoder3D(in_channels=in_channels_sate,
                                       hid_channels_1=hid_channels_1,
                                       hid_channels_2=hid_channels_2,
                                       out_channels=out_channels,
                                       up_samples=down_samples,
                                       num_groups=num_groups)

        self._encoder_radar = Encoder3D(in_channels=in_channels_radar,
                                        hid_channels_1=hid_channels_1,
                                        hid_channels_2=hid_channels_2,
                                        out_channels=out_channels,
                                        down_samples=down_samples,
                                        num_groups=num_groups)

        self._decoder_radar = Decoder3D(in_channels=in_channels_radar,
                                        hid_channels_1=hid_channels_1,
                                        hid_channels_2=hid_channels_2,
                                        out_channels=out_channels,
                                        up_samples=down_samples,
                                        num_groups=num_groups)

        self._LISTA = AttentiveLISTA3D(num_atoms=num_atoms,
                                       num_dims=num_dims,
                                       num_iters=num_iters,
                                       device=device)

        self._mlp = TranslatorWithResidual3D(C_in=128, C_hid=196, C_out=128, incep_ker=[1, 3, 5])

    def generation(self, input_z):
        ex = self._LISTA.generation(input_z)  # [B, K, T, h, w] -> [B, D, T, h, w]
        x_generation = self._decoder_radar(ex)  # [B, D, T, h, w] -> [B, C, T, H, W]
        x_generation = torch.sigmoid(x_generation)
        return ex, x_generation
    
    def get_dict(self):
        return self._LISTA.get_dict()

    def forward(self, satellite, vil):
        """
        Args:
            satellite: [B, C, H, W, T] 或 [B, C, T, H, W]
            vil: [B, C, H, W, T] 或 [B, C, T, H, W]
        Returns:
            x_recon_trans: [B, C, T, H, W]
            z: [B, K, T, h, w]
            ...
        """
        # 统一输入格式为 [B, C, T, H, W]
        if satellite.dim() == 5:
            # 判断是 [B, C, H, W, T] 还是 [B, C, T, H, W]
            if satellite.size(2) < satellite.size(4):  # H < T，说明是 [B, C, H, W, T]
                satellite = satellite.permute(0, 1, 4, 2, 3).contiguous()  # -> [B, C, T, H, W]
                vil = vil.permute(0, 1, 4, 2, 3).contiguous()
        
        # 3D 编码（时空联合）
        ex = self._encoder_sate(satellite)  # [B, C, T, H, W] -> [B, D, T, h, w]
        ex_radar = self._encoder_radar(vil)  # [B, C, T, H, W] -> [B, D, T, h, w]

        # 3D 稀疏编码
        z, ex_recon, dictionary = self._LISTA(ex)  # [B, D, T, h, w] -> [B, K, T, h, w]
        z_radar, ex_recon_radar, dictionary_radar = self._LISTA(ex_radar)

        # 3D 转换
        z_trans = self._mlp(z)  # [B, K, T, h, w] -> [B, K, T, h, w]

        # 3D 解码
        ex_trans, x_recon_trans = self.generation(z_trans)  # [B, K, T, h, w] -> [B, C, T, H, W]

        # 计算损失
        reconstruction_loss = segmented_weighted_loss(x_recon_trans, vil)
        z_diff_loss = F.mse_loss(z_trans, z_radar)
        latent_dist_loss = z_diff_loss
        latent_trans_loss = torch.sum((ex_trans - ex_radar).pow(2), dim=1).mean()

        return x_recon_trans, z, latent_dist_loss, latent_trans_loss, reconstruction_loss, dictionary


# 3D SSCVAEDouble (双解码器版本)
class SSCVAEDouble3D(nn.Module):
    def __init__(self,
                 in_channels_sate,
                 in_channels_radar,
                 hid_channels_1,
                 hid_channels_2,
                 out_channels,
                 down_samples,
                 num_groups,
                 num_atoms,
                 num_dims,
                 num_iters,
                 device):
        super(SSCVAEDouble3D, self).__init__()

        self._encoder_sate = Encoder3D(in_channels=in_channels_sate,
                                       hid_channels_1=hid_channels_1,
                                       hid_channels_2=hid_channels_2,
                                       out_channels=out_channels,
                                       down_samples=down_samples,
                                       num_groups=num_groups)

        self._decoder_sate = Decoder3D(in_channels=in_channels_sate,
                                       hid_channels_1=hid_channels_1,
                                       hid_channels_2=hid_channels_2,
                                       out_channels=out_channels,
                                       up_samples=down_samples,
                                       num_groups=num_groups)

        self._encoder_radar = Encoder3D(in_channels=in_channels_radar,
                                        hid_channels_1=hid_channels_1,
                                        hid_channels_2=hid_channels_2,
                                        out_channels=out_channels,
                                        down_samples=down_samples,
                                        num_groups=num_groups)

        self._decoder_radar = Decoder3D(in_channels=in_channels_radar,
                                        hid_channels_1=hid_channels_1,
                                        hid_channels_2=hid_channels_2,
                                        out_channels=out_channels,
                                        up_samples=down_samples,
                                        num_groups=num_groups)

        self._LISTA = AttentiveLISTA3D(num_atoms=num_atoms,
                                       num_dims=num_dims,
                                       num_iters=num_iters,
                                       device=device)

    def generation(self, input_z, is_sate=True):
        if is_sate:
            ex = self._LISTA.generation(input_z)
            x_generation = self._decoder_sate(ex)
        else:
            ex = self._LISTA.generation(input_z)
            x_generation = self._decoder_radar(ex)
        
        x_generation = torch.sigmoid(x_generation)
        return x_generation
    
    def get_dict(self):
        return self._LISTA.get_dict()

    def forward(self, satellite, vil):
        """
        Args:
            satellite: [B, C, H, W, T] 或 [B, C, T, H, W]
            vil: [B, C, H, W, T] 或 [B, C, T, H, W]
        Returns:
            x_recon_sate: [B, C, T, H, W]
            x_recon_radar: [B, C, T, H, W]
            z_sate: [B, K, T, h, w]
            z_radar: [B, K, T, h, w]
            total_latent_loss: scalar
            dictionary: [D, K]
        """
        # 统一输入格式为 [B, C, T, H, W]
        if satellite.dim() == 5:
            if satellite.size(2) < satellite.size(4):  # H < T，说明是 [B, C, H, W, T]
                satellite = satellite.permute(0, 1, 4, 2, 3).contiguous()
                vil = vil.permute(0, 1, 4, 2, 3).contiguous()
        
        # 3D 编码
        ex_sate = self._encoder_sate(satellite)  # [B, C, T, H, W] -> [B, D, T, h, w]
        ex_radar = self._encoder_radar(vil)

        # 3D 稀疏编码
        z_sate, ex_recon_sate, dictionary = self._LISTA(ex_sate)  # [B, D, T, h, w] -> [B, K, T, h, w]
        z_radar, ex_recon_radar, _ = self._LISTA(ex_radar)
        
        # 3D 解码
        x_recon_sate = torch.sigmoid(self._decoder_sate(ex_recon_sate))  # [B, K, T, h, w] -> [B, C, T, H, W]
        x_recon_radar = torch.sigmoid(self._decoder_radar(ex_recon_radar))

        # 计算损失
        latent_loss_sate = torch.sum((ex_recon_sate - ex_sate).pow(2), dim=1).mean()
        latent_loss_radar = torch.sum((ex_recon_radar - ex_radar).pow(2), dim=1).mean()
        total_latent_loss = latent_loss_sate + latent_loss_radar

        return x_recon_sate, x_recon_radar, z_sate, z_radar, total_latent_loss, dictionary
