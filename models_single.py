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
        print("Dictionary initialized with pretrained values.")

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

class TranslatorWithResidual1115(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7], groups=4, dropout_p=0.1):
        super(TranslatorWithResidual, self).__init__()

        # 输入特征预处理
        self.input_norm = nn.GroupNorm(groups, C_in)
        
        # Initial convolution block with residual
        self.initial = Residual(C_in, C_hid, kernel_size=3, stride=1, padding=1, dropout_p=dropout_p)

        # Branches for multi-scale features (inception-style)
        self.branches = nn.ModuleList([
            Residual(C_hid, C_out, kernel_size=k, stride=1, padding=k//2, dropout_p=dropout_p)
            for k in incep_ker
        ])

        # 多尺度特征融合 - 使用注意力加权
        self.branch_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(C_out, C_out // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_out // 4, C_out, kernel_size=1),
                nn.Sigmoid()
            ) for _ in incep_ker
        ])

        # Merge layers for final output - 改进的融合方式
        self.merge = nn.Sequential(
            nn.Conv2d(len(incep_ker) * C_out, C_out, kernel_size=1, bias=False),
            nn.GroupNorm(groups, C_out),
            nn.GELU(),
            nn.Conv2d(C_out, C_out, kernel_size=3, stride=1, padding=1, groups=groups, bias=False),
            nn.GroupNorm(groups, C_out),
            nn.Dropout2d(p=dropout_p)
        )

        # 全局残差连接（输入到输出的直接映射）
        self.skip = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
            nn.GroupNorm(groups, C_out)
        )
        
        # 可学习的残差门控权重
        self.skip_gate = nn.Parameter(torch.tensor(0.5))
        
        # 特征对齐模块（用于域转换）
        self.align = CBAM(in_channels=C_out, reduction=16, kernel_size=7)
        
        # 最终特征增强
        self.final_enhance = nn.Sequential(
            Residual(C_out, C_out, kernel_size=3, stride=1, padding=1, dropout_p=dropout_p),
            nn.GroupNorm(groups, C_out)
        )

    def forward(self, x):
        # 输入归一化
        x_norm = self.input_norm(x)
        
        # 保存原始输入用于残差连接
        x_skip = self.skip(x_norm)
        
        # 初始特征提取
        h = self.initial(x_norm)
        
        # 多尺度特征提取 + 注意力加权
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_feat = branch(h)
            # 应用通道注意力
            attn = self.branch_attention[i](branch_feat)
            branch_feat = branch_feat * attn
            branch_outputs.append(branch_feat)
        
        # 特征融合
        x = torch.cat(branch_outputs, dim=1)
        x = self.merge(x)
        
        # 全局残差连接（带可学习门控）
        x = x + self.skip_gate * x_skip
        
        # 特征对齐（CBAM注意力）
        x = self.align(x)
        
        # 最终特征增强
        x = self.final_enhance(x)

        return x
# class TranslatorWithResidual(nn.Module):
#     def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7], groups=4, dropout_p=0.1):
#         super(TranslatorWithResidual, self).__init__()

#         # Initial convolution block with residual
#         self.initial = Residual(C_in, C_hid, kernel_size=3, stride=1, padding=1, dropout_p=dropout_p)

#         # 改成（注意：要求 C_out 能被 len(incep_ker) 整除；不整除就用 ceil 并保留1x1合并）
#         per = C_out // len(incep_ker)
#         self.branches = nn.ModuleList([
#             Residual(C_hid, per, kernel_size=k, stride=1, padding=k//2, dropout_p=dropout_p)
#             for k in incep_ker
#         ])

#         # 合并层：若 per * len(incep_ker) == C_out，则 1x1 可直接去掉，只留BN+Dropout
#         self.merge = nn.Sequential(
#             nn.BatchNorm2d(per * len(incep_ker)),
#             nn.Dropout2d(p=0.2)
#         )

#     def forward(self, x):
#         residual = self.initial(x)  # 保存初步特征

#         branch_outputs = [branch(residual) for branch in self.branches]
#         x = torch.cat(branch_outputs, dim=1)
#         x = self.merge(x)
#         print(x.size())
#         input()
#         return x

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
            focal_loss_val += focal_loss(pred, target, alpha=self.focal_alpha, gamma=self.focal_gamma, threshold=self.threshold)
            focal_loss_val /= pred.shape[2]

            # 2) Segmented Weighted Loss (MSE)


            # 3) SSIM Loss (逐帧计算)
            ssim_loss_total = 0
            window = create_window(11, pred.shape[1]).to(pred.device)
            ssim_val = ssim(pred, target, window=window, val_range=1)
            ssim_loss_total += 1 - ssim_val
            ssim_loss = ssim_loss_total / pred.shape[2]

            # 4) Edge Loss (逐帧计算)
            edge_loss_total = 0
            pred_edge = sobel_edges(pred)
            target_edge = sobel_edges(target)
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
                                     device=device)

        #self._mlp = translator(C_in=128, C_hid=512, C_out=128)
        #self._mlp = TranslatorWithResidual(C_in=128, C_hid=196, C_out=128, incep_ker=[1,3,5], dropout_p=0.2)
        self._mlp = TranslatorWithResidual(C_in=128, C_hid=196, C_out=128, incep_ker=[1,3,5])
        #self._mlp = TranslatorV2(C_in=128, C_hid=128, C_hid2=64, C_out=128, dropout_p=0.1)
        #self._mlp = TranslatorTransformerV2(C_in=128, num_heads=4, depth=2, mlp_ratio=2.0, dropout_p=0.1)
        self.criterion = CombinedLoss()

    def generation(self, input_z):
        ex = self._LISTA.generation(input_z)  # [B, K, h, w] -> [B, D, h, w]
        x_generation = self._decoder_radar(ex)  # [B, D, h, w] -> [B, C, H, W]
        x_generation = torch.sigmoid(x_generation)
        return ex,x_generation
    
    def get_dict(self):
        return self._LISTA.get_dict()

    def forward(self, satellite, vil):
        ex = self._encoder_sate(satellite)  # [B, C, H, W] -> [B, D, h, w]
        ex_radar = self._encoder_radar(vil)

        z, ex_recon, dictionary = self._LISTA(ex)  # z: [B, K, h, w]
        z_radar, ex_recon_radar, dictionary_radar = self._LISTA(ex_radar)

        z_trans = self._mlp(z)  # [B, K, h, w]

        ex_trans,x_recon_trans = self.generation(z_trans)  # [B, C, H, W]

        reconstruction_loss = segmented_weighted_loss(x_recon_trans, vil)
        #reconstruction_loss, loss_dict = self.criterion(x_recon_trans, vil)

        # 保持张量 shape，不 reshape
        z_diff_loss = F.mse_loss(z_trans, z_radar)
        latent_dist_loss = z_diff_loss


        latent_trans_loss = torch.sum((ex_trans - ex_radar).pow(2), dim=1).mean()

        return x_recon_trans, z, latent_dist_loss,latent_trans_loss, reconstruction_loss, dictionary

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
                                     device=device)

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
        #print(satellite.size())
        # 训练卫星部分
        ex_sate = self._encoder_sate(satellite)  # [B, C, H, W] -> [B, D, h, w]

        z_sate, ex_recon_sate, dictionary = self._LISTA(ex_sate)  # [B, D, h, w] -> [B, K, h, w], [B, D, h, w]
        
        x_recon_sate = self._decoder_sate(ex_recon_sate)
        x_recon_sate = torch.sigmoid(x_recon_sate)

        # 计算卫星的潜在损失
        latent_loss_sate = torch.sum((ex_recon_sate - ex_sate).pow(2), dim=1).mean()

        # 训练雷达部分
        ex_radar = self._encoder_radar(vil)  # [B, C, H, W] -> [B, D, h, w]

        z_radar, ex_recon_radar, dictionary = self._LISTA(ex_radar)  # [B, D, h, w] -> [B, K, h, w], [B, D, h, w]
        
        x_recon_radar = self._decoder_radar(ex_recon_radar)
        x_recon_radar = torch.sigmoid(x_recon_radar)
        '''save_tensor_to_txt(z_sate, "z_.txt")
        save_tensor_to_txt(z_radar, "z_radar_.txt")

        input("Saved z and z_trans. Press Enter to continue...")'''

        # 计算雷达的潜在损失
        latent_loss_radar = torch.sum((ex_recon_radar - ex_radar).pow(2), dim=1).mean()

        # 将卫星和雷达的潜在损失相加（或者加权）
        total_latent_loss = latent_loss_sate + latent_loss_radar

        return x_recon_sate,x_recon_radar, z_sate,z_radar, total_latent_loss, dictionary
