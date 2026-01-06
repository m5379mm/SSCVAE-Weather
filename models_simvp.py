"""
SimVP: Simpler yet Better Video Prediction
论文: https://arxiv.org/abs/2206.05099
基于官方实现简化并适配 SEVIR 数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class BasicConv2d(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias
        )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvSC(nn.Module):
    """Spatial Convolution with shortcut"""
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2

        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride, padding),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        else:
            self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding)
        
        self.norm = nn.GroupNorm(2, C_out)
        self.act = nn.SiLU(inplace=True)

        if C_in != C_out or downsampling or upsampling:
            if upsampling:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(C_in, C_out, 1, 1, 0),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
            else:
                self.shortcut = nn.Conv2d(C_in, C_out, 1, stride, 0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.act(self.norm(self.conv(x)))
        return y + self.shortcut(x)


class GroupConv2d(nn.Module):
    """Group Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, act_norm=True):
        super().__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride, padding, groups=groups
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """Grouped Inception for Spatial-Temporal modeling"""
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        
        layers = []
        for ker in incep_ker:
            layers.append(
                GroupConv2d(
                    C_hid, C_out, kernel_size=ker, stride=1,
                    padding=ker//2, groups=groups, act_norm=True
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


class AttentionModule(nn.Module):
    """Spatial-Temporal Attention"""
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shorcut
        return x


class GASubBlock(nn.Module):
    """Grouped Attention SubBlock"""
    def __init__(self, dim, kernel_size=21, mlp_ratio=4., 
                 drop=0., drop_path=0.1, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, 1),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):
    """SimVP Encoder (Spatial)"""
    def __init__(self, C_in, C_hid, N_S):
        super().__init__()
        strides = [1, 2, 2, 2]
        
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, kernel_size=3, downsampling=False),
            *[ConvSC(C_hid, C_hid, kernel_size=3, downsampling=(i < len(strides) and strides[i] == 2)) 
              for i in range(N_S)]
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        
        z = self.enc(x)
        
        _, C_, H_, W_ = z.shape
        z = z.reshape(B, T, C_, H_, W_)
        return z


class Decoder(nn.Module):
    """SimVP Decoder (Spatial)"""
    def __init__(self, C_hid, C_out, N_S):
        super().__init__()
        strides = [2, 2, 2, 1]
        
        layers = []
        for i in range(N_S):
            layers.append(
                ConvSC(C_hid, C_hid, kernel_size=3, upsampling=(strides[i] == 2))
            )
        layers.append(ConvSC(C_hid, C_out, kernel_size=3, upsampling=False))
        
        self.dec = nn.Sequential(*layers)

    def forward(self, z):  # z: [B, T, C, H, W]
        B, T, C, H, W = z.shape
        z = z.reshape(B * T, C, H, W)
        
        y = self.dec(z)
        
        _, C_, H_, W_ = y.shape
        y = y.reshape(B, T, C_, H_, W_)
        return y


class MidMetaNet(nn.Module):
    """Middle MetaNet for temporal modeling"""
    def __init__(self, in_channels, out_channels, input_length, output_length,
                 hid_S=64, hid_T=512, N_T=8, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        
        # Temporal reduction
        self.reduce = nn.Conv2d(in_channels * input_length, hid_T, 1, 1, 0)
        
        # Temporal transformer blocks
        self.blocks = nn.ModuleList([
            GASubBlock(hid_T, kernel_size=21, mlp_ratio=mlp_ratio, 
                      drop=drop, drop_path=drop_path)
            for _ in range(N_T)
        ])
        
        # Temporal expansion
        self.expand = nn.Conv2d(hid_T, out_channels * output_length, 1, 1, 0)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # Reshape to [B, T*C, H, W]
        x = x.reshape(B, T * C, H, W)
        
        # Reduce
        x = self.reduce(x)  # [B, hid_T, H, W]
        
        # Transform
        for blk in self.blocks:
            x = blk(x)
        
        # Expand
        x = self.expand(x)  # [B, T_out*C, H, W]
        
        # Reshape to [B, T_out, C, H, W]
        x = x.reshape(B, self.output_length, -1, H, W)
        
        return x


class SimVP_Model(nn.Module):
    """
    SimVP Model for Video Prediction
    
    Args:
        in_shape: (T, C, H, W) - 输入形状
        hid_S: Spatial hidden dimension
        hid_T: Temporal hidden dimension
        N_S: Number of spatial blocks
        N_T: Number of temporal blocks
        output_frames: 输出帧数，默认与输入相同
    """
    def __init__(self, in_shape, hid_S=64, hid_T=512, N_S=4, N_T=8, 
                 mlp_ratio=4., drop=0., drop_path=0.1, output_frames=None):
        super().__init__()
        T, C, H, W = in_shape
        self.input_frames = T
        self.output_frames = output_frames if output_frames is not None else T
        
        self.enc = Encoder(C, hid_S, N_S)
        
        # Calculate spatial dimensions after encoding
        # Assuming 3 downsamplings (strides=[1,2,2,2])
        enc_shape = (hid_S, H // 4, W // 4)
        
        self.hid = MidMetaNet(
            hid_S, hid_S, 
            input_length=self.input_frames,
            output_length=self.output_frames,
            hid_S=hid_S, hid_T=hid_T, N_T=N_T,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path
        )
        
        self.dec = Decoder(hid_S, C, N_S)

    def forward(self, x, **kwargs):
        """
        x: [B, T, C, H, W]
        返回: [B, T_out, C, H, W]
        """
        B, T, C, H, W = x.shape
        assert T == self.input_frames, f"Expected {self.input_frames} frames, got {T}"
        
        # Encode
        z = self.enc(x)  # [B, T, C', H', W']
        
        # Temporal modeling
        z = self.hid(z)  # [B, T_out, C', H', W']
        
        # Decode
        y = self.dec(z)  # [B, T_out, C, H, W]
        
        return y


class SimVP_Predictor(nn.Module):
    """
    SimVP Predictor Wrapper for SEVIR dataset
    输入7帧历史，预测6帧未来
    """
    def __init__(self, input_channels=1, img_size=128, 
                 input_frames=7, output_frames=6,
                 hid_S=64, hid_T=256, N_S=4, N_T=8):
        super().__init__()
        
        self.model = SimVP_Model(
            in_shape=(input_frames, input_channels, img_size, img_size),
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            mlp_ratio=8.0,
            drop=0.0,
            drop_path=0.1,
            output_frames=output_frames
        )
        
        self.input_frames = input_frames
        self.output_frames = output_frames

    def forward(self, x):
        """
        x: [B, T=7, C, H, W]
        返回: [B, T=6, C, H, W]
        """
        return self.model(x)


# ======================= 辅助函数 =======================

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_simvp():
    """测试 SimVP 模型"""
    print("=" * 60)
    print("SimVP Model Test")
    print("=" * 60)
    
    # 配置
    batch_size = 2
    input_frames = 7
    output_frames = 6
    channels = 1
    img_size = 128
    
    # 创建模型
    model = SimVP_Predictor(
        input_channels=channels,
        img_size=img_size,
        input_frames=input_frames,
        output_frames=output_frames,
        hid_S=64,
        hid_T=256,
        N_S=4,
        N_T=8
    )
    
    # 测试输入
    x = torch.randn(batch_size, input_frames, channels, img_size, img_size)
    
    print(f"\n输入形状: {x.shape}")
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 前向传播
    with torch.no_grad():
        y = model(x)
    
    print(f"输出形状: {y.shape}")
    
    # 验证
    assert y.shape == (batch_size, output_frames, channels, img_size, img_size), \
        f"Expected shape {(batch_size, output_frames, channels, img_size, img_size)}, got {y.shape}"
    
    print("\n✅ 测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_simvp()


