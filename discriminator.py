"""
Discriminator for GAN-based Training
用于 SSCVAE 微调的判别器，提升重建质量
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN 判别器 (70x70 receptive field)
    
    优点:
    - 关注局部纹理和细节
    - 训练更稳定
    - 计算效率高
    - 适合图像重建任务
    """
    
    def __init__(self, in_channels=1, ndf=64, n_layers=3):
        """
        Args:
            in_channels: 输入通道数 (1 for VIL, 3 for satellite)
            ndf: 第一层的特征通道数
            n_layers: 下采样层数
        """
        super().__init__()
        
        layers = []
        
        # 第一层 (不用 BatchNorm)
        layers.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 中间层
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.extend([
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # 最后一层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.extend([
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                     kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 输出层
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 或 [B*T, C, H, W] (如果是多帧，需要先展平)
        
        Returns:
            output: [B, 1, H', W'] patch-wise predictions
        """
        return self.model(x)


class TemporalDiscriminator(nn.Module):
    """
    时序判别器 - 用于多帧判别
    
    可以同时判断:
    1. 单帧的真实性
    2. 时序连贯性
    """
    
    def __init__(self, in_channels=1, ndf=64, n_layers=3, num_frames=7):
        """
        Args:
            in_channels: 每帧的通道数
            ndf: 基础特征通道数
            n_layers: 空间下采样层数
            num_frames: 时间帧数
        """
        super().__init__()
        
        self.num_frames = num_frames
        
        # 空间特征提取 (per-frame)
        self.spatial_encoder = nn.ModuleList()
        
        # 第一层
        self.spatial_encoder.append(nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        # 中间层
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.spatial_encoder.append(nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        # 时序卷积 (3D Conv)
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(ndf * nf_mult, ndf * nf_mult, 
                     kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 输出层
        self.output = nn.Conv3d(ndf * nf_mult, 1, kernel_size=(3, 4, 4), stride=1, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] 多帧输入
        
        Returns:
            output: [B, 1, T', H', W'] 时空判别结果
        """
        B, T, C, H, W = x.size()
        
        # 逐帧提取空间特征
        features = []
        for t in range(T):
            feat = x[:, t]  # [B, C, H, W]
            for layer in self.spatial_encoder:
                feat = layer(feat)
            features.append(feat)
        
        # 堆叠为 [B, C', H', W', T]
        features = torch.stack(features, dim=-1)  # [B, C', H', W', T]
        features = features.permute(0, 1, 4, 2, 3)  # [B, C', T, H', W']
        
        # 时序卷积
        features = self.temporal_conv(features)  # [B, C', T, H', W']
        
        # 输出
        output = self.output(features)  # [B, 1, T', H', W']
        
        return output


class GANLoss(nn.Module):
    """
    GAN Loss 封装
    
    支持多种 GAN loss:
    - vanilla (BCE)
    - lsgan (MSE)
    - wgan
    """
    
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        Args:
            gan_mode: 'vanilla' | 'lsgan' | 'wgan'
            target_real_label: 真实样本的标签
            target_fake_label: 假样本的标签
        """
        super().__init__()
        self.gan_mode = gan_mode
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        """创建与 prediction 相同大小的标签张量"""
        if target_is_real:
            target_tensor = torch.full_like(prediction, self.target_real_label)
        else:
            target_tensor = torch.full_like(prediction, self.target_fake_label)
        return target_tensor
    
    def __call__(self, prediction, target_is_real):
        """
        计算 GAN loss
        
        Args:
            prediction: 判别器输出
            target_is_real: True for real, False for fake
        
        Returns:
            loss: GAN loss
        """
        if self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        
        return loss


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("Testing Discriminators...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试 PatchGAN
    print("\n1. Testing PatchGAN Discriminator:")
    disc = PatchGANDiscriminator(in_channels=1, ndf=64, n_layers=3).to(device)
    x = torch.randn(4, 1, 256, 256).to(device)
    out = disc(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in disc.parameters()):,}")
    
    # 测试 Temporal Discriminator
    print("\n2. Testing Temporal Discriminator:")
    temp_disc = TemporalDiscriminator(in_channels=1, ndf=64, n_layers=3, num_frames=7).to(device)
    x_temp = torch.randn(2, 7, 1, 256, 256).to(device)
    out_temp = temp_disc(x_temp)
    print(f"   Input: {x_temp.shape}")
    print(f"   Output: {out_temp.shape}")
    print(f"   Parameters: {sum(p.numel() for p in temp_disc.parameters()):,}")
    
    # 测试 GAN Loss
    print("\n3. Testing GAN Loss:")
    criterion = GANLoss(gan_mode='lsgan').to(device)
    
    fake_pred = torch.randn(4, 1, 30, 30).to(device)
    real_pred = torch.randn(4, 1, 30, 30).to(device)
    
    loss_fake = criterion(fake_pred, target_is_real=False)
    loss_real = criterion(real_pred, target_is_real=True)
    
    print(f"   Loss (fake): {loss_fake.item():.4f}")
    print(f"   Loss (real): {loss_real.item():.4f}")
    
    print("\n✅ All tests passed!")

