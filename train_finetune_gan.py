"""
SSCVAE 全模型微调 + GAN Loss
整合判别器提升重建图像真实感
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import argparse
import json
from types import SimpleNamespace
import csv
from tqdm import tqdm

# 配置 cuDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from utils.utils import get_recon_loss, hoyer_metric
from models import SSCVAE
from discriminator import PatchGANDiscriminator, GANLoss
from utils.visualization import plot_dict
from data import SevirTimeTransDataset
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss decreased ({self.best_loss} -> {val_loss}). Saving model...")
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve for {self.counter} epochs.")
        if self.counter >= self.patience:
            self.early_stop = True


def get_lrs(optimizer):
    return [group['lr'] for group in optimizer.param_groups]


def get_weights(epoch):
    """动态调整损失权重"""
    if epoch < 30:
        lambda_gan = 0.01    # 保持一定的对抗强度
        lambda_trans = 0.2
    elif epoch < 60:
        lambda_gan = 0.03    # 逐渐增加
        lambda_trans = 0.5
    else:
        lambda_gan = 0.05    # 最终稳定
        lambda_trans = 0.7
    
    return dict(
        recon=1.0,
        trans=lambda_trans,
        gan=lambda_gan,
        sparse=1.0
    )


def train(data_args, model_args, train_args, test_args):
    model_fold_path = os.path.join(train_args.save_path, 'models')
    image_fold_path = os.path.join(train_args.save_path, 'images')
    dict_fold_path = os.path.join(train_args.save_path, 'dicts')
    os.makedirs(model_fold_path, exist_ok=True)
    os.makedirs(image_fold_path, exist_ok=True)
    os.makedirs(dict_fold_path, exist_ok=True)

    data_transform = {"train": transforms.Compose([]), "val": transforms.Compose([])}

    train_dataset = SevirTimeTransDataset(root_dir=data_args.root_dir, mode="train", transform=data_transform["train"])
    val_dataset = SevirTimeTransDataset(root_dir=data_args.root_dir, mode="val", transform=data_transform["val"])
    train_loader = DataLoader(train_dataset, batch_size=data_args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    train_image_num = len(train_dataset)
    val_image_num = len(val_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ============ 模型初始化 ============
    print("📦 初始化生成器 (SSCVAE)...")
    model = SSCVAE(**vars(model_args), device=device, use_time_attention=True).to(device)
    
    # 加载最佳预训练权重
    pretrained_paths = [
        "/root/autodl-tmp/results/sscvae_recon_sevir_gan/models/best_model.pt"         # 基础预训练
    ]
    
    loaded = False
    for path in pretrained_paths:
        if os.path.exists(path):
            print(f"📥 加载预训练权重: {path}")
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
            print("✅ 预训练权重加载成功！")
            loaded = True
            break
    
    if not loaded:
        print("⚠️  未找到预训练权重，将从头训练（不推荐）")
    
    # ============ 判别器初始化 ============
    print("\n🎭 初始化判别器 (PatchGAN)...")
    discriminator = PatchGANDiscriminator(
        in_channels=1,  # VIL 单通道
        ndf=64,
        n_layers=3
    ).to(device)
    
    print(f"   生成器参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   判别器参数: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # ============ 优化器设置 ============
    # 生成器：分层学习率
    optimizer_G = torch.optim.AdamW([
        {'params': model._encoder_sate.parameters(), 'lr': 5e-6},
        {'params': model._encoder_radar.parameters(), 'lr': 5e-6},
        {'params': model._LISTA.parameters(), 'lr': 1e-5},
        {'params': model._decoder_radar.parameters(), 'lr': 3e-5},
        {'params': model._mlp.parameters(), 'lr': 5e-5},
    ], weight_decay=1e-5)
    
    # 判别器：独立优化器
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), 
                               lr=1e-5,  # 进一步降低到 1e-5
                               weight_decay=1e-5)
    
    # ============ 损失函数 ============
    criterion_GAN = GANLoss(gan_mode='lsgan').to(device)  # LSGAN 更稳定
    
    # ============ 学习率调度器 ============
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=10, verbose=True
    )
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    early_stopping = EarlyStopping(
        patience=50, 
        verbose=True, 
        path=os.path.join(model_fold_path, 'best_model.pt')
    )

    # ============ CSV 日志 ============
    csv_filename = os.path.join(train_args.save_path, 'training_losses.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'Epoch', 'Train Recon Loss', 'Train Latent Trans Loss', 'Train Latent Dist Loss',
            'Train GAN Loss', 'Train D Loss', 'Train Total Loss', 'Train Sparsity',
            'Val Recon Loss', 'Val Latent Trans Loss', 'Val Latent Dist Loss',
            'Val GAN Loss', 'Val Total Loss', 'Val Sparsity', 
            'LR Generator', 'LR Discriminator'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print("\n🚀 开始训练 (全模型微调 + GAN)...\n")
    
    # ============ 训练循环 ============
    for epoch in range(train_args.epochs):
        model.train()
        discriminator.train()
        
        train_losses = {
            "latent_dist": 0, 
            "latent_trans": 0, 
            "recon": 0, 
            "gan": 0,
            "d_loss": 0,
            "total": 0, 
            "sparsity": 0
        }
        
        w = get_weights(epoch)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_args.epochs}", ncols=120)
        for satellite, vil in pbar:
            satellite, vil = satellite.to(device), vil.to(device)
            bs = satellite.size(0)
            
            # ==================== 更新判别器 ====================
            optimizer_D.zero_grad()
            
            # 生成假图像
            with torch.no_grad():
                x_recon_trans, z, _, _, _, _ = model(satellite, vil)
            
            # 调试：打印所有形状信息（仅第一个 batch）
            if epoch == 0 and bs == satellite.size(0):  # 第一个完整 batch
                print(f"\n{'='*60}")
                print(f"🔍 调试信息 (Epoch {epoch+1}, 判别器输入)")
                print(f"{'='*60}")
                print(f"satellite shape:     {satellite.shape}")
                print(f"vil shape:           {vil.shape}")
                print(f"x_recon_trans shape: {x_recon_trans.shape}")
                print(f"vil is 5D?           {len(vil.shape) == 5}")
                print(f"x_recon_trans is 5D? {len(x_recon_trans.shape) == 5}")
            
            # 处理多帧：展平时间维度
            if len(vil.shape) == 5:  # [B, C, H, W, T]
                B, C, H, W, T = vil.size()
                vil_flat = vil.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)  # [B*T, C, H, W]
                
                # x_recon_trans 已经是 [B, T, C, H, W] 格式，直接 reshape
                if len(x_recon_trans.shape) == 5:
                    B_r, T_r, C_r, H_r, W_r = x_recon_trans.size()  # 注意顺序！
                    x_recon_flat = x_recon_trans.contiguous().view(B_r * T_r, C_r, H_r, W_r)
                else:
                    x_recon_flat = x_recon_trans
                    
                if epoch == 0 and bs == satellite.size(0):
                    print(f"\n转换后:")
                    print(f"vil_flat shape:      {vil_flat.shape}")
                    print(f"x_recon_flat shape:  {x_recon_flat.shape}")
                    print(f"{'='*60}\n")
            else:
                vil_flat = vil
                x_recon_flat = x_recon_trans
            
            # 判别真实图像
            pred_real = discriminator(vil_flat)
            loss_D_real = criterion_GAN(pred_real, target_is_real=True)
            
            # 判别假图像
            pred_fake = discriminator(x_recon_flat.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_is_real=False)
            
            # 判别器总损失
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ==================== 更新生成器 ====================
            optimizer_G.zero_grad()
            
            # 生成图像并计算所有损失
            x_recon_trans, z, latent_dist_loss, latent_trans_loss, recon_loss, dictionary = model(satellite, vil)
            
            # GAN loss: 欺骗判别器
            if len(x_recon_trans.shape) == 5:  # x_recon_trans 是 [B, T, C, H, W]
                B, T, C, H, W = x_recon_trans.size()
                x_recon_flat = x_recon_trans.contiguous().view(B * T, C, H, W)
            else:
                x_recon_flat = x_recon_trans
            
            pred_fake = discriminator(x_recon_flat)
            loss_GAN = criterion_GAN(pred_fake, target_is_real=True)
            
            # 生成器总损失
            loss_G = (w["recon"] * recon_loss +
                     0.3 * latent_dist_loss +
                     w["trans"] * latent_trans_loss +
                     w["gan"] * loss_GAN)
            
            loss_G.backward()
            optimizer_G.step()
            
            # 计算稀疏度
            sparsity_loss = hoyer_metric(z)
            
            # 累积损失
            train_losses["latent_dist"] += latent_dist_loss.item() * bs
            train_losses["latent_trans"] += latent_trans_loss.item() * bs
            train_losses["recon"] += recon_loss.item() * bs
            train_losses["gan"] += loss_GAN.item() * bs
            train_losses["d_loss"] += loss_D.item() * bs
            train_losses["total"] += loss_G.item() * bs
            train_losses["sparsity"] += sparsity_loss.item() * bs
            
            # 更新进度条
            pbar.set_postfix({
                'G': f'{loss_G.item():.4f}',
                'D': f'{loss_D.item():.4f}',
                'GAN': f'{loss_GAN.item():.3f}',
                'Recon': f'{recon_loss.item():.4f}'
            })
        
        # 平均训练损失
        for key in train_losses:
            train_losses[key] /= train_image_num
        
        # ==================== 验证 ====================
        model.eval()
        discriminator.eval()
        
        val_losses = {
            "latent_dist": 0, 
            "latent_trans": 0, 
            "recon": 0, 
            "gan": 0,
            "total": 0, 
            "sparsity": 0
        }
        
        with torch.no_grad():
            for satellite, vil in val_loader:
                satellite, vil = satellite.to(device), vil.to(device)
                bs = satellite.size(0)
                
                x_recon_trans, z, latent_dist_loss, latent_trans_loss, recon_loss, dictionary = model(satellite, vil)
                
                # GAN loss
                if len(x_recon_trans.shape) == 5:  # x_recon_trans 是 [B, T, C, H, W]
                    B, T, C, H, W = x_recon_trans.size()
                    x_recon_flat = x_recon_trans.contiguous().view(B * T, C, H, W)
                else:
                    x_recon_flat = x_recon_trans
                
                pred_fake = discriminator(x_recon_flat)
                loss_GAN = criterion_GAN(pred_fake, target_is_real=True)
                
                loss = (w["recon"] * recon_loss +
                       0.3 * latent_dist_loss +
                       w["trans"] * latent_trans_loss +
                       w["gan"] * loss_GAN)
                
                sparsity_loss = hoyer_metric(z)
                
                val_losses["latent_dist"] += latent_dist_loss.item() * bs
                val_losses["latent_trans"] += latent_trans_loss.item() * bs
                val_losses["recon"] += recon_loss.item() * bs
                val_losses["gan"] += loss_GAN.item() * bs
                val_losses["total"] += loss.item() * bs
                val_losses["sparsity"] += sparsity_loss.item() * bs
        
        for key in val_losses:
            val_losses[key] /= val_image_num
        
        # ==================== 记录日志 ====================
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Epoch': epoch + 1,
                'Train Recon Loss': train_losses['recon'],
                'Train Latent Trans Loss': train_losses['latent_trans'],
                'Train Latent Dist Loss': train_losses['latent_dist'],
                'Train GAN Loss': train_losses['gan'],
                'Train D Loss': train_losses['d_loss'],
                'Train Total Loss': train_losses['total'],
                'Train Sparsity': train_losses['sparsity'],
                'Val Recon Loss': val_losses['recon'],
                'Val Latent Trans Loss': val_losses['latent_trans'],
                'Val Latent Dist Loss': val_losses['latent_dist'],
                'Val GAN Loss': val_losses['gan'],
                'Val Total Loss': val_losses['total'],
                'Val Sparsity': val_losses['sparsity'],
                'LR Generator': get_lrs(optimizer_G),
                'LR Discriminator': get_lrs(optimizer_D)[0]
            })
        
        # 打印摘要
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train - Total: {train_losses['total']:.5f} | Recon: {train_losses['recon']:.5f} | GAN: {train_losses['gan']:.4f} | D: {train_losses['d_loss']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.5f} | Recon: {val_losses['recon']:.5f} | GAN: {val_losses['gan']:.4f}")
        
        # 学习率调度
        scheduler_G.step(val_losses['total'])
        scheduler_D.step(train_losses['d_loss'])
        
        # Early stopping
        early_stopping(val_losses['total'], model)
        
        if early_stopping.early_stop:
            print("\n🛑 Early stopping triggered. Loading best model...")
            model.load_state_dict(torch.load(early_stopping.path), strict=False)
            break
        
        # 定期保存
        if (epoch + 1) % train_args.save_frequency == 0:
            plot_dict(dictionary, dict_fold_path, f'dictionary{epoch + 1:d}.png')
            torch.save(model.state_dict(), os.path.join(model_fold_path, f'model{epoch + 1:d}.pt'))
            torch.save(discriminator.state_dict(), os.path.join(model_fold_path, f'discriminator{epoch + 1:d}.pt'))
    
    print("\n✅ 训练完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config JSON')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    data_args = SimpleNamespace(**config['data'])
    model_args = SimpleNamespace(**config['model'])
    train_args = SimpleNamespace(**config['train'])
    test_args = SimpleNamespace(**config['test'])

    train(data_args, model_args, train_args, test_args)

