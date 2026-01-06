
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
torch.backends.cudnn.benchmark = True  # 加速训练
torch.backends.cudnn.deterministic = False

from utils.utils import get_recon_loss, hoyer_metric
from models import SSCVAEDouble, SSCVAE
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
    # 动态调权
    if epoch <0:
        lambda_pair = 0
        lambda_trans = 0
        lambda_sparse = 0
    elif epoch < 60:
        lambda_pair = 0.2
        lambda_trans = 0.2
        lambda_sparse = 1.0
    elif epoch < 90:
        lambda_pair = 0.5
        lambda_trans = 0.5
        lambda_sparse = 1.0
    else:
        lambda_pair = 0.7
        lambda_trans = 0.7
        lambda_sparse = 1.0

    return dict(
        recon=1.0,
        pair=lambda_pair,
        trans=lambda_trans,
        temp=0.08,
        sparse=lambda_sparse
    )
def freeze_all_layers(model):
    """冻结所有层的参数"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_mlp(model):
    """只解冻 MLP 模块的参数"""
    if hasattr(model, "_mlp"):
        for param in model._mlp.parameters():
            param.requires_grad = True

def unfreeze_lista_temporal(model):
    """只解冻 LISTA 的时间注意力模块"""
    if hasattr(model, "_LISTA") and hasattr(model._LISTA, "_time_attention"):
        for param in model._LISTA._time_attention.parameters():
            param.requires_grad = True
        print("  ✅ 解冻 LISTA 时间注意力模块")

def unfreeze_decoder(model):
    """解冻 Decoder 模块的参数"""
    if hasattr(model, "_decoder_radar"):
        for param in model._decoder_radar.parameters():
            param.requires_grad = True

def unfreeze_all_for_finetuning(model):
    """解冻所有层进行微调（推荐用于多帧适应）"""
    for param in model.parameters():
        param.requires_grad = True

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
    # ✅ 启用时间注意力以训练 LISTA 的 temporal 部分
    model = SSCVAE(**vars(model_args), device=device, use_time_attention=True).to(device)

    # ✅ 加载预训练权重（非常重要！）
    pretrained_path = "/root/autodl-tmp/results/sscvae_recon_sevir_trans/models/best_model.pt"
    
    # 检查是否有阶段1的训练结果（用于阶段2）
    stage1_path = "/root/autodl-tmp/results/sscvae_recon_sevir_trans_lista/models/best_model.pt"
    
    if os.path.exists(stage1_path):
        print(f"📥 发现阶段1训练结果，加载: {stage1_path}")
        pretrained_path = stage1_path
    
    if os.path.exists(pretrained_path):
        print(f"📥 加载预训练权重: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
        print("✅ 预训练权重加载成功！")
    else:
        print(f"⚠️  预训练权重不存在: {pretrained_path}")
        print("   将从头开始训练（不推荐，效果会很差）")

    # ============ 训练策略选择 ============
    # 策略1: 只训练MLP (快速，但效果受限于单帧预训练特征)
    # 策略2: 训练MLP + Decoder (中等效果，让解码适应多帧)
    # 策略3: 训练 LISTA temporal + MLP (时序建模能力) ⭐推荐用于时序微调
    # 策略4: 微调整个模型 (最佳效果，所有模块适应多帧)
    
    TRAINING_STRATEGY = "lista_temporal_mlp"  # ✅ 训练 LISTA 时间注意力 + MLP
    
    if TRAINING_STRATEGY == "mlp_only":
        print("🔧 训练策略: 只训练 MLP")
        freeze_all_layers(model)
        unfreeze_mlp(model)
        optim_groups = [{
            'params': [p for p in model._mlp.parameters() if p.requires_grad],
            'lr': 3e-3,
        }]
    
    elif TRAINING_STRATEGY == "lista_temporal_mlp":
        print("🔧 训练策略: 训练 LISTA 时间注意力 + MLP（时序微调）⭐")
        freeze_all_layers(model)
        unfreeze_lista_temporal(model)  # 解冻 LISTA 时间注意力
        unfreeze_mlp(model)              # 解冻 MLP
        
        # 收集可训练参数
        lista_temporal_params = []
        if hasattr(model, "_LISTA") and hasattr(model._LISTA, "_time_attention"):
            lista_temporal_params = [p for p in model._LISTA._time_attention.parameters() if p.requires_grad]
        
        mlp_params = [p for p in model._mlp.parameters() if p.requires_grad]
        
        # ✅ 调整学习率：降低 LISTA temporal 的学习率，提高 MLP 的学习率
        optim_groups = [
            {'params': lista_temporal_params, 'lr': 1e-4},  # LISTA temporal 降低到 1e-4
            {'params': mlp_params, 'lr': 1e-4},             # MLP 也用 1e-4，统一学习率
        ]
    
    elif TRAINING_STRATEGY == "mlp_decoder":
        print("🔧 训练策略: 训练 MLP + Decoder")
        freeze_all_layers(model)
        unfreeze_mlp(model)
        unfreeze_decoder(model)
        optim_groups = [
            {'params': [p for p in model._mlp.parameters() if p.requires_grad], 'lr': 3e-3},
            {'params': [p for p in model._decoder_radar.parameters() if p.requires_grad], 'lr': 1e-4},
        ]
    
    elif TRAINING_STRATEGY == "finetune_all":
        print("🔧 训练策略: 微调整个模型（推荐用于多帧适应）")
        unfreeze_all_for_finetuning(model)
        # 使用差异化学习率：新模块（MLP）用高学习率，预训练模块用低学习率
        optim_groups = [
            {'params': model._encoder_sate.parameters(), 'lr': 1e-5},
            {'params': model._encoder_radar.parameters(), 'lr': 1e-5},
            {'params': model._LISTA.parameters(), 'lr': 1e-5},
            {'params': model._decoder_radar.parameters(), 'lr': 5e-5},
            {'params': model._mlp.parameters(), 'lr': 1e-4},
        ]
    
    else:
        raise ValueError(f"未知的训练策略: {TRAINING_STRATEGY}")

    # 验证：打印可训练参数
    print("\n✅ 可训练的参数（requires_grad=True）:")
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.numel()} 参数")
            trainable_params += param.numel()
    print(f"总可训练参数数量: {trainable_params:,}\n")

    optimizer = torch.optim.AdamW(optim_groups, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(model_fold_path, 'best_model.pt'))

    csv_filename = os.path.join(train_args.save_path, 'training_losses.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Recon Loss', 'Train Latent Trans Loss', 'Train Latent Dist Loss', 
                      'Train Total Loss', 'Train Sparsity', 'Val Recon Loss', 'Val Latent Trans Loss', 
                      'Val Latent Dist Loss', 'Val Total Loss', 'Val Sparsity', 'Learning Rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(train_args.epochs):
        model.train()
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

        train_losses = {"latent_dist": 0, "latent_trans": 0, "recon": 0, "total": 0, "sparsity": 0}
        w = get_weights(epoch)
        for satellite, vil in tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100):
            satellite, vil = satellite.to(device), vil.to(device)
            optimizer.zero_grad()
            # SSCVAE 返回 6 个值
            x_recon_trans, z, latent_dist_loss, latent_trans_loss, recon_loss, dictionary = model(satellite, vil)

            loss = (w["recon"]  * recon_loss +
                    0.3*latent_dist_loss+
                    w["trans"]  * latent_trans_loss)

            loss.backward()
            optimizer.step()
            sparsity_loss = hoyer_metric(z)

            bs = satellite.size(0)
            train_losses["latent_dist"] += latent_dist_loss.item() * bs
            train_losses["latent_trans"] += latent_trans_loss.item() * bs
            train_losses["recon"] += recon_loss.item() * bs
            train_losses["total"] += loss.item() * bs
            train_losses["sparsity"] += sparsity_loss.item() * bs

        for key in train_losses:
            train_losses[key] /= train_image_num

        model.eval()

        val_losses = {"latent_dist": 0, "latent_trans": 0, "recon": 0, "total": 0, "sparsity": 0}
        with torch.no_grad():
            for satellite, vil in val_loader:
                satellite, vil = satellite.to(device), vil.to(device)
                # SSCVAE 返回 6 个值
                x_recon_trans, z, latent_dist_loss, latent_trans_loss, recon_loss, dictionary = model(satellite, vil)
                # print(latent_dist_loss)
                loss = (w["recon"]  * recon_loss +
                       0.3*latent_dist_loss+
                        w["trans"]  * latent_trans_loss)
                sparsity_loss = hoyer_metric(z)
                bs = satellite.size(0)
                val_losses["latent_dist"] += latent_dist_loss.item() * bs
                val_losses["latent_trans"] += latent_trans_loss.item() * bs
                val_losses["recon"] += recon_loss.item() * bs
                val_losses["total"] += loss.item() * bs
                val_losses["sparsity"] += sparsity_loss.item() * bs

        for key in val_losses:
            val_losses[key] /= val_image_num

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Epoch': epoch + 1,
                'Train Recon Loss': train_losses['recon'],
                'Train Latent Trans Loss': train_losses['latent_trans'],
                'Train Latent Dist Loss': train_losses['latent_dist'],
                'Train Total Loss': train_losses['total'],
                'Train Sparsity': train_losses['sparsity'],
                'Val Recon Loss': val_losses['recon'],
                'Val Latent Trans Loss': val_losses['latent_trans'],
                'Val Latent Dist Loss': val_losses['latent_dist'],
                'Val Total Loss': val_losses['total'],
                'Val Sparsity': val_losses['sparsity'],
                'Learning Rate': get_lrs(optimizer)
            })

        scheduler.step(val_losses['total'])
        early_stopping(val_losses['total'], model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Loading best model...")
            model.load_state_dict(torch.load(early_stopping.path), strict=False)
            break

        if (epoch + 1) % train_args.save_frequency == 0:
            plot_dict(dictionary, dict_fold_path, f'dictionary{epoch + 1:d}.png')
            torch.save(model.state_dict(), os.path.join(model_fold_path, f'model{epoch + 1:d}.pt'))

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