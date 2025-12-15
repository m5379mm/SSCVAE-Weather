
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
    # # 解冻解码器的参数
    # for param in model._decoder_radar.parameters():
    #     param.requires_grad = True

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
    model = SSCVAE(**vars(model_args), device=device).to(device)

    load_path = os.path.join(train_args.origin_path, 'models', 'best_model.pt')
    load_path = "/root/autodl-tmp/results/sscvae_recon_sevir_trans/models/best_model.pth"
    print(load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint, strict=False)

    # 冻结所有层
    freeze_all_layers(model)

    # 只解冻 MLP 层
    unfreeze_mlp(model)

    # 验证：打印可训练参数
    print("\n✅ 可训练的参数（requires_grad=True）:")
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.numel()} 参数")
            trainable_params += param.numel()
    print(f"总可训练参数数量: {trainable_params:,}\n")

    # 设置优化器，仅训练 MLP 层
    optim_groups = []
    if hasattr(model, "_mlp"):
        optim_groups.append({
            'params': [p for p in model._mlp.parameters() if p.requires_grad],
            'lr': 3e-3,
        })
    else:
        raise ValueError("模型没有 _mlp 属性，无法训练！")

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
            model.load_state_dict(torch.load(early_stopping.path))
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

# def freeze_bn_for_frozen_modules(model):
#     train_these = {"_temporal_tr", "_mlp","_LISTA","_decoder_radar"}
#     for name, m in model.named_modules():
#         if not any(name.startswith(t) for t in train_these):
#             if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#                 m.eval()

# def train(data_args, model_args, train_args, test_args):
#     model_fold_path = os.path.join(train_args.save_path, 'models')
#     image_fold_path = os.path.join(train_args.save_path, 'images')
#     dict_fold_path = os.path.join(train_args.save_path, 'dicts')
#     os.makedirs(model_fold_path, exist_ok=True)
#     os.makedirs(image_fold_path, exist_ok=True)
#     os.makedirs(dict_fold_path, exist_ok=True)

#     data_transform = {"train": transforms.Compose([]), "val": transforms.Compose([])}

#     train_dataset = SevirTimeTransDataset(root_dir=data_args.root_dir, mode="train", transform=data_transform["train"])
#     val_dataset = SevirTimeTransDataset(root_dir=data_args.root_dir, mode="val", transform=data_transform["val"])
#     train_loader = DataLoader(train_dataset, batch_size=data_args.batch_size, shuffle=True, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

#     train_image_num = len(train_dataset)
#     val_image_num = len(val_dataset)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = SSCVAE(**vars(model_args), device=device).to(device)

#     load_path = os.path.join(train_args.origin_path, 'models', 'best_model.pt')
#     print(load_path)
#     checkpoint = torch.load(load_path)
#     model.load_state_dict(checkpoint, strict=False)

#     for p in model.parameters():
#         p.requires_grad = False
#     if hasattr(model, "_temporal_tr"):
#         for p in model._temporal_tr.parameters():
#             p.requires_grad = True
#     if hasattr(model, "_mlp"):
#         print("need")
#         input()
#         for p in model._mlp.parameters():
#             p.requires_grad = True
#     if hasattr(model, "_LISTA"):
#         for p in model._mlp.parameters():
#             p.requires_grad = False

#     optim_groups = []
#     if hasattr(model, "_temporal_tr"):
#         optim_groups.append({
#             'params': [p for p in model._temporal_tr.parameters() if p.requires_grad],
#             'lr': 3e-4,
#         })
#     if hasattr(model, "_mlp"):
#         optim_groups.append({
#             'params': [p for p in model._mlp.parameters() if p.requires_grad],
#             'lr': 3e-4,
#         })
#     if hasattr(model, "LISTA"):
#         optim_groups.append({
#             'params': [p for p in model._mlp.parameters() if p.requires_grad],
#             'lr': 2e-4,
#         })
#     optim_groups.append({
#         'params': [p for p in model._decoder_radar.parameters() if p.requires_grad],
#         'lr': 1e-4,
#     })

#     optimizer = torch.optim.AdamW(optim_groups, weight_decay=1e-5)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-5, verbose=True)
#     early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(model_fold_path, 'best_model.pt'))

#     csv_filename = os.path.join(train_args.save_path, 'training_losses.csv')
#     with open(csv_filename, 'w', newline='') as csvfile:
#         fieldnames = ['Epoch', 'Train Recon Loss', 'Train Latent Trans Loss', 'Train Latent Dist Loss', 
#                       'Train Total Loss', 'Train Sparsity', 'Val Recon Loss', 'Val Latent Trans Loss', 
#                       'Val Latent Dist Loss', 'Val Total Loss', 'Val Sparsity', 'Learning Rate']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#     for name, param in model.named_parameters():
#         print(name, param.requires_grad)

#     for epoch in range(train_args.epochs):
#         model.train()
#         #freeze_bn_for_frozen_modules(model)

#         train_losses = {"latent_dist": 0, "latent_trans": 0, "recon": 0, "total": 0, "sparsity": 0}
#         w = get_weights(epoch)
#         for satellite, vil in tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100):
#             satellite, vil = satellite.to(device), vil.to(device)
#             optimizer.zero_grad()
#             x_recon_trans, z, latent_dist_loss, latent_trans_loss, recon_loss, dictionary, sparsity_loss = model(satellite, vil)

#             loss = (w["recon"]  * recon_loss +
#                     w["pair"]   * latent_dist_loss +
#                     w["trans"]  * latent_trans_loss)

#             loss.backward()
#             optimizer.step()
#             sparsity_loss = hoyer_metric(z)

#             bs = satellite.size(0)
#             train_losses["latent_dist"] += latent_dist_loss.item() * bs
#             train_losses["latent_trans"] += latent_trans_loss.item() * bs
#             train_losses["recon"] += recon_loss.item() * bs
#             train_losses["total"] += loss.item() * bs
#             train_losses["sparsity"] += sparsity_loss.item() * bs

#         for key in train_losses:
#             train_losses[key] /= train_image_num

#         model.eval()
#         freeze_bn_for_frozen_modules(model)

#         val_losses = {"latent_dist": 0, "latent_trans": 0, "recon": 0, "total": 0, "sparsity": 0}
#         with torch.no_grad():
#             for satellite, vil in val_loader:
#                 satellite, vil = satellite.to(device), vil.to(device)
#                 x_recon_trans, z, latent_dist_loss, latent_trans_loss, recon_loss, _, sparsity_loss = model(satellite, vil)
#                 loss_sparse = z.abs().mean()
#                 loss_temp = F.mse_loss(z[:, 1:], z[:, :-1]) if z.dim() == 5 and z.size(1) > 1 else 0.0
#                 loss = (w["recon"]  * recon_loss +
#                         w["pair"]   * latent_dist_loss +
#                         w["trans"]  * latent_trans_loss)
#                 sparsity_loss = hoyer_metric(z)
#                 bs = satellite.size(0)
#                 val_losses["latent_dist"] += latent_dist_loss.item() * bs
#                 val_losses["latent_trans"] += latent_trans_loss.item() * bs
#                 val_losses["recon"] += recon_loss.item() * bs
#                 val_losses["total"] += loss.item() * bs
#                 val_losses["sparsity"] += sparsity_loss.item() * bs

#         for key in val_losses:
#             val_losses[key] /= val_image_num

#         with open(csv_filename, 'a', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writerow({
#                 'Epoch': epoch + 1,
#                 'Train Recon Loss': train_losses['recon'],
#                 'Train Latent Trans Loss': train_losses['latent_trans'],
#                 'Train Latent Dist Loss': train_losses['latent_dist'],
#                 'Train Total Loss': train_losses['total'],
#                 'Train Sparsity': train_losses['sparsity'],
#                 'Val Recon Loss': val_losses['recon'],
#                 'Val Latent Trans Loss': val_losses['latent_trans'],
#                 'Val Latent Dist Loss': val_losses['latent_dist'],
#                 'Val Total Loss': val_losses['total'],
#                 'Val Sparsity': val_losses['sparsity'],
#                 'Learning Rate': get_lrs(optimizer)
#             })

#         scheduler.step(val_losses['total'])
#         early_stopping(val_losses['total'], model)

#         if early_stopping.early_stop:
#             print("Early stopping triggered. Loading best model...")
#             model.load_state_dict(torch.load(early_stopping.path))
#             break

#         if (epoch + 1) % train_args.save_frequency == 0:
#             plot_dict(dictionary, dict_fold_path, f'dictionary{epoch + 1:d}.png')
#             torch.save(model.state_dict(), os.path.join(model_fold_path, f'model{epoch + 1:d}.pt'))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, required=True, help='Path to config JSON')
#     args = parser.parse_args()

#     with open(args.config, 'r') as f:
#         config = json.load(f)
#     data_args = SimpleNamespace(**config['data'])
#     model_args = SimpleNamespace(**config['model'])
#     train_args = SimpleNamespace(**config['train'])
#     test_args = SimpleNamespace(**config['test'])

#     train(data_args, model_args, train_args, test_args) 