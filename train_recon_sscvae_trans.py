# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from torchvision import transforms

# # import os
# # import argparse
# # import json
# # from types import SimpleNamespace
# # import csv
# # from tqdm import tqdm  

# # from utils.utils import GrayDataset, UltrasoundDataset, MiniImagenet, get_recon_loss, hoyer_metric,MNISTDataset
# # from models import SSCVAEDouble,SSCVAE
# # from utils.visualization import plot_dict
# # from data import SevirTransDataset
# # import numpy as np

# # class EarlyStopping:
# #     def __init__(self, patience=10, min_delta=0, verbose=False, path='checkpoint.pth'):
# #         """
# #         Early stopping logic.
        
# #         :param patience: Number of epochs to wait for improvement before stopping.
# #         :param min_delta: Minimum change to qualify as an improvement.
# #         :param verbose: Whether to print messages when an improvement is found.
# #         :param path: Path to save the model checkpoint.
# #         """
# #         self.patience = patience
# #         self.min_delta = min_delta
# #         self.verbose = verbose
# #         self.counter = 0
# #         self.best_loss = np.inf
# #         self.early_stop = False
# #         self.path = path

# #     def __call__(self, val_loss, model):
# #         """
# #         Check if early stopping condition is met and save the model.
        
# #         :param val_loss: The current validation loss.
# #         :param model: The model to save if early stopping is triggered.
# #         """
# #         if self.best_loss - val_loss > self.min_delta:
# #             self.best_loss = val_loss
# #             self.counter = 0
# #             if self.verbose:
# #                 print(f"Validation loss decreased ({self.best_loss} -> {val_loss}). Saving model...")
# #             torch.save(model.state_dict(), self.path)
# #         else:
# #             self.counter += 1
# #             if self.verbose:
# #                 print(f"Validation loss did not improve for {self.counter} epochs.")
            
# #         if self.counter >= self.patience:
# #             self.early_stop = True


# # '''hyperparameters'''
# # parser = argparse.ArgumentParser(description='program args')
# # parser.add_argument('-c', '--config', type=str, required=True, help='config file path')
# # args = parser.parse_args()
# # with open(args.config, 'r') as json_data:
# #     config_data = json.load(json_data)
# # data_args = SimpleNamespace(**config_data['data'])
# # model_args = SimpleNamespace(**config_data['model'])
# # train_args = SimpleNamespace(**config_data['train'])
# # test_args = SimpleNamespace(**config_data['test'])


# # '''make dir'''
# # model_fold_path = os.path.join(train_args.save_path, 'models')
# # model_load_path = os.path.join(train_args.origin_path, 'models')
# # dictionary_fold_path = os.path.join(train_args.save_path, 'models')
# # image_fold_path = os.path.join(train_args.save_path, 'images')
# # dict_fold_path = os.path.join(train_args.save_path, 'dicts')

# # if not os.path.exists(train_args.save_path):
# #     os.makedirs(model_fold_path)
# #     os.makedirs(image_fold_path)
# #     os.makedirs(os.path.join(image_fold_path, 'origin'))
# #     os.makedirs(os.path.join(image_fold_path, 'recon'))
# #     os.makedirs(dict_fold_path)

# # print(data_args.dataset)
# # '''dataset'''
# # data_transform = {
# #     "train": transforms.Compose([]),
# #     "val": transforms.Compose([])
# # }

# # if data_args.dataset == 'sevir':
# #     train_dataset = SevirTransDataset(
# #         root_dir=data_args.root_dir,
# #         k_folds=data_args.k_folds,
# #         fold_index=data_args.fold_index,
# #         mode="train",
# #         transform=data_transform["train"]
# #     )

# #     val_dataset = SevirTransDataset(
# #         root_dir=data_args.root_dir,
# #         k_folds=data_args.k_folds,
# #         fold_index=data_args.fold_index,
# #         mode="val",
# #         transform=data_transform["val"]
# #     )
# # else:
# #     raise ValueError("dataset: {} isn't allowed.".format(data_args.dataset))

# # train_loader = DataLoader(train_dataset,
# #                           batch_size=data_args.batch_size,
# #                           shuffle=True,
# #                           pin_memory=True)
# # val_loader = DataLoader(val_dataset,
# #                         batch_size=data_args.batch_size,
# #                         shuffle=False,
# #                         pin_memory=True)

# # train_image_num = len(train_dataset)
# # val_image_num = len(val_dataset)
# # print("train_image_num:", train_image_num)
# # print("val_image_num:", val_image_num)


# # '''model'''
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # # 冻结模型所有层的参数
# # def freeze_all_layers(model):
# #     for param in model.parameters():
# #         param.requires_grad = False

# # model = SSCVAE(in_channels_radar=model_args.in_channels_radar,
# #                in_channels_sate =model_args.in_channels_sate,
# #                hid_channels_1=model_args.hid_channels_1,
# #                hid_channels_2=model_args.hid_channels_2,
# #                out_channels=model_args.out_channels,
# #                down_samples=model_args.down_samples,
# #                num_groups=model_args.num_groups,
# #                num_atoms=model_args.num_atoms,
# #                num_dims=model_args.num_dims,
# #                num_iters=model_args.num_iters,
# #                device=device).to(device)

# # sscvae_double = SSCVAEDouble(in_channels_radar=model_args.in_channels_radar,
# #                in_channels_sate =model_args.in_channels_sate,
# #                hid_channels_1=model_args.hid_channels_1,
# #                hid_channels_2=model_args.hid_channels_2,
# #                out_channels=model_args.out_channels,
# #                down_samples=model_args.down_samples,
# #                num_groups=model_args.num_groups,
# #                num_atoms=model_args.num_atoms,
# #                num_dims=model_args.num_dims,
# #                num_iters=model_args.num_iters,
# #                device=device).to(device)

# # # 加载预训练模型的权重
# # load_path = os.path.join(model_load_path, f'model{test_args.model_id:d}.pt')
# # #load_path="/root/autodl-tmp/results/sscvae_recon_sevir_trans_6.8错误的loss/models/best_model.pth"
# # checkpoint = torch.load(load_path)
# # pretrained_state_dict = checkpoint

# # for name, param in model.named_parameters():
# #     if name in pretrained_state_dict:
# #         param.data.copy_(pretrained_state_dict[name])
# #         if 'encoder' in name:  # 只冻结 encoder
# #             print(f"冻结 Encoder 层: {name}")
# #             param.requires_grad = False
# #         elif 'LISTA._Dict' in name:  # 可选：冻结 LISTA 字典
# #             print(f"冻结 LISTA 字典: {name}")
# #             param.requires_grad = True
# #         else:
# #             param.requires_grad = False  # 其余保持可训练
# #     else:
# #         print(f"警告: {name} 没有在预训练模型中找到。")

# # # ✅ 显式确认：解冻 translator 模块
# # print("\n✅ 解冻 translator 模块中的参数:")
# # for name, param in model.named_parameters():
# #     if "mlp" in name:
# #         param.requires_grad = True
# #         print(f" - 解冻: {name}")

# # # ✅ 检查仍在训练的层
# # print("\n仍在训练的层（requires_grad=True）:")
# # for name, param in model.named_parameters():
# #     if param.requires_grad:
# #         print(f" - {name}")

# # # 设置模型为训练模式
# # model.train()

# # # 设置优化器（只包含可训练参数）
# # optimizer = torch.optim.Adam([
# #     {'params': model._mlp.parameters(), 'lr': 1e-3},               # translator 学得快
# #     {'params': model._LISTA.parameters(), 'lr': 1e-5},    # decoder 微调，慢慢适应 z_trans
# # ])


# # # 学习率调度器
# # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
# #     optimizer,
# #     mode='min',
# #     factor=0.1,
# #     patience=10,
# #     threshold=1e-5,
# #     threshold_mode='rel',
# #     cooldown=0,
# #     verbose=True
# # )

# # # 参数统计
# # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # print('The number of parameters of model is', num_params)

# # # CSV记录初始化（不变）
# # csv_filename = os.path.join(train_args.save_path, 'training_losses.csv')
# # with open(csv_filename, 'w', newline='') as csvfile:
# #     fieldnames = ['Epoch', 'Train Recon Loss', 'Train Latent Trans Loss',  'Train Latent Dist Loss','Train Total Loss', 'Train Sparsity',
# #                            'Val Recon Loss', 'Val Latent Trans Loss', 'Val Latent Dist Loss', 'Val Total Loss', 'Val Sparsity', 'Learning Rate']
# #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# #     writer.writeheader()

    
# # def get_lr(optimizer):
# #     for param_group in optimizer.param_groups:
# #         return param_group['lr']



# # # 初始化早停机制
# # early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(model_fold_path, 'best_model.pth'))

# # for epoch in range(train_args.epochs):
# #     # 训练阶段
# #     train_latent_dist_loss_item = 0
# #     train_reconstruction_loss_item = 0
# #     train_latent_trans_loss_item = 0
# #     train_total_loss_item = 0
# #     train_sparsity_item = 0
# #     index = 0
# #     model.train() 
# #     for batch_idx, (satellite, vil) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)):
# #         satellite = satellite.to(device)
# #         bs, _, _, _ = satellite.shape
# #         vil  = vil.to(device)
# #         bs, _, _, _ = vil.shape

# #         optimizer.zero_grad()

# #         # forward
# #         x_recon_trans, z, latent_loss, latent_dist_loss,latent_trans_loss, reconstruction_loss, dictionary = model(satellite, vil)
# #         sparsity = hoyer_metric(z)


# #             # 可调节各部分 loss 的权重
# #         loss = 0.7*latent_dist_loss + reconstruction_loss + 0.8*latent_trans_loss+0.5*latent_loss

# #         loss.backward()
# #         for name, param in model._mlp.named_parameters():
# #             if param.grad is not None:
# #                 print(f"{name} grad mean: {param.grad.abs().mean().item():.6f}")



# #         # statistics
# #         train_latent_dist_loss_item += latent_dist_loss.item() * bs
# #         train_latent_trans_loss_item += latent_trans_loss.item() * bs
# #         train_reconstruction_loss_item += reconstruction_loss.item() * bs
# #         train_total_loss_item += loss.item() * bs
# #         train_sparsity_item += sparsity.item() * bs

# #         # optimize
# #         optimizer.step()

# #     # Average loss over the entire training set
# #     train_latent_dist_loss_item /= train_image_num
# #     train_latent_trans_loss_item /= train_image_num
# #     train_reconstruction_loss_item /= train_image_num
# #     train_total_loss_item /= train_image_num
# #     train_sparsity_item /= train_image_num

# #     # Validation阶段
# #     val_latent_dist_loss_item = 0
# #     val_reconstruction_loss_item = 0
# #     val_latent_trans_loss_item = 0
# #     val_recon_loss_item = 0
# #     val_total_loss_item = 0
# #     val_sparsity_item = 0

# #     model.eval()
# #     with torch.no_grad():
# #         for batch_idx, (satellite, vil) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", ncols=100)):
# #             satellite = satellite.to(device)
# #             bs, _, _, _ = satellite.shape
# #             vil  = vil.to(device)
# #             bs, _, _, _ = vil.shape

# #             # forward
# #             x_recon_trans, z, latent_loss, latent_dist_loss,latent_trans_loss, reconstruction_loss, _ = model(satellite, vil)
# #             sparsity = hoyer_metric(z)

# #             loss = 0.7*latent_dist_loss + reconstruction_loss + 0.8*latent_trans_loss+0.5*latent_loss
            
# #             # statistics
# #             val_latent_dist_loss_item += latent_dist_loss.item() * bs
# #             val_latent_trans_loss_item += latent_trans_loss.item() * bs
# #             val_reconstruction_loss_item += reconstruction_loss.item() * bs
# #             val_total_loss_item += loss.item() * bs
# #             val_sparsity_item += sparsity.item() * bs


# #     val_latent_dist_loss_item /= val_image_num
# #     val_latent_trans_loss_item /= val_image_num
# #     val_reconstruction_loss_item /= val_image_num
# #     val_total_loss_item /= val_image_num
# #     val_sparsity_item /= val_image_num

# #     # Write loss to CSV
# #     with open(csv_filename, 'a', newline='') as csvfile:
# #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# #         writer.writerow({
# #             'Epoch': epoch + 1,
# #             'Train Recon Loss': train_reconstruction_loss_item,
# #             'Train Latent Trans Loss': train_latent_trans_loss_item,
# #             'Train Latent Dist Loss': train_latent_dist_loss_item,
# #             'Train Total Loss': train_total_loss_item,
# #             'Train Sparsity': train_sparsity_item,
# #             'Val Recon Loss': val_reconstruction_loss_item,
# #             'Val Latent Trans Loss': val_latent_trans_loss_item,
# #             'Val Latent Dist Loss': val_latent_dist_loss_item,
# #             'Val Total Loss': val_total_loss_item,
# #             'Val Sparsity': val_sparsity_item,
# #             'Learning Rate': get_lr(optimizer)
# #         })
# #     scheduler.step(val_total_loss_item)

# #     # Early stopping check
# #     early_stopping(val_total_loss_item, model)  # Monitor the validation loss

# #     if early_stopping.early_stop:
# #         print("Early stopping triggered. Rolling back to best model...")
# #         model.load_state_dict(torch.load(os.path.join(model_fold_path, 'best_model.pth')))
# #         break

# #     # Save the model periodically
# #     if (epoch + 1) % train_args.save_frequency == 0:
# #         plot_dict(dictionary, dict_fold_path, f'dictionary{epoch + 1:d}.png')
# #         save_path = os.path.join(model_fold_path, f'model{epoch + 1:d}.pt')
# #         torch.save(model.state_dict(), save_path)
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms

# import os
# import argparse
# import json
# from types import SimpleNamespace
# import csv
# from tqdm import tqdm

# from utils.utils import get_recon_loss, hoyer_metric
# from models import SSCVAEDouble, SSCVAE
# from utils.visualization import plot_dict
# from data import SevirTransDataset
# import numpy as np

# class EarlyStopping:
#     def __init__(self, patience=10, min_delta=0, verbose=False, path='checkpoint.pth'):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.verbose = verbose
#         self.counter = 0
#         self.best_loss = np.inf
#         self.early_stop = False
#         self.path = path

#     def __call__(self, val_loss, model):
#         if self.best_loss - val_loss > self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             if self.verbose:
#                 print(f"Validation loss decreased ({self.best_loss} -> {val_loss}). Saving model...")
#             torch.save(model.state_dict(), self.path)
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"Validation loss did not improve for {self.counter} epochs.")
#         if self.counter >= self.patience:
#             self.early_stop = True

# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']

# def train_one_fold(fold, data_args, model_args, train_args, test_args):
#     fold_path = os.path.join(train_args.save_path, f"fold_{fold}")
#     model_fold_path = os.path.join(fold_path, 'models')
#     image_fold_path = os.path.join(fold_path, 'images')
#     dict_fold_path = os.path.join(fold_path, 'dicts')
#     os.makedirs(model_fold_path, exist_ok=True)
#     os.makedirs(image_fold_path, exist_ok=True)
#     os.makedirs(dict_fold_path, exist_ok=True)

#     data_transform = {"train": transforms.Compose([]), "val": transforms.Compose([])}

#     train_dataset = SevirTransDataset(
#         root_dir=data_args.root_dir,
#         k_folds=data_args.k_folds,
#         fold_index=fold,
#         mode="train",
#         transform=data_transform["train"]
#     )
#     val_dataset = SevirTransDataset(
#         root_dir=data_args.root_dir,
#         k_folds=data_args.k_folds,
#         fold_index=fold,
#         mode="val",
#         transform=data_transform["val"]
#     )
#     train_loader = DataLoader(train_dataset, batch_size=data_args.batch_size, shuffle=True, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=data_args.batch_size, shuffle=False, pin_memory=True)
#     train_image_num = len(train_dataset)
#     val_image_num = len(val_dataset)

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = SSCVAE(**vars(model_args), device=device).to(device)

#     load_path = os.path.join(train_args.origin_path, 'models', f'best_model.pt')
    
#     #load_path="/root/autodl-tmp/results/sscvae_recon_sevir_trans_6.8错误的loss/models/best_model.pth"
#     checkpoint = torch.load(load_path)
#     model.load_state_dict(checkpoint, strict=False)
#     for name, param in model.named_parameters():
#         if 'encoder' in name:
#             param.requires_grad = False
#         elif 'LISTA._Dict' in name:
#             param.requires_grad = False
#         elif 'mlp' in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False

#     # 使用 AdamW 优化器
#     optimizer = torch.optim.AdamW([
#         {'params': model._mlp.parameters(), 'lr': 1e-3},  # 高学习率调整
#         {'params': model._decoder_radar.parameters(), 'lr': 1e-5},  # 较低学习率调整
#     ], weight_decay=1e-5)  # 可以添加权重衰减（L2正则化）

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=0.1,
#         patience=10,
#         threshold=1e-5,
#         verbose=True
#     )

#     early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(model_fold_path, 'best_model.pth'))

#     csv_filename = os.path.join(fold_path, 'training_losses.csv')
#     with open(csv_filename, 'w', newline='') as csvfile:
#         fieldnames = ['Epoch', 'Train Recon Loss', 'Train Latent Trans Loss', 'Train Latent Dist Loss','Train Total Loss', 'Train Sparsity',
#                       'Val Recon Loss', 'Val Latent Trans Loss', 'Val Latent Dist Loss', 'Val Total Loss', 'Val Sparsity', 'Learning Rate']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#     for epoch in range(train_args.epochs):
#         model.train()
#         train_losses = {"latent_dist": 0, "latent_trans": 0, "recon": 0, "total": 0, "sparsity": 0}
#         for satellite, vil in tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch+1}", ncols=100):
#             satellite, vil = satellite.to(device), vil.to(device)
#             optimizer.zero_grad()
#             x_recon_trans, z, latent_loss, latent_dist_loss, latent_trans_loss, recon_loss, dictionary = model(satellite, vil)
#             sparsity = hoyer_metric(z)
#             loss = 0.7 * latent_dist_loss + recon_loss + latent_trans_loss + 0.5 * latent_loss
#             loss.backward()
#             optimizer.step()
#             bs = satellite.size(0)
#             train_losses["latent_dist"] += latent_dist_loss.item() * bs
#             train_losses["latent_trans"] += latent_trans_loss.item() * bs
#             train_losses["recon"] += recon_loss.item() * bs
#             train_losses["total"] += loss.item() * bs
#             train_losses["sparsity"] += sparsity.item() * bs

#         for key in train_losses:
#             train_losses[key] /= train_image_num

#         model.eval()
#         val_losses = {"latent_dist": 0, "latent_trans": 0, "recon": 0, "total": 0, "sparsity": 0}
#         with torch.no_grad():
#             for satellite, vil in val_loader:
#                 satellite, vil = satellite.to(device), vil.to(device)
#                 x_recon_trans, z, latent_loss, latent_dist_loss, latent_trans_loss, recon_loss, _ = model(satellite, vil)
#                 sparsity = hoyer_metric(z)
#                 loss = 0.7 * latent_dist_loss + recon_loss + 0.8 * latent_trans_loss + 0.5 * latent_loss
#                 bs = satellite.size(0)
#                 val_losses["latent_dist"] += latent_dist_loss.item() * bs
#                 val_losses["latent_trans"] += latent_trans_loss.item() * bs
#                 val_losses["recon"] += recon_loss.item() * bs
#                 val_losses["total"] += loss.item() * bs
#                 val_losses["sparsity"] += sparsity.item() * bs

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
#                 'Learning Rate': get_lr(optimizer)
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

#     for fold in range(data_args.k_folds):
#         print(f"\n===== Starting Fold {fold + 1}/{data_args.k_folds} =====")
#         train_one_fold(fold, data_args, model_args, train_args, test_args)
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
    print(load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint, strict=False)

    # 冻结所有层
    freeze_all_layers(model)

    # 只解冻 MLP 层
    unfreeze_mlp(model)

    # 设置优化器，仅训练 MLP 层
    optim_groups = []
    if hasattr(model, "_mlp"):
        optim_groups.append({
            'params': [p for p in model._mlp.parameters() if p.requires_grad],
            'lr': 3e-3,
        })
# # 为 _decoder_radar 层设置学习率
#     if hasattr(model, "_decoder_radar"):
#         optim_groups.append({
#             'params': [p for p in model._decoder_radar.parameters() if p.requires_grad],
#             'lr': 1e-4,  # 你可以调整解码器的学习率
#         })

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
            x_recon_trans, z, latent_dist_loss,latent_trans_loss,  recon_loss, dictionary, sparsity_loss = model(satellite, vil)

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
                x_recon_trans, z, latent_dist_loss,latent_trans_loss,  recon_loss, dictionary, sparsity_loss = model(satellite, vil)
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