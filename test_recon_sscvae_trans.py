import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import argparse
import json
from types import SimpleNamespace
import csv
import numpy as np
from utils.utils import MNISTDataset,GrayDataset, UltrasoundDataset, MiniImagenet, hoyer_metric, compute_indicators, slice_image, recon_image
from models import SSCVAE
from visualization import plot_images, plot_dict_tsne
from data import SevirTimeTransDataset

'''hyperparameters'''
parser = argparse.ArgumentParser(description='program args')
parser.add_argument('-c', '--config', type=str, required=True, help='config file path')
args = parser.parse_args()
with open(args.config, 'r') as json_data:
    config_data = json.load(json_data)
data_args = SimpleNamespace(**config_data['data'])
model_args = SimpleNamespace(**config_data['model'])
train_args = SimpleNamespace(**config_data['train'])
test_args = SimpleNamespace(**config_data['test'])


'''make dir'''
model_fold_path = os.path.join(train_args.save_path, 'models')
image_fold_path = os.path.join(train_args.save_path, 'images')


'''dataset'''
data_transform = {
    "test": transforms.Compose([])
}

if data_args.dataset == 'sevir':
    test_dataset = SevirTimeTransDataset(root_dir=data_args.root_dir,
                                 mode="test",
                                 transform=data_transform["test"])
else:
    raise ValueError("dataset: {} isn't allowed.".format(data_args.dataset))

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=True)

test_image_num = len(test_dataset)
#print("test_image_num:", test_image_num)


'''model'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ✅ 启用时间注意力以支持 LISTA temporal 微调后的模型
model = SSCVAE(**vars(model_args), device=device, use_time_attention=True).to(device)

# 使用配置文件中的路径，而不是硬编码
load_path = os.path.join(model_fold_path, f'best_model.pt')
print(f"📂 加载模型权重: {load_path}")

if not os.path.exists(load_path):
    raise FileNotFoundError(f"模型文件不存在: {load_path}")

model.load_state_dict(torch.load(load_path, map_location=device), strict=False)  # strict=False 忽略多余或缺失的键
model.eval()
print("✅ 模型加载完成，进入评估模式")


'''test'''
csv_filename = os.path.join(train_args.save_path, 'testing_indicators.csv')
print(f"📊 测试结果将保存到: {csv_filename}")

with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Name', 'Sparsity', 'PSNR', 'SSIM', 'NMI', 'LPIPS']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

plot_dict = True
import numpy as np

# 定义保存路径（使用配置文件中的路径）
PRED_PATH = os.path.join(train_args.save_path, "images", "reconstructed_images_single")
TRUE_PATH = os.path.join(train_args.save_path, "images", "true_images_single")

# 创建保存路径的文件夹（如果没有的话）
os.makedirs(PRED_PATH, exist_ok=True)
os.makedirs(TRUE_PATH, exist_ok=True)
print(f"📁 重建图像保存路径: {PRED_PATH}")
print(f"📁 真实图像保存路径: {TRUE_PATH}")

# # 用np.save替换plot_images，保存为.npy文件
# with torch.no_grad():
#     for batch_idx, (satellite, vil) in enumerate(test_loader):
#         satellite = satellite.to(device)
#         bs, _, _, _, T = satellite.shape
#         vil = vil.to(device)
#         bs, _, _, _, _ = vil.shape

#         x_recon_trans_bchwt, ex_trans_seq, latent_dist_loss, latent_trans_loss, reconstruction_loss, dictionary, sparsity_loss = model(satellite, vil)

#         if plot_dict:
#             plot_dict_tsne(dictionary, train_args.save_path, 'sscvae_tsne.png')
#             plot_dict = False

#         # 初始化存储每一帧的指标
#         psnr_list, ssim_list, nmi_list, lpips_list = [], [], [], []
#         sparsity = 1

#         # 逐帧计算指标并存储
#         for t in range(T):
#             # vil: [B, C, H, W, T], x_recon_trans_bchwt: [B, T, C, H, W]
#             PSNR, SSIM, NMI, LPIPS = compute_indicators(vil[:, :, :, :, t], x_recon_trans_bchwt[:, t, :, :, :])
#             psnr_list.append(PSNR)
#             ssim_list.append(SSIM)
#             nmi_list.append(NMI)
#             lpips_list.append(LPIPS)

#             # 保存图像为.npy文件
#             # 保存原始图像 (vil) 到 TRUE_PATH
#             np.save(os.path.join(TRUE_PATH, f"{batch_idx}_{t}_vil.npy"), vil[:, :, :, :, t].cpu().numpy())
#             # 保存重建图像 (x_recon_trans_bchwt) 到 PRED_PATH
#             np.save(os.path.join(PRED_PATH, f"{batch_idx}_{t}_recon.npy"), x_recon_trans_bchwt[:, t, :, :, :].cpu().numpy())
#             print(batch_idx, t)

#         # 计算平均指标
#         avg_psnr = torch.mean(torch.tensor(psnr_list))
#         avg_ssim = torch.mean(torch.tensor(ssim_list))
#         avg_nmi = torch.mean(torch.tensor(nmi_list))
#         avg_lpips = torch.mean(torch.tensor(lpips_list))

#         # 写入结果
#         with open(csv_filename, 'a', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writerow({'Name': str(batch_idx),
#                              'Sparsity': sparsity,
#                              'PSNR': avg_psnr.item(),
#                              'SSIM': avg_ssim.item(),
#                              'NMI': avg_nmi.item(),
#                              'LPIPS': avg_lpips.item()})

with torch.no_grad():
    print(f"\n🚀 开始测试，共 {test_image_num} 个样本...")
    for batch_idx, (satellite, vil) in enumerate(test_loader):
        satellite = satellite.to(device)
        bs, _, _, _,T = satellite.shape
        vil  = vil.to(device)
        bs, _, _, _,_ = vil.shape

        x_recon_trans, z, latent_dist_loss, latent_trans_loss, reconstruction_loss, dictionary = model(satellite,vil)

        if plot_dict:
            plot_dict_tsne(dictionary, train_args.save_path, 'sscvae_tsne.png')
            plot_dict = False

        # 初始化存储每一帧的指标
        psnr_list, ssim_list, nmi_list, lpips_list = [], [], [], []
        sparsity = 1

        # 逐帧计算指标并存储
        for t in range(T):
            # vil: [B, C, H, W, T] -> [:, :, :, :, t] -> [B, C, H, W]
            # x_recon_trans_bchwt: [B, T, C, H, W] -> [:, t, :, :, :] -> [B, C, H, W]
            PSNR, SSIM, NMI, LPIPS = compute_indicators(vil[:, :, :, :, t], x_recon_trans[:, t, :, :, :])
            psnr_list.append(PSNR)
            ssim_list.append(SSIM)
            nmi_list.append(NMI)
            lpips_list.append(LPIPS)
            
            # 保存图像为.npy文件
            # 保存原始图像 (vil) 和重建图像 (x_recon_trans_bchwt) 为.npy文件
            np.save(os.path.join(PRED_PATH, f"{batch_idx}_{t}_vil.npy"), vil[:, :, :, :, t].cpu().numpy())
            np.save(os.path.join(TRUE_PATH, f"{batch_idx}_{t}_recon.npy"), x_recon_trans[:, t, :, :, :].cpu().numpy())
            # plot_images(vil[:, :, :, :, t], x_recon_trans[:, t, :, :, :], image_fold_path, str(batch_idx)+'+'+str(t), channels=1)

        # 计算平均指标
        avg_psnr = torch.mean(torch.tensor(psnr_list))
        avg_ssim = torch.mean(torch.tensor(ssim_list))
        avg_nmi = torch.mean(torch.tensor(nmi_list))
        avg_lpips = torch.mean(torch.tensor(lpips_list))

        # 写入结果
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Name': str(batch_idx),
                             'Sparsity': sparsity,
                             'PSNR': avg_psnr.item(),
                             'SSIM': avg_ssim.item(),
                             'NMI': avg_nmi.item(),
                             'LPIPS': avg_lpips.item()})
        
        # 每处理10个样本打印一次进度
        if (batch_idx + 1) % 10 == 0:
            print(f"  进度: {batch_idx + 1}/{test_image_num} | PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

print(f"\n✅ 测试完成！结果已保存到: {csv_filename}")