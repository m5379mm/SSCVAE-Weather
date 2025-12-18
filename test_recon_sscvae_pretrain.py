import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import argparse
import json
from types import SimpleNamespace
import csv

from utils.utils import MNISTDataset,GrayDataset, UltrasoundDataset, MiniImagenet, hoyer_metric, compute_indicators, slice_image, recon_image
from models import SSCVAEDouble
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

model = SSCVAEDouble(in_channels_radar=model_args.in_channels_radar,
               in_channels_sate =model_args.in_channels_sate,
               hid_channels_1=model_args.hid_channels_1,
               hid_channels_2=model_args.hid_channels_2,
               out_channels=model_args.out_channels,
               down_samples=model_args.down_samples,
               num_groups=model_args.num_groups,
               num_atoms=model_args.num_atoms,
               num_dims=model_args.num_dims,
               num_iters=model_args.num_iters,
               device=device).to(device)

# load_path = os.path.join(model_fold_path, f'best_model.pt')
# load_path = "/root/autodl-tmp/results/sscvae_recon_sevir/models/best_model.pt"
load_path = "/root/autodl-tmp/results/sscvae_recon_sevir_trans/models/best_model.pt"
model.load_state_dict(torch.load(load_path), strict=False)  # strict=False 忽略多余或缺失的键
model.eval()


'''test'''
csv_filename = os.path.join(train_args.save_path, 'testing_indicators.csv')
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Name', 'Sparsity', 'PSNR', 'SSIM', 'NMI', 'LPIPS']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

plot_dict = True
with torch.no_grad():
    for batch_idx, (satellite, vil) in enumerate(test_loader):
        satellite = satellite.to(device)
        print(satellite.shape)
        bs, _, _, _,T = satellite.shape  # 输入是 [B, C, H, W, T]
        vil  = vil.to(device)
        bs, _, _, _,_ = vil.shape

        x_recon_sate,x_recon_radar, z_sate,z_radar, total_latent_loss, dictionary = model(satellite,vil)
        
        # 模型输出是 [B, T, C, H, W]，输入是 [B, C, H, W, T]
        # 不需要转换输出，而是调整索引方式

        if plot_dict:
            plot_dict_tsne(dictionary, train_args.save_path, 'sscvae_tsne.png')
            plot_dict = False

        sparsity_sate = hoyer_metric(z_sate).item()
        sparsity_radar = hoyer_metric(z_radar).item()
        sparsity = (sparsity_radar+sparsity_sate)/2
               # 初始化存储每一帧的指标
        psnr_list, ssim_list, nmi_list, lpips_list = [], [], [], []

        # 逐帧计算指标并存储
        for t in range(T):
            # 计算每一帧的 PSNR, SSIM, NMI, LPIPS
            # satellite: [B, C, H, W, T] -> [:, :, :, :, t] -> [B, C, H, W]
            # x_recon_sate: [B, T, C, H, W] -> [:, t, :, :, :] -> [B, C, H, W]
            PSNR, SSIM, NMI, LPIPS = compute_indicators(satellite[:, :, :, :, t], x_recon_sate[:, t, :, :, :])
            psnr_list.append(PSNR)
            ssim_list.append(SSIM)
            nmi_list.append(NMI)
            lpips_list.append(LPIPS)

            # 绘制每一帧的图像
            plot_images(satellite[:, :, :, :, t], x_recon_sate[:, t, :, :, :], image_fold_path, str(batch_idx)+str(t), channels=3)
            print(789789789789789)

        # 计算每个指标的平均值
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


        # 初始化存储每一帧的指标
        psnr_list, ssim_list, nmi_list, lpips_list = [], [], [], []

        # 逐帧计算指标并存储
        for t in range(T):
            print(t)
            # vil: [B, C, H, W, T] -> [:, :, :, :, t] -> [B, C, H, W]
            # x_recon_radar: [B, T, C, H, W] -> [:, t, :, :, :] -> [B, C, H, W]
            PSNR, SSIM, NMI, LPIPS = compute_indicators(vil[:, :, :, :, t], x_recon_radar[:, t, :, :, :])
            psnr_list.append(PSNR)
            ssim_list.append(SSIM)
            nmi_list.append(NMI)
            lpips_list.append(LPIPS)
            
            plot_images(vil[:, :, :, :, t], x_recon_radar[:, t, :, :, :], image_fold_path, str(batch_idx)+str(t), channels=1)

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
