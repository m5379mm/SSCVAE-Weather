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
from models import SSCVAE
from visualization import plot_images, plot_dict_tsne
from data import SevirTransDataset

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
if data_args.dataset == 'mnist':
    data_transform = {
        "test": transforms.Compose([
            transforms.Resize((32, 32)),  # 调整图像到 32x32
            transforms.ToTensor()
        ])
    }

if data_args.dataset == 'gray':
    test_dataset = GrayDataset(root_dir=data_args.root_dir,
                               seed=data_args.seed,
                               train_ratio=data_args.train_ratio,
                               val_ratio=data_args.val_ratio,
                               mode="test",
                               patch_size=data_args.patch_size,
                               stride_size=data_args.stride_size,
                               transform=data_transform["test"])
elif data_args.dataset == 'ultrasound':
    test_dataset = UltrasoundDataset(root_dir=data_args.root_dir,
                                     quality=data_args.quality,
                                     seed=data_args.seed,
                                     train_ratio=data_args.train_ratio,
                                     val_ratio=data_args.val_ratio,
                                     mode="test",
                                     transform=data_transform["test"])
elif data_args.dataset == 'imagenet':
    test_dataset = MiniImagenet(root_dir=data_args.root_dir,
                                mode='test',
                                patch_size=data_args.patch_size,
                                stride_size=data_args.stride_size,
                                transform=data_transform["test"])
elif data_args.dataset == 'sevir':
    test_dataset = SevirTransDataset(root_dir=data_args.root_dir,
                                 mode="test",
                                 transform=data_transform["test"])

elif data_args.dataset == 'mnist':
    test_dataset = MNISTDataset(root_dir=data_args.root_dir,
                                 mode="test",
                                 transform=data_transform["test"])
else:
    raise ValueError("dataset: {} isn't allowed.".format(data_args.dataset))

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=True)

test_image_num = len(test_dataset)
print("test_image_num:", test_image_num)


'''model'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SSCVAE(in_channels_radar=model_args.in_channels_radar,
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

load_path = os.path.join(model_fold_path, f'model{test_args.model_id:d}.pt')
model.load_state_dict(torch.load(load_path))
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
        bs, _, _, _ = satellite.shape
        vil  = vil.to(device)
        bs, _, _, _ = vil.shape

        x_recon_trans, z, latent_loss,latent_dist_loss,reconstruction_loss, dictionary = model(satellite,vil)

        if plot_dict:
            plot_dict_tsne(dictionary, train_args.save_path, 'sscvae_tsne.png')
            plot_dict = False

        sparsity = hoyer_metric(z).item()
        PSNR, SSIM, NMI, LPIPS = compute_indicators(vil, x_recon_trans)

        # if (index + 1) % 100 == 0:
        plot_images(vil,
                    x_recon_trans,
                    satellite,
                    image_fold_path,
                    str(batch_idx),
                    channels=3)

        # write result
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Name': str(batch_idx),
                             'Sparsity': sparsity,
                             'PSNR': PSNR,
                             'SSIM': SSIM,
                             'NMI': NMI,
                             'LPIPS': LPIPS})
