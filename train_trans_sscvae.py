import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import argparse
import json
from types import SimpleNamespace
import csv

from data import SevirTransDataset
from utils.utils get_recon_loss, hoyer_metric, get_noise
from models import SSCVAE


'''hyperparameters'''
parser = argparse.ArgumentParser(description='program args')
parser.add_argument('-c', '--config', type=str, required=True, help='config file path')
args = parser.parse_args()
with open(args.config, 'r') as json_data:
    config_data = json.load(json_data)
data_args = SimpleNamespace(**config_data['data'])
model_args = SimpleNamespace(**config_data['model'])
train_args = SimpleNamespace(**config_data['train'])


'''make dir'''
model_fold_path = os.path.join(train_args.save_path, 'models')
image_fold_path = os.path.join(train_args.save_path, 'images')
dict_fold_path = os.path.join(train_args.save_path, 'dicts')

if not os.path.exists(train_args.save_path):
    os.makedirs(model_fold_path)
    os.makedirs(image_fold_path)
    os.makedirs(os.path.join(image_fold_path, 'origin'))
    os.makedirs(os.path.join(image_fold_path, 'recon'))
    os.makedirs(dict_fold_path)


'''dataset'''
data_transform = {
    "train": transforms.Compose([transforms.ToTensor()]),
    "val": transforms.Compose([transforms.ToTensor()])
}
if data_args.dataset == 'imagenet':
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(256)]),
        "val": transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(256)])
    }

if data_args.dataset == 'gray':
    train_dataset = GrayDataset(root_dir=data_args.root_dir,
                                seed=data_args.seed,
                                train_ratio=data_args.train_ratio,
                                val_ratio=data_args.val_ratio,
                                mode="train",
                                patch_size=data_args.patch_size,
                                stride_size=data_args.stride_size,
                                transform=data_transform["train"])
    val_dataset = GrayDataset(root_dir=data_args.root_dir,
                              seed=data_args.seed,
                              train_ratio=data_args.train_ratio,
                              val_ratio=data_args.val_ratio,
                              mode="val",
                              patch_size=data_args.patch_size,
                              stride_size=data_args.stride_size,
                              transform=data_transform["val"])
elif data_args.dataset == 'ultrasound':
    train_dataset = UltrasoundDataset(root_dir=data_args.root_dir,
                                      quality=data_args.quality,
                                      seed=data_args.seed,
                                      train_ratio=data_args.train_ratio,
                                      val_ratio=data_args.val_ratio,
                                      mode="train",
                                      transform=data_transform["train"])
    val_dataset = UltrasoundDataset(root_dir=data_args.root_dir,
                                    quality=data_args.quality,
                                    seed=data_args.seed,
                                    train_ratio=data_args.train_ratio,
                                    val_ratio=data_args.val_ratio,
                                    mode="val",
                                    transform=data_transform["val"])
elif data_args.dataset == 'imagenet':
    train_dataset = MiniImagenet(root_dir=data_args.root_dir,
                                 mode="train",
                                 patch_size=data_args.patch_size,
                                 stride_size=data_args.stride_size,
                                 transform=data_transform["train"])
    val_dataset = MiniImagenet(root_dir=data_args.root_dir,
                               mode="val",
                               patch_size=data_args.patch_size,
                               stride_size=data_args.stride_size,
                               transform=data_transform["val"])
elif data_args.dataset == 'sevir':
    train_dataset = SevirTransDataset(root_dir=data_args.root_dir,
                                 mode="train",
                                 patch_size=data_args.patch_size,
                                 stride_size=data_args.stride_size,
                                 transform=data_transform["train"])
    val_dataset = SevirTransDataset(root_dir=data_args.root_dir,
                               mode="val",
                               patch_size=data_args.patch_size,
                               stride_size=data_args.stride_size,
                               transform=data_transform["val"])

else:
    raise ValueError("dataset: {} isn't allowed.".format(data_args.dataset))

train_loader = DataLoader(train_dataset,
                          batch_size=data_args.batch_size,
                          shuffle=True,
                          pin_memory=True)
val_loader = DataLoader(val_dataset,
                        batch_size=data_args.batch_size,
                        shuffle=False,
                        pin_memory=True)

train_image_num = len(train_dataset)
val_image_num = len(val_dataset)
print("train_image_num:", train_image_num)
print("val_image_num:", val_image_num)


'''model'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SSCVAE(in_channels=model_args.in_channels,
               hid_channels_1=model_args.hid_channels_1,
               hid_channels_2=model_args.hid_channels_2,
               out_channels=model_args.out_channels,
               down_samples=model_args.down_samples,
               num_groups=model_args.num_groups,
               num_atoms=model_args.num_atoms,
               num_dims=model_args.num_dims,
               num_iters=model_args.num_iters,
               device=device).to(device)

model.train()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)
optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)


'''train'''
csv_filename = os.path.join(train_args.save_path, 'training_losses.csv')
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Epoch', 'Train Recon Loss', 'Train Latent Loss', 'Train Total Loss', 'Train Sparsity',
                           'Val Recon Loss', 'Val Latent Loss', 'Val Total Loss', 'Val Sparsity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for epoch in range(train_args.epochs):
    # train
    train_recon_loss_item = 0
    train_latent_loss_item = 0
    train_total_loss_item = 0
    train_sparsity_item = 0

    for image_origin, _ in train_loader:
        # get the inputs
        image_origin = image_origin.to(device)
        bs, _, _, _ = image_origin.shape

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        image_recon, z, latent_loss,latent_dist_loss,reconstruction_loss, _ = model(image_noise)
        #recon_loss = get_recon_loss(image_origin, image_recon)
        sparsity = hoyer_metric(z)

        # backward
        loss = latent_loss+latent_dist_loss+reconstruction_loss
        loss.backward()

        # statistics
        train_recon_loss_item += reconstruction_loss.item() * bs
        train_latent_loss_item += latent_loss.item() * bs
        train_latent_dist_loss_item += latent_dist_loss.item()*bs
        train_total_loss_item += loss.item() * bs
        train_sparsity_item += sparsity.item() * bs

        # optimize
        optimizer.step()

    train_recon_loss_item /= train_image_num
    train_latent_loss_item /= train_image_num
    train_latent_dist_loss_item /= train_latent_dist_loss
    train_total_loss_item /= train_image_num
    train_sparsity_item /= train_image_num

    # validation
    val_recon_loss_item = 0
    val_latent_loss_item = 0
    val_total_loss_item = 0
    val_sparsity_item = 0

    with torch.no_grad():
        for image_origin, _ in val_loader:
            # get the inputs
            image_origin = image_origin.to(device)
            bs, _, _, _ = image_origin.shape

            # forward
            image_recon, z, latent_loss,latent_dist_loss,reconstruction_loss,dictionary = model(image_noise)
            #recon_loss = get_recon_loss(image_origin, image_recon)
            sparsity = hoyer_metric(z)

            loss = recon_loss + latent_loss

            # statistics
            val_recon_loss_item += reconstruction_loss.item() * bs
            val_latent_loss_item += latent_loss.item() * bs
            val_latent_dist_loss_item += latent_dist_loss.item()*bs
            val_total_loss_item += loss.item() * bs
            val_sparsity_item += sparsity.item() * bs

    val_recon_loss_item /= val_image_num
    val_latent_loss_item /= val_image_num
    val_latent_dist_loss_item /= val_image_num
    val_total_loss_item /= val_image_num
    val_sparsity_item /= val_image_num

    # write loss
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Epoch': epoch + 1,
                         'Train Recon Loss': train_recon_loss_item,
                         'Train Latent Loss': train_latent_loss_item,
                         'Train Total Loss': train_total_loss_item,
                         'Train Sparsity': train_sparsity_item,
                         'Val Recon Loss': val_recon_loss_item,
                         'Val Latent Loss': val_latent_loss_item,
                         'Val Total Loss': val_total_loss_item,
                         'Val Sparsity': val_sparsity_item})

    # save model
    if (epoch + 1) % train_args.save_frequency == 0:
        save_path = os.path.join(model_fold_path, f'model{epoch + 1:d}.pt')
        torch.save(model.state_dict(), save_path)
