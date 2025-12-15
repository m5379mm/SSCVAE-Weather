import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from visualization import plot_images, plot_dict_tsne
import os
import argparse
import json
from types import SimpleNamespace
import csv
from tqdm import tqdm  

from utils.utils import GrayDataset, UltrasoundDataset, MiniImagenet, get_recon_loss, hoyer_metric,MNISTDataset
from models import SSCVAEDouble
from utils.visualization import plot_dict
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


'''make dir'''
model_fold_path = os.path.join(train_args.save_path, 'models')
dictionary_fold_path = os.path.join(train_args.save_path, 'models')
image_fold_path = os.path.join(train_args.save_path, 'images')
dict_fold_path = os.path.join(train_args.save_path, 'dicts')

if not os.path.exists(train_args.save_path):
    os.makedirs(model_fold_path)
    #os.makedirs(dictionary_fold_path)
    os.makedirs(image_fold_path)
    os.makedirs(os.path.join(image_fold_path, 'origin'))
    os.makedirs(os.path.join(image_fold_path, 'recon'))
    os.makedirs(dict_fold_path)

print(data_args.dataset)
'''dataset'''
data_transform = {
    "train": transforms.Compose([]),
    "val": transforms.Compose([])
}

if data_args.dataset == 'sevir':
    train_dataset = SevirTimeTransDataset(root_dir=data_args.root_dir,
                                 mode="train",
                                 transform=data_transform["train"])
    val_dataset = SevirTimeTransDataset(root_dir=data_args.root_dir,
                               mode="val",
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

model.train()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)
optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

'''train'''
csv_filename = os.path.join(train_args.save_path, 'training_losses.csv')
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Epoch', 'Train Recon Loss', 'Train Latent Loss','Train Total Loss', 'Train Sparsity',
                           'Val Recon Loss', 'Val Latent Loss', 'Val Total Loss', 'Val Sparsity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

patience = 100  # 连续多少个epoch没提升后停止训练，按需调整
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(train_args.epochs):
    # 训练阶段
    train_recon_loss_item = 0
    train_latent_loss_item = 0
    train_total_loss_item = 0
    train_sparsity_item = 0

    for batch_idx, (satellite, vil) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)):
        satellite = satellite.to(device)
        bs, _, _, _,_ = satellite.shape
        vil = vil.to(device)
        bs, _, _, _,_ = vil.shape

        optimizer.zero_grad()
        x_recon_sate, x_recon_radar, z_sate, z_radar, total_latent_loss, dictionary = model(satellite, vil)

        recon_loss_sate = get_recon_loss(satellite, x_recon_sate)
        recon_loss_radar = get_recon_loss(vil, x_recon_radar)
        recon_loss = recon_loss_sate + recon_loss_radar

        sparsity = (hoyer_metric(z_sate) + hoyer_metric(z_radar)) / 2

        loss = recon_loss + total_latent_loss
        loss.backward()

        train_recon_loss_item += recon_loss.item() * bs
        train_latent_loss_item += total_latent_loss.item() * bs
        train_total_loss_item += loss.item() * bs
        train_sparsity_item += sparsity.item() * bs

        optimizer.step()

    train_recon_loss_item /= train_image_num
    train_latent_loss_item /= train_image_num
    train_total_loss_item /= train_image_num
    train_sparsity_item /= train_image_num

    # 验证阶段
    val_recon_loss_item = 0
    val_latent_loss_item = 0
    val_total_loss_item = 0
    val_sparsity_item = 0

    with torch.no_grad():
        for batch_idx, (satellite, vil) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", ncols=100)):
            satellite = satellite.to(device)
            bs, _, _, _,_ = satellite.shape
            vil = vil.to(device)
            bs, _, _, _,_ = vil.shape

            x_recon_sate, x_recon_radar, z_sate, z_radar, total_latent_loss, dictionary = model(satellite, vil)
            recon_loss_sate = get_recon_loss(satellite, x_recon_sate)
            recon_loss_radar = get_recon_loss(vil, x_recon_radar)
            recon_loss = recon_loss_sate + recon_loss_radar
            sparsity = (hoyer_metric(z_sate) + hoyer_metric(z_radar)) / 2

            loss = recon_loss + total_latent_loss

            val_recon_loss_item += recon_loss.item() * bs
            val_latent_loss_item += total_latent_loss.item() * bs
            val_total_loss_item += loss.item() * bs
            val_sparsity_item += sparsity.item() * bs

    val_recon_loss_item /= val_image_num
    val_latent_loss_item /= val_image_num
    val_total_loss_item /= val_image_num
    val_sparsity_item /= val_image_num

    # 写入训练和验证指标csv
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
    scheduler.step(val_total_loss_item)
        # 打印当前学习率
    print(f"Epoch {epoch+1} Current LR: {scheduler.get_last_lr()}")
    # 早停判断
    if val_total_loss_item < best_val_loss:
        best_val_loss = val_total_loss_item
        epochs_no_improve = 0
        best_model_path = os.path.join(model_fold_path, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch+1}: Validation loss improved to {best_val_loss:.6f}, saving best model.")
    else:
        epochs_no_improve += 1
        print(f"Epoch {epoch+1}: No improvement in validation loss for {epochs_no_improve} epochs.")

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

    # 定期保存模型和字典图像
    if (epoch + 1) % train_args.save_frequency == 0:
        plot_dict(dictionary, dict_fold_path, f'dictionary{epoch + 1:d}.png')
        save_path = os.path.join(model_fold_path, f'model{epoch + 1:d}.pt')
        torch.save(model.state_dict(), save_path)

