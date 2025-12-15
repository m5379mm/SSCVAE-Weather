import torch
from torch.utils.data import Dataset,random_split
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_mutual_information
import lpips
from torchvision import datasets, transforms
import os
from PIL import Image
import pandas as pd
import random

class MNISTDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 mode: str,
                 train_val_split: float = 0.8,
                 transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # 加载 MNIST 数据集
        full_train_data = datasets.MNIST(root=root_dir, train=True, download=True)
        test_data = datasets.MNIST(root=root_dir, train=False, download=True)

        # 训练集和验证集的划分
        train_size = int(len(full_train_data) * train_val_split)
        val_size = len(full_train_data) - train_size
        train_data, val_data = random_split(full_train_data, [train_size, val_size])

        if mode == 'train':
            self.mnist_data = train_data
        elif mode == 'val':
            self.mnist_data = val_data
        elif mode == 'test':
            self.mnist_data = test_data
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, index):
        # 获取图像和标签
        img, label = self.mnist_data[index]
        #print(label)

        # 如果提供了转换操作（例如标准化），应用它
        if self.transform is not None:
            img = self.transform(img)
        #print(img.shape())
        return img, label

class GrayDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 seed: int,
                 train_ratio: float,
                 val_ratio: float,
                 mode: str,
                 patch_size: int,
                 stride_size: int,
                 transform=None):
        names = os.listdir(root_dir)
        random.seed(seed)
        random.shuffle(names)
        
        n = len(names)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)

        if mode == 'train':
            self.img_names = names[: train_n]
        elif mode == 'val':
            self.img_names = names[train_n : train_n + val_n]
        elif mode == 'test':
            self.img_names = names[train_n + val_n :]
        else:
            raise ValueError("mode: {} isn't allowed.".format(mode))

        self.root_dir = root_dir
        self.mode = mode
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.transform = transform

        self.indices = []
        for name in self.img_names:
            img_path = os.path.join(root_dir, name)
            img = Image.open(img_path)
            width, height = img.size
            if width >= patch_size and height >= patch_size:
                if mode == 'test':
                    self.indices.append(name)
                else:
                    num_patches_x = (width - patch_size) // stride_size + 1
                    num_patches_y = (height - patch_size) // stride_size + 1
                    for i in range(num_patches_x):
                        for j in range(num_patches_y):
                            x = i * self.stride_size
                            y = j * self.stride_size
                            self.indices.append((name, x, y))

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        if self.mode == 'test':
            name = self.indices[index]
            img_path = os.path.join(self.root_dir, name)
            img = Image.open(img_path)

            width, height = img.size
            num_patches_x = (width - self.patch_size) // self.stride_size + 1
            num_patches_y = (height - self.patch_size) // self.stride_size + 1
            cropped_width = (num_patches_x - 1) * self.stride_size + self.patch_size
            cropped_height = (num_patches_y - 1) * self.stride_size + self.patch_size
            
            image = img.crop((0, 0, cropped_width, cropped_height))
        else:
            name, x, y = self.indices[index]
            img_path = os.path.join(self.root_dir, name)
            img = Image.open(img_path)

            image = img.crop((x, y, x + self.patch_size, y + self.patch_size))
            name = name.replace(".jpg", f"_{x}_{y}.jpg")

        if image.mode != 'L':
            raise ValueError("image: {} isn't L mode.".format(name))
        if self.transform is not None:
            image = self.transform(image)

        return image, name

class UltrasoundDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 quality: str,
                 seed: int,
                 train_ratio: float,
                 val_ratio: float,
                 mode: str,
                 transform=None):
        organs = ['breast', 'carotid', 'kidney', 'liver', 'thyroid']
        all_image_infos = []
        for organ in organs:
            image_path = os.path.join(root_dir, organ, quality)
            image_names = os.listdir(image_path)
            image_infos = []
            for image_name in image_names:
                image_info = {
                    'organ': organ,
                    'name': image_name
                }
                image_infos.append(image_info)
            all_image_infos.extend(image_infos)
        random.seed(seed)
        random.shuffle(all_image_infos)

        n = len(all_image_infos)
        train_n = int(n * train_ratio) 
        val_n = int(n * val_ratio)

        if mode == 'train':
            self.image_infos = all_image_infos[: train_n]
        elif mode == 'val':
            self.image_infos = all_image_infos[train_n : train_n + val_n]
        elif mode == 'test':
            self.image_infos = all_image_infos[train_n + val_n :]
        else:
            raise ValueError("mode: {} isn't allowed.".format(mode))

        self.root_dir = root_dir
        self.quality = quality
        self.transform = transform

    def __len__(self):
        return len(self.image_infos)
    
    def __getitem__(self, index):
        organ = self.image_infos[index]['organ']
        name_ = self.image_infos[index]['name']
        name = organ + '_' + name_
        img_path = os.path.join(self.root_dir, organ, self.quality, name_)
        img = Image.open(img_path)

        if img.mode != 'L':
            raise ValueError("image: {} isn't L mode.".format(name))
        if self.transform is not None:
            img = self.transform(img)

        return img, name

class UltrasoundTransDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 seed: int,
                 train_ratio: float,
                 val_ratio: float,
                 mode: str,
                 transform=None):
        organs = ['breast', 'carotid', 'kidney', 'liver', 'thyroid']
        all_image_infos = []
        for organ in organs:
            image_path = os.path.join(root_dir, organ, 'high_quality')
            image_names = os.listdir(image_path)
            image_infos = []
            for image_name in image_names:
                image_info = {
                    'organ': organ,
                    'name': image_name
                }
                image_infos.append(image_info)
            all_image_infos.extend(image_infos)
        random.seed(seed)
        random.shuffle(all_image_infos)

        n = len(all_image_infos)
        train_n = int(n * train_ratio) 
        val_n = int(n * val_ratio)

        if mode == 'train':
            self.image_infos = all_image_infos[: train_n]
        elif mode == 'val':
            self.image_infos = all_image_infos[train_n : train_n + val_n]
        elif mode == 'test':
            self.image_infos = all_image_infos[train_n + val_n :]
        else:
            raise ValueError("mode: {} isn't allowed.".format(mode))

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_infos)
    
    def __getitem__(self, index):
        organ = self.image_infos[index]['organ']
        name_ = self.image_infos[index]['name']
        name = organ + '_' + name_
        img_path_low = os.path.join(self.root_dir, organ, 'low_quality', name_)
        img_path_high = os.path.join(self.root_dir, organ, 'high_quality', name_)
        img_low = Image.open(img_path_low)
        img_high = Image.open(img_path_high)

        if img_low.mode != 'L' or img_high.mode != 'L':
            raise ValueError("image: {} isn't L mode.".format(name))
        if self.transform is not None:
            img_low = self.transform(img_low)
            img_high = self.transform(img_high)

        return img_low, img_high, name

class MiniImagenet(Dataset):
    def __init__(self,
                 root_dir: str,
                 mode: str,
                 patch_size: int,
                 stride_size: int,
                 transform=None):
        csv_name = mode + '.csv'
        images_dir = os.path.join(root_dir, "images")
        csv_path = os.path.join(root_dir, csv_name)
        csv_data = pd.read_csv(csv_path)

        self.images_dir = images_dir
        self.mode = mode
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.transform = transform

        names = [i for i in csv_data["filename"].values]

        self.indices = []
        for name in names:
            img_path = os.path.join(images_dir, name)
            img = Image.open(img_path)
            width, height = img.size
            if width >= patch_size and height >= patch_size:
                self.indices.append(name)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        name = self.indices[index]
        img_path = os.path.join(self.images_dir, name)
        img = Image.open(img_path)

        if self.mode == 'test':
            width, height = img.size
            num_patches_x = (width - self.patch_size) // self.stride_size + 1
            num_patches_y = (height - self.patch_size) // self.stride_size + 1
            cropped_width = (num_patches_x - 1) * self.stride_size + self.patch_size
            cropped_height = (num_patches_y - 1) * self.stride_size + self.patch_size
            
            image = img.crop((0, 0, cropped_width, cropped_height))
        else:
            image = img

        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(name))
        if self.transform is not None:
            image = self.transform(image)

        return image, name

def get_noise(image_origin, sigma):
    image_noise = torch.empty_like(image_origin)
    image_noise.copy_(image_origin)
    noise = torch.randn_like(image_origin)
    image_noise = image_noise + sigma * noise / 255
    image_noise = torch.clamp(image_noise, 0, 1)
    return image_noise

def slice_image(image, patch_size, stride):
    n, c, h, w = image.shape 
    patch_h = patch_size 
    patch_w = patch_size

    n_patches_h = (h-patch_h) // stride + 1
    n_patches_w = (w-patch_w) // stride + 1 

    out_shape = (n_patches_h * n_patches_w, c, patch_h, patch_w)  
    patches = torch.zeros(out_shape)
    
    index = 0
    for hi in range(0, h-patch_h+1, stride):
        for wi in range(0, w-patch_w+1, stride):  
            patches[index] = image[:, :, hi:hi+patch_h, wi:wi+patch_w]
            index += 1

    return patches

def recon_image(patches, ori_shape, patch_size, stride):
    n_patches, c, h, w = patches.shape  
    n, c, orig_h, orig_w = ori_shape

    reconstructed = torch.zeros(ori_shape)

    patch_h = patch_size
    patch_w = patch_size

    idx = 0
    for hi in range(0, orig_h-patch_h+1, stride):
        for wi in range(0, orig_w-patch_w+1, stride):
            reconstructed[:,:,hi:hi+patch_h,wi:wi+patch_w] = patches[idx]
            idx += 1

    return reconstructed

# def get_recon_loss(images_origin, images_recon):
#     recon_loss = (images_origin - images_recon).pow(2).mean()
#     return recon_loss
def get_recon_loss(images_origin, images_recon):
    # images_origin 和 images_recon 现在是五维张量 [B, T, C, H, W]
    
    # 计算每个像素的平方误差
    recon_loss = (images_origin - images_recon).pow(2)  # [B, T, C, H, W]
    
    # 计算每一帧的平均损失 [B, T, C, H, W] -> [B, T]
    recon_loss = recon_loss.view(recon_loss.size(0), recon_loss.size(1), -1)  # [B, T, C*H*W]
    recon_loss = recon_loss.mean(dim=-1)  # 对每一帧计算每个样本的平均损失，结果是 [B, T]

    # 对时间维度（T）进行平均，以获得最终损失 [B, T] -> [B]
    recon_loss = recon_loss.mean(dim=1)  # 对时间维度求平均损失，结果是 [B]
    
    # 对所有样本的损失进行平均，确保是标量
    recon_loss = recon_loss.mean()  # 对所有样本的损失进行平均，确保是标量

    return recon_loss

def hoyer_metric(z):
    b, t, c, h, w = z.shape  # 获取五维张量的维度
    C = torch.tensor(c, device=z.device)  # 确保设备一致

    # 计算每个样本在 T 和 C 维度上的 L1 和 L2 范数
    l1_norm = torch.norm(z, p=1, dim=(2, 3, 4), keepdim=True)  # [B, T, 1, 1, 1]
    l2_norm = torch.norm(z, p=2, dim=(2, 3, 4), keepdim=True)  # [B, T, 1, 1, 1]

    # 计算稀疏性得分
    sparsity_score = (torch.sqrt(C) - l1_norm / l2_norm) / (torch.sqrt(C) - 1)
    
    # 返回稀疏性得分的均值
    hoyer_metric_value = torch.mean(sparsity_score)

    return hoyer_metric_value

# def hoyer_metric(z):
#     b, K, h, w = z.shape
#     K = torch.tensor(K)
    
#     l1_norm = torch.norm(z, p=1, dim=1, keepdim=True)  # [B, 1, H, W]
#     l2_norm = torch.norm(z, p=2, dim=1, keepdim=True)  # [B, 1, H, W]

#     sparsity_score = (torch.sqrt(K) - l1_norm / l2_norm) / (torch.sqrt(K) - 1)
#     hoyer_metric_value = torch.mean(sparsity_score)

#     return hoyer_metric_value

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)

def compute_indicators(image_true, image_restored):
    # [1, 1, H, W]
    LPIPS = lpips_model(image_true[0], image_restored[0])

    image_true = image_true[0][0].detach().cpu().numpy()
    image_restored = image_restored[0][0].detach().cpu().numpy()

    PSNR = peak_signal_noise_ratio(image_true, image_restored, data_range=1)
    SSIM = structural_similarity(image_true, image_restored, data_range=1)
    NMI = normalized_mutual_information(image_true, image_restored)

    return PSNR, SSIM, NMI, LPIPS.item()
