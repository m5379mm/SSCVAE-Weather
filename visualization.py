import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch
from utils.display import get_cmap
from utils.fixedValues import PREPROCESS_SCALE_SEVIR, PREPROCESS_OFFSET_SEVIR  # 归一化常量
# Get colormaps for encoded types
vis_cmap,vis_norm,vis_vmin,vis_vmax = get_cmap('vis',encoded=True)
ir069_cmap,ir069_norm,ir069_vmin,ir069_vmax = get_cmap('ir069',encoded=True)
ir107_cmap,ir107_norm,ir107_vmin,ir107_vmax = get_cmap('ir107',encoded=True)
vil_cmap,vil_norm,vil_vmin,vil_vmax = get_cmap('vil',encoded=True)
lght_cmap,lght_norm,lght_vmin,lght_vmax = get_cmap('lght',encoded=True)

def save_channels_to_file(image, image_name, image_type, file_path):
    """ 保存图像的每个通道到文件 """
    # 假设图像有 3 个通道
    if image.shape[2] == 3:
        channels = ['R', 'G', 'B']
    else:
        channels = ['Channel 1']

    # 创建文件路径
    file_path = os.path.join(file_path, f"{image_name}_{image_type}_channels.txt")

    with open(file_path, 'w') as f:
        for i, channel in enumerate(channels):
            f.write(f"{channel}:\n")
            # 将通道的二维矩阵写入文件
            np.savetxt(f, image[:, :, i], fmt='%.6f')  # 保存每个通道的矩阵内容
            f.write("\n\n")

def plot_image_three_channel(image, image_fold_path, image_name, channels):
    # 创建 1 行 4 列的子图
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))  # 增加图像大小，使图像更清晰）
    # 绘制 IR 6.9 图像
    ir069 = image[:,:,0]
    # ir069 = ir069*2.9795348520448117-1.622075799366636
    ir069 = ir069 / PREPROCESS_SCALE_SEVIR['ir069'] - PREPROCESS_OFFSET_SEVIR['ir069']
    #print(ir069)
    axs[0].imshow(ir069,cmap=ir069_cmap,norm=ir069_norm,vmin=ir069_vmin,vmax=ir069_vmax)  # 使用 inferno 色图
    axs[0].set_title('IR 6.9')
    axs[0].axis('off')

    # 绘制 IR 10.7 图像
    ir107 = image[:,:,1]
    # ir107 = ir107* 2.84261423726697-1.3066503280089603
    ir107 = ir107 / PREPROCESS_SCALE_SEVIR['ir107'] - PREPROCESS_OFFSET_SEVIR['ir107']
    axs[1].imshow(ir107,cmap=ir107_cmap,norm=ir107_norm,vmin=ir107_vmin,vmax=ir107_vmax)  # 使用 plasma 色图
    axs[1].set_title('IR 10.7')
    axs[1].axis('off')

    axs[2].imshow(image[:, :, 2],cmap=lght_cmap,norm=lght_norm,vmin=lght_vmin,vmax=lght_vmax)  # 使用 cividis 色图
    axs[2].set_title('VIL')
    axs[2].axis('off')

    # 保存图像
    plt.tight_layout()  # 调整子图间的间距
    plt.savefig(os.path.join(image_fold_path, image_name), dpi=300)
    plt.close()

def plot_image_one_channel(image, image_fold_path, image_name, channels):
    # 创建 1 行 4 列的子图
    # 创建单个图像
    image = image*(4.6395+0.7035)-0.7035
    image = image/PREPROCESS_SCALE_SEVIR['vil']-PREPROCESS_OFFSET_SEVIR['vil']
        
    #print(image)
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))  # 1 行 1 列的子图，适合单通道图像
    # 绘制图像
    im = axs.imshow(image[:, :, :],cmap=vil_cmap,norm=vil_norm,vmin=vil_vmin,vmax=vil_vmax)  # 使用 cividis 色图
    
    axs.set_title(image_name)  # 设置标题为图像的名称
    axs.axis('off')  # 关闭坐标轴

    # 添加颜色条
    plt.colorbar(im, ax=axs)

    # 保存图像
    plt.tight_layout()  # 调整子图间的间距
    plt.savefig(os.path.join(image_fold_path, image_name), dpi=300)
    plt.close()


'''def plot_image(image, image_fold_path, image_name, channels):
    # 创建一个合适的图形，按通道数量排布子图
    plt.figure(figsize=(channels * 5, 5))  # 设置图形的大小（每个通道占5单位宽）

    # 如果是多通道图像，逐个绘制每个通道
    for i in range(channels):
        plt.subplot(1, channels, i + 1)  # 1行，channels列
        # 绘制每个通道的图像
        im = plt.imshow(image[:, :, i], cmap='hot')  # 使用热图色条
        plt.title(f"Channel {i+1}")
        plt.axis('off')  # 关闭坐标轴
        
        # 添加颜色条
        plt.colorbar(im)

    # 保存图像
    plt.tight_layout()  # 调整子图间的间距
    plt.savefig(os.path.join(image_fold_path, image_name), dpi=300)
    plt.close()'''
    
def plot_images(image_origin, image_recon, image_fold_path, image_name, channels):
    if isinstance(image_name, torch.Tensor):
        image_name = str(image_name.item())
    image_name= image_name+f"_channels{channels}"
    if(channels==1):
        # 处理原始图像
        image_origin = image_origin[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
        plot_image_one_channel(image_origin, os.path.join(image_fold_path, 'origin'), image_name, channels)
        # 保存原始图像的每个通道到文件
        #save_channels_to_file(image_origin, image_name, "origin", image_fold_path)

        # 处理重建图像
        image_recon = image_recon[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
        plot_image_one_channel(image_recon, os.path.join(image_fold_path, 'recon'), image_name, channels)
        # 保存重建图像的每个通道到文件
        #save_channels_to_file(image_recon, image_name, "recon", image_fold_path)
    else:
        # 处理原始图像
        image_origin = image_origin[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
        plot_image_three_channel(image_origin, os.path.join(image_fold_path, 'origin'), image_name, channels)
        # 保存原始图像的每个通道到文件
        #save_channels_to_file(image_origin, image_name, "origin", image_fold_path)

        # 处理重建图像
        image_recon = image_recon[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
        plot_image_three_channel(image_recon, os.path.join(image_fold_path, 'recon'), image_name, channels)
        # 保存重建图像的每个通道到文件
        #save_channels_to_file(image_recon, image_name, "recon", image_fold_path)'''


'''def plot_images(image_origin1, image_origin3,image_recon, image_fold_path, image_name, channels):
    if isinstance(image_name, torch.Tensor):
        image_name = str(image_name.item())
    image_name= image_name+f"_channels1"
    # 处理原始图像
    image_origin1 = image_origin1[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
    plot_image_one_channel(image_origin1, os.path.join(image_fold_path, 'origin'), image_name, 1)
    # 保存原始图像的每个通道到文件
    #save_channels_to_file(image_origin, image_name, "origin", image_fold_path)
    image_name= image_name+f"_channels3"
    image_origin3 = image_origin3[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
    plot_image_three_channel(image_origin3, os.path.join(image_fold_path, 'origin'), image_name, 3)
    # 处理重建图像
    image_name= image_name+f"_channels1"
    image_recon = image_recon[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
    plot_image_one_channel(image_recon, os.path.join(image_fold_path, 'recon'), image_name, 1)
    # 保存重建图像的每个通道到文件
    #save_channels_to_file(image_recon, image_name, "recon", image_fold_path)'''


def plot_dict(dictionary, image_fold_path, image_name):
    atom_size, num_atoms = dictionary.shape
    dictionary = dictionary.detach().cpu().numpy()

    fig, ax = plt.subplots(23, 23, figsize=(8, 8))
    for i, axis in enumerate(ax.flatten()):
        if i < num_atoms:
            row = i // 23
            col = i % 23
            atom = dictionary[:, i].reshape((16, 16))
            ax[row, col].imshow(atom, cmap='gray_r')
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            for spine in ax[row, col].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
        else:
            axis.axis('off')
        
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    plt.savefig(os.path.join(image_fold_path, image_name))
    plt.close()

def plot_dict_tsne(dictionary, image_fold_path, image_name):
    tsne = TSNE(n_components=2, random_state=42)
    dictionary_2d = tsne.fit_transform(dictionary.T.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.scatter(dictionary_2d[:, 0], dictionary_2d[:, 1], s=10)
    plt.title('Visualization of Dictionary')
    plt.savefig(os.path.join(image_fold_path, image_name))
    plt.close()
