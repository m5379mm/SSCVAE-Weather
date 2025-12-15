import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch


def plot_image(image, image_fold_path, image_name, channels):
    if channels == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(image_fold_path, image_name), dpi=300)
    plt.close()
    
'''def plot_images(image_origin, image_recon, image_fold_path, image_name, channels):
    if isinstance(image_name, torch.Tensor):
        image_name = str(image_name.item())
    image_origin = image_origin[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
    plot_image(image_origin, os.path.join(image_fold_path, 'origin'), image_name, channels)

    image_recon = image_recon[0].permute(1, 2, 0).contiguous().detach().cpu().numpy()
    plot_image(image_recon, os.path.join(image_fold_path, 'recon'), image_name, channels)'''

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
