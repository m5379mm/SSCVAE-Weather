"""
对比不同训练阶段的效果
可视化展示：阶段0 vs 阶段1 vs 阶段2(GAN)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_training_stages():
    """对比三个阶段的训练损失曲线"""
    
    # 定义各阶段的结果路径
    stages = {
        'Stage 0: 预训练': '/root/autodl-tmp/results/sscvae_recon_sevir_trans/training_losses.csv',
        'Stage 1: LISTA+MLP': '/root/autodl-tmp/results/sscvae_recon_sevir_trans_lista/training_losses.csv',
        'Stage 2: 全模型+GAN': '/root/autodl-tmp/results/sscvae_recon_sevir_gan/training_losses.csv',
    }
    
    # 加载数据
    dfs = {}
    for stage_name, path in stages.items():
        if os.path.exists(path):
            dfs[stage_name] = pd.read_csv(path)
            print(f"✓ 加载 {stage_name}: {len(dfs[stage_name])} epochs")
        else:
            print(f"✗ 未找到 {stage_name}: {path}")
    
    if len(dfs) == 0:
        print("❌ 没有找到任何训练日志！")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SSCVAE 训练阶段对比', fontsize=16, fontweight='bold')
    
    # 颜色方案
    colors = {
        'Stage 0: 预训练': '#1f77b4',
        'Stage 1: LISTA+MLP': '#ff7f0e',
        'Stage 2: 全模型+GAN': '#2ca02c',
    }
    
    # 1. Train Recon Loss
    ax = axes[0, 0]
    for stage_name, df in dfs.items():
        ax.plot(df['Epoch'], df['Train Recon Loss'], 
               label=stage_name, color=colors[stage_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Val Recon Loss
    ax = axes[0, 1]
    for stage_name, df in dfs.items():
        ax.plot(df['Epoch'], df['Val Recon Loss'], 
               label=stage_name, color=colors[stage_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Val Reconstruction Loss ⭐')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Sparsity
    ax = axes[0, 2]
    for stage_name, df in dfs.items():
        ax.plot(df['Epoch'], df['Train Sparsity'], 
               label=stage_name, color=colors[stage_name], linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hoyer Metric')
    ax.set_title('Sparsity (Hoyer Metric)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Latent Trans Loss
    ax = axes[1, 0]
    for stage_name, df in dfs.items():
        ax.plot(df['Epoch'], df['Train Latent Trans Loss'], 
               label=stage_name, color=colors[stage_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Latent Translation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Total Loss
    ax = axes[1, 1]
    for stage_name, df in dfs.items():
        ax.plot(df['Epoch'], df['Val Total Loss'], 
               label=stage_name, color=colors[stage_name], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Val Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. GAN Loss (如果存在)
    ax = axes[1, 2]
    has_gan = False
    for stage_name, df in dfs.items():
        if 'Train GAN Loss' in df.columns:
            ax.plot(df['Epoch'], df['Train GAN Loss'], 
                   label=f'{stage_name} (Train)', color=colors[stage_name], linewidth=2)
            ax.plot(df['Epoch'], df['Val GAN Loss'], 
                   label=f'{stage_name} (Val)', color=colors[stage_name], 
                   linewidth=2, linestyle='--', alpha=0.7)
            has_gan = True
        
        # 如果有 D Loss，也画出来
        if 'Train D Loss' in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df['Epoch'], df['Train D Loss'], 
                    label='Discriminator Loss', color='red', linewidth=2, linestyle=':')
            ax2.set_ylabel('D Loss', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')
    
    if has_gan:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('GAN Loss')
        ax.set_title('GAN Loss (Stage 2 only)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'GAN Loss\n(仅在阶段2可用)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # 保存图片
    save_path = '/root/autodl-tmp/results/training_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 对比图已保存到: {save_path}")
    
    # 打印最终结果对比
    print("\n" + "=" * 70)
    print("📊 各阶段最终效果对比")
    print("=" * 70)
    
    results = []
    for stage_name, df in dfs.items():
        last_row = df.iloc[-1]
        results.append({
            'Stage': stage_name,
            'Epochs': len(df),
            'Final Train Recon': f"{last_row['Train Recon Loss']:.5f}",
            'Final Val Recon': f"{last_row['Val Recon Loss']:.5f}",
            'Final Sparsity': f"{last_row['Train Sparsity']:.2f}",
            'Min Val Recon': f"{df['Val Recon Loss'].min():.5f}",
        })
    
    for r in results:
        print(f"\n{r['Stage']}:")
        print(f"  训练轮数: {r['Epochs']}")
        print(f"  最终 Train Recon Loss: {r['Final Train Recon']}")
        print(f"  最终 Val Recon Loss: {r['Final Val Recon']}")
        print(f"  最佳 Val Recon Loss: {r['Min Val Recon']} ⭐")
        print(f"  稀疏度: {r['Final Sparsity']}")
    
    print("\n" + "=" * 70)


def plot_single_stage(stage_name='Stage 2: 全模型+GAN'):
    """单独绘制某个阶段的详细图表"""
    
    path_map = {
        'Stage 0: 预训练': '/root/autodl-tmp/results/sscvae_recon_sevir_trans/training_losses.csv',
        'Stage 1: LISTA+MLP': '/root/autodl-tmp/results/sscvae_recon_sevir_trans_lista/training_losses.csv',
        'Stage 2: 全模型+GAN': '/root/autodl-tmp/results/sscvae_recon_sevir_gan/training_losses.csv',
    }
    
    if stage_name not in path_map:
        print(f"❌ 未知阶段: {stage_name}")
        return
    
    path = path_map[stage_name]
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return
    
    df = pd.read_csv(path)
    print(f"✓ 加载 {stage_name}: {len(df)} epochs")
    
    # 创建详细图表
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{stage_name} - 详细训练曲线', fontsize=16, fontweight='bold')
    
    # 1. Recon Loss (Train & Val)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['Epoch'], df['Train Recon Loss'], label='Train', linewidth=2)
    ax1.plot(df['Epoch'], df['Val Recon Loss'], label='Val', linewidth=2)
    ax1.set_title('Reconstruction Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Latent Losses
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['Epoch'], df['Train Latent Trans Loss'], label='Trans Loss', linewidth=2)
    ax2.plot(df['Epoch'], df['Train Latent Dist Loss'], label='Dist Loss', linewidth=2)
    ax2.set_title('Latent Losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Total Loss
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df['Epoch'], df['Train Total Loss'], label='Train', linewidth=2)
    ax3.plot(df['Epoch'], df['Val Total Loss'], label='Val', linewidth=2)
    ax3.set_title('Total Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Sparsity
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(df['Epoch'], df['Train Sparsity'], label='Train', linewidth=2, alpha=0.7)
    ax4.plot(df['Epoch'], df['Val Sparsity'], label='Val', linewidth=2, alpha=0.7)
    ax4.set_title('Sparsity (Hoyer Metric)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Hoyer Metric')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. GAN Loss (如果有)
    if 'Train GAN Loss' in df.columns:
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(df['Epoch'], df['Train GAN Loss'], label='Train GAN', linewidth=2)
        ax5.plot(df['Epoch'], df['Val GAN Loss'], label='Val GAN', linewidth=2)
        ax5.set_title('GAN Loss')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. D Loss
        if 'Train D Loss' in df.columns:
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.plot(df['Epoch'], df['Train D Loss'], label='D Loss', linewidth=2, color='red')
            ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Target (0.3-0.5)')
            ax6.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)
            ax6.set_title('Discriminator Loss')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Loss')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
    
    # 7. Learning Rate
    ax7 = fig.add_subplot(gs[2, :])
    lr_data = df['Learning Rate'].apply(eval)  # 转换字符串列表
    
    if isinstance(lr_data.iloc[0], list):
        for i, lr_name in enumerate(['Encoder', 'LISTA', 'Decoder', 'MLP']):
            if i < len(lr_data.iloc[0]):
                ax7.plot(df['Epoch'], [lr[i] for lr in lr_data], 
                        label=lr_name, linewidth=2, marker='o', markersize=3)
    
    if 'LR Discriminator' in df.columns:
        ax7.plot(df['Epoch'], df['LR Discriminator'], 
                label='Discriminator', linewidth=2, marker='s', markersize=3)
    
    ax7.set_title('Learning Rate Schedule')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Learning Rate')
    ax7.set_yscale('log')
    ax7.legend()
    ax7.grid(True, alpha=0.3, which='both')
    
    # 保存
    save_path = f'/root/autodl-tmp/results/{stage_name.replace(" ", "_").replace(":", "")}_detailed.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 详细图表已保存到: {save_path}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("🎨 SSCVAE 训练可视化工具")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # 绘制指定阶段
        stage_map = {
            '0': 'Stage 0: 预训练',
            '1': 'Stage 1: LISTA+MLP',
            '2': 'Stage 2: 全模型+GAN',
        }
        stage_name = stage_map.get(sys.argv[1], sys.argv[1])
        print(f"\n绘制单个阶段: {stage_name}")
        plot_single_stage(stage_name)
    else:
        # 绘制对比图
        print("\n绘制所有阶段对比...")
        compare_training_stages()
    
    print("\n✅ 完成！\n")



