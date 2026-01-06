#!/usr/bin/env python3
"""
快速集成增强损失函数到训练脚本

这个脚本会自动修改您的训练脚本，将原来的 segmented_weighted_loss 替换为增强的损失函数。

使用方法：
    python integrate_enhanced_loss.py --script train_finetune_lista.py --config balanced
    
配置选项：
    - conservative: 保守配置（稳定，适合初次尝试）
    - balanced: 平衡配置（推荐）
    - aggressive: 激进配置（追求更多细节）
"""

import argparse
import os
import shutil
from datetime import datetime


LOSS_CONFIGS = {
    'conservative': {
        'use_perceptual': True,
        'use_edge': True,
        'use_ssim': True,
        'use_focal': False,
        'perceptual_weight': 0.05,
        'edge_weight': 0.3,
        'ssim_weight': 0.3,
        'segmented_weight': 1.0
    },
    'balanced': {
        'use_perceptual': True,
        'use_edge': True,
        'use_ssim': True,
        'use_focal': False,
        'perceptual_weight': 0.1,
        'edge_weight': 0.5,
        'ssim_weight': 0.5,
        'segmented_weight': 1.0
    },
    'aggressive': {
        'use_perceptual': True,
        'use_edge': True,
        'use_ssim': True,
        'use_focal': False,
        'perceptual_weight': 0.2,
        'edge_weight': 1.0,
        'ssim_weight': 0.8,
        'segmented_weight': 1.0
    },
    'edge_focused': {
        'use_perceptual': True,
        'use_edge': True,
        'use_ssim': False,
        'use_focal': False,
        'perceptual_weight': 0.1,
        'edge_weight': 1.5,
        'ssim_weight': 0.0,
        'segmented_weight': 1.0
    }
}


def backup_file(filepath):
    """创建备份文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ 已创建备份: {backup_path}")
    return backup_path


def generate_import_code():
    """生成导入代码"""
    return """
# ==================== Enhanced Loss (Added) ====================
from enhanced_losses import EnhancedReconstructionLoss
# ===============================================================
"""


def generate_criterion_code(config_name):
    """生成损失函数初始化代码"""
    config = LOSS_CONFIGS[config_name]
    
    code = f"""
# ==================== Initialize Enhanced Loss (Added) ====================
criterion = EnhancedReconstructionLoss(
    use_perceptual={config['use_perceptual']},
    use_edge={config['use_edge']},
    use_ssim={config['use_ssim']},
    use_focal={config['use_focal']},
    perceptual_weight={config['perceptual_weight']},
    edge_weight={config['edge_weight']},
    ssim_weight={config['ssim_weight']},
    segmented_weight={config['segmented_weight']}
).to(device)

print("\\n" + "="*60)
print("🎨 Using Enhanced Reconstruction Loss")
print("="*60)
print(f"Configuration: {config_name}")
print(f"  Segmented Weight:  {{criterion.segmented_weight:.3f}}")
if criterion.use_perceptual:
    print(f"  Perceptual Weight: {{criterion.perceptual_weight:.3f}} ✓")
if criterion.use_edge:
    print(f"  Edge Weight:       {{criterion.edge_weight:.3f}} ✓")
if criterion.use_ssim:
    print(f"  SSIM Weight:       {{criterion.ssim_weight:.3f}} ✓")
print("="*60 + "\\n")
# ==========================================================================
"""
    return code


def generate_loss_computation_code():
    """生成损失计算代码"""
    return """
        # ==================== Enhanced Loss Computation (Modified) ====================
        # Original: reconstruction_loss = segmented_weighted_loss(x_recon_trans, vil)
        reconstruction_loss, loss_dict = criterion(x_recon_trans, vil)
        
        # Optional: Print loss breakdown every N batches
        if batch_idx % 50 == 0:
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
            print(f"  Batch {batch_idx} | {loss_str}")
        # ==============================================================================
"""


def print_manual_instructions(config_name):
    """打印手动集成说明"""
    config = LOSS_CONFIGS[config_name]
    
    print("\n" + "=" * 80)
    print("📝 手动集成增强损失函数的步骤")
    print("=" * 80)
    
    print("\n步骤 1: 在导入部分添加（文件顶部）")
    print("-" * 80)
    print(generate_import_code())
    
    print("\n步骤 2: 在模型初始化后添加（约第60-80行，model.to(device)之后）")
    print("-" * 80)
    print(generate_criterion_code(config_name))
    
    print("\n步骤 3: 替换损失计算（在训练循环中）")
    print("-" * 80)
    print("将以下代码：")
    print("    reconstruction_loss = segmented_weighted_loss(x_recon_trans, vil)")
    print("\n替换为：")
    print(generate_loss_computation_code())
    
    print("\n步骤 4: 在CSV记录中添加各项损失（可选）")
    print("-" * 80)
    print("""
# 在 fieldnames 中添加
fieldnames = ['Epoch', 'Train Recon Loss', 'Train Perceptual Loss', 
              'Train Edge Loss', 'Train SSIM Loss', ...]

# 在训练循环中累积各项损失
train_perceptual_loss_item += loss_dict.get('perceptual', 0) * bs
train_edge_loss_item += loss_dict.get('edge', 0) * bs
train_ssim_loss_item += loss_dict.get('ssim', 0) * bs
""")
    
    print("\n" + "=" * 80)
    print("✅ 完成以上步骤后，重新运行训练脚本即可！")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='集成增强损失函数')
    parser.add_argument('--script', type=str, default='train_finetune_lista.py',
                       help='训练脚本名称')
    parser.add_argument('--config', type=str, default='balanced',
                       choices=list(LOSS_CONFIGS.keys()),
                       help='损失配置')
    parser.add_argument('--show-only', action='store_true',
                       help='仅显示集成说明，不修改文件')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🔧 增强损失函数集成工具")
    print("=" * 80)
    print(f"\n目标脚本: {args.script}")
    print(f"损失配置: {args.config}")
    
    # 显示配置详情
    config = LOSS_CONFIGS[args.config]
    print(f"\n配置详情:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    
    # 打印手动集成说明
    print_manual_instructions(args.config)
    
    # 生成配置文件
    config_file = f'/root/autodl-tmp/Sevir/loss_config_{args.config}.py'
    with open(config_file, 'w') as f:
        f.write(f"""# Enhanced Loss Configuration: {args.config}
# Auto-generated by integrate_enhanced_loss.py

LOSS_CONFIG = {config}

# Usage in training script:
# from loss_config_{args.config} import LOSS_CONFIG
# criterion = EnhancedReconstructionLoss(**LOSS_CONFIG).to(device)
""")
    print(f"\n💾 已生成配置文件: {config_file}")
    
    print("\n" + "=" * 80)
    print("📚 相关文档")
    print("=" * 80)
    print("  - 完整解决方案: /root/autodl-tmp/Sevir/SOLUTION_SMOOTH_OUTPUT.md")
    print("  - 损失函数代码: /root/autodl-tmp/Sevir/enhanced_losses.py")
    print("=" * 80)


if __name__ == "__main__":
    main()


