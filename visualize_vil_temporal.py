#!/usr/bin/env python3
"""
Visualize VIL data at frames 0, 20, 40 for each file
专门绘制每个文件的 VIL 数据在第 0、20、40 帧
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid warnings
import matplotlib.pyplot as plt
import os
import sys
import warnings
from tqdm import tqdm

# Suppress matplotlib warning about callbacks
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add display module path
sys.path.append('/root/autodl-tmp/Sevir')
from utils.display import get_cmap

# ==================== Configuration ====================
DATA_DIR = '/root/autodl-tmp/earthformer-satellite-to-radar-main/data/test'
OUTPUT_DIR = '/root/autodl-tmp/Sevir/vil_temporal_visualizations'
FRAMES = [0, 20, 40]  # Frames to visualize

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get VIL colormap
vil_cmap, vil_norm, vil_vmin, vil_vmax = get_cmap('vil', encoded=True)


def visualize_vil_frames(file_path, frames=[0, 20, 40], save_path=None):
    """
    Visualize VIL data at specific frames
    
    Args:
        file_path: Path to HDF5 file
        frames: List of frame indices to visualize
        save_path: Path to save the visualization
    """
    filename = os.path.basename(file_path)
    
    with h5py.File(file_path, 'r') as f:
        vil_data = f['vil'][:]  # [H, W, T]
        total_frames = vil_data.shape[2]
        
        # Validate frame indices
        valid_frames = [f for f in frames if 0 <= f < total_frames]
        if len(valid_frames) != len(frames):
            print(f"Warning: Some frames are out of range for {filename} (total: {total_frames})")
            frames = valid_frames
        
        if len(frames) == 0:
            print(f"Error: No valid frames for {filename}")
            return
    
    # Create figure
    fig, axes = plt.subplots(1, len(frames), figsize=(6*len(frames), 6))
    if len(frames) == 1:
        axes = [axes]
    
    # Plot each frame
    for idx, (ax, frame_idx) in enumerate(zip(axes, frames)):
        vil_frame = vil_data[:, :, frame_idx]
        
        # Plot
        im = ax.imshow(vil_frame, cmap=vil_cmap, norm=vil_norm,
                      vmin=vil_vmin, vmax=vil_vmax)
        
        # Statistics
        non_zero = np.sum(vil_frame > 0)
        total_pixels = vil_frame.size
        non_zero_pct = 100 * non_zero / total_pixels
        
        # Title with statistics
        ax.set_title(f'Frame {frame_idx}\n'
                    f'Range: [{vil_frame.min()}, {vil_frame.max()}]\n'
                    f'Mean: {vil_frame.mean():.2f} | Non-zero: {non_zero_pct:.1f}%',
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('VIL (kg/m²)', rotation=270, labelpad=15)
    
    # Main title
    fig.suptitle(f'VIL Temporal Evolution\n{filename}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_name = f"{filename.replace('.h5', '')}_vil_frames.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # Use 'all' to close all figures cleanly
    
    return save_path


def visualize_vil_all_frames(file_path, save_path=None):
    """
    Visualize all VIL frames in a grid
    
    Args:
        file_path: Path to HDF5 file
        save_path: Path to save the visualization
    """
    filename = os.path.basename(file_path)
    
    with h5py.File(file_path, 'r') as f:
        vil_data = f['vil'][:]  # [H, W, T]
        total_frames = vil_data.shape[2]
    
    # Select frames to display (every 5 frames)
    display_frames = list(range(0, total_frames, 5))
    num_frames = len(display_frames)
    
    # Calculate grid size
    ncols = min(7, num_frames)
    nrows = (num_frames + ncols - 1) // ncols
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if num_frames > 1 else [axes]
    
    # Plot each frame
    for idx, frame_idx in enumerate(display_frames):
        vil_frame = vil_data[:, :, frame_idx]
        
        ax = axes[idx]
        im = ax.imshow(vil_frame, cmap=vil_cmap, norm=vil_norm,
                      vmin=vil_vmin, vmax=vil_vmax)
        
        ax.set_title(f'Frame {frame_idx}\nMax: {vil_frame.max()}',
                    fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    # Main title
    fig.suptitle(f'VIL All Frames (Every 5)\n{filename}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_name = f"{filename.replace('.h5', '')}_vil_all_frames.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close('all')  # Use 'all' to close all figures cleanly
    
    return save_path


def batch_process(data_dir, num_files=None, mode='three_frames'):
    """
    Batch process multiple files
    
    Args:
        data_dir: Directory containing HDF5 files
        num_files: Number of files to process (None for all)
        mode: 'three_frames' or 'all_frames'
    """
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    
    if not files:
        print(f"❌ No .h5 files found in {data_dir}")
        return
    
    if num_files is not None:
        files = files[:num_files]
    
    print(f"Found {len(files)} files to process")
    print(f"Mode: {mode}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    success_count = 0
    error_count = 0
    
    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(data_dir, filename)
        
        try:
            if mode == 'three_frames':
                save_path = visualize_vil_frames(file_path, frames=FRAMES)
            elif mode == 'all_frames':
                save_path = visualize_vil_all_frames(file_path)
            else:
                print(f"Unknown mode: {mode}")
                return
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {filename}: {e}")
            error_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"   Success: {success_count}/{len(files)}")
    print(f"   Errors: {error_count}/{len(files)}")
    print(f"   Output: {OUTPUT_DIR}")


def compare_files(file_paths, frames=[0, 20, 40]):
    """
    Compare VIL data across multiple files
    
    Args:
        file_paths: List of file paths to compare
        frames: Frames to display for each file
    """
    num_files = len(file_paths)
    num_frames = len(frames)
    
    # Create figure
    fig, axes = plt.subplots(num_files, num_frames, 
                             figsize=(5*num_frames, 5*num_files))
    
    if num_files == 1:
        axes = axes.reshape(1, -1)
    if num_frames == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each file
    for file_idx, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        
        with h5py.File(file_path, 'r') as f:
            vil_data = f['vil'][:]
        
        # Plot each frame
        for frame_idx, frame_num in enumerate(frames):
            vil_frame = vil_data[:, :, frame_num]
            
            ax = axes[file_idx, frame_idx]
            im = ax.imshow(vil_frame, cmap=vil_cmap, norm=vil_norm,
                          vmin=vil_vmin, vmax=vil_vmax)
            
            # Title
            if file_idx == 0:
                ax.set_title(f'Frame {frame_num}', fontsize=12, fontweight='bold')
            
            # Y-axis label (filename)
            if frame_idx == 0:
                ax.set_ylabel(filename, fontsize=10, rotation=0, 
                            ha='right', va='center')
            
            ax.axis('off')
            
            # Colorbar on the right
            if frame_idx == num_frames - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('VIL', rotation=270, labelpad=15)
    
    fig.suptitle(f'VIL Comparison Across {num_files} Files',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, 'vil_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # Use 'all' to close all figures cleanly
    
    print(f"✅ Comparison saved to: {save_path}")
    return save_path


# ==================== Main Program ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize VIL data at frames 0, 20, 40'
    )
    parser.add_argument('--mode', type=str, default='three_frames',
                       choices=['three_frames', 'all_frames', 'compare'],
                       help='Visualization mode')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                       help='Data directory')
    parser.add_argument('--num_files', type=int, default=None,
                       help='Number of files to process (None for all)')
    parser.add_argument('--frames', type=int, nargs='+', default=FRAMES,
                       help='Frame indices to visualize')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--file', type=str, default=None,
                       help='Single file to process')
    parser.add_argument('--compare_files', type=str, nargs='+', default=None,
                       help='Files to compare')
    
    args = parser.parse_args()
    
    # Update global settings
    OUTPUT_DIR = args.output_dir
    FRAMES = args.frames
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"VIL Temporal Visualization Tool")
    print(f"={'='*60}")
    print(f"Frames to visualize: {FRAMES}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    if args.mode == 'compare' and args.compare_files:
        # Compare mode
        compare_files(args.compare_files, frames=FRAMES)
    
    elif args.file:
        # Single file mode
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            exit(1)
        
        print(f"Processing single file: {args.file}")
        
        if args.mode == 'three_frames':
            save_path = visualize_vil_frames(args.file, frames=FRAMES)
        elif args.mode == 'all_frames':
            save_path = visualize_vil_all_frames(args.file)
        
        print(f"✅ Saved to: {save_path}")
    
    else:
        # Batch mode
        batch_process(args.data_dir, args.num_files, args.mode)
    
    print(f"\n🎉 All done!")

