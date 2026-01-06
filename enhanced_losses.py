"""
Enhanced Loss Functions for Image Reconstruction

Addresses the problem of overly smooth outputs by combining multiple loss terms:
- Perceptual Loss (LPIPS): Preserves semantic features
- Edge Loss (Sobel): Sharpens boundaries
- SSIM Loss: Maintains structural similarity
- Segmented Weighted Loss: Emphasizes strong convection regions
- Focal Loss (optional): Handles class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from math import exp


# ============================================================================
# Segmented Weighted Loss (Original)
# ============================================================================

def segmented_weighted_loss(x_recon, x_target):
    """
    Weighted MAE loss with emphasis on strong convection regions
    
    Args:
        x_recon: Reconstructed image [B, C, H, W, T] or [B, C, H, W]
        x_target: Target image, same shape as x_recon
    
    Returns:
        loss: Scalar loss value
    """
    # Normalize to [0, 255] if needed
    x_target_255 = x_target * 255 if x_target.max() <= 1 else x_target
    x_recon_255 = x_recon * 255 if x_recon.max() <= 1 else x_recon
    
    weights = torch.ones_like(x_target_255)
    
    # Intensity-based weighting
    weights[(x_target_255 >= 16) & (x_target_255 < 181)] = 1.0
    weights[(x_target_255 >= 181) & (x_target_255 < 219)] = 5.0
    weights[(x_target_255 >= 219) & (x_target_255 <= 255)] = 10.0
    
    abs_diff = torch.abs(x_recon - x_target)
    
    # Average over all dimensions
    loss = (weights * abs_diff).mean()
    
    return loss


# ============================================================================
# Edge Loss (Sobel)
# ============================================================================

def sobel_edges(x):
    """
    Compute edge magnitude using Sobel operator
    
    Args:
        x: Input tensor [B, C, H, W]
    
    Returns:
        Edge magnitude [B, C, H, W]
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    # Replicate for all channels
    sobel_x = sobel_x.repeat(x.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(x.size(1), 1, 1, 1)
    
    grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
    
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)


def edge_loss(pred, target):
    """
    Edge-preserving loss using Sobel operator
    
    Args:
        pred: Predicted image [B, C, H, W, T] or [B, C, H, W]
        target: Target image, same shape as pred
    
    Returns:
        loss: Scalar edge loss
    """
    # Handle temporal dimension if present
    if len(pred.shape) == 5:  # [B, C, H, W, T]
        B, C, H, W, T = pred.shape
        pred = pred.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)
        target = target.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)
    
    pred_edges = sobel_edges(pred)
    target_edges = sobel_edges(target)
    
    return F.mse_loss(pred_edges, target_edges)


# ============================================================================
# SSIM Loss
# ============================================================================

def gaussian(window_size, sigma):
    """Create Gaussian kernel"""
    gauss = torch.tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create 2D Gaussian window"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window, window_size=11, size_average=True, val_range=1.0):
    """
    Compute SSIM between two images
    
    Args:
        img1, img2: Images [B, C, H, W]
        window: Gaussian window
        window_size: Size of Gaussian kernel
        size_average: If True, return mean SSIM
        val_range: Value range of images
    
    Returns:
        SSIM value
    """
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    
    window = window.to(img1.device).type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_loss(pred, target, window_size=11):
    """
    SSIM loss function
    
    Args:
        pred: Predicted image [B, C, H, W, T] or [B, C, H, W]
        target: Target image, same shape as pred
        window_size: Size of Gaussian kernel
    
    Returns:
        loss: 1 - SSIM (lower is better)
    """
    # Handle temporal dimension if present
    if len(pred.shape) == 5:  # [B, C, H, W, T]
        B, C, H, W, T = pred.shape
        pred = pred.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)
        target = target.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)
    
    window = create_window(window_size, pred.size(1)).to(pred.device)
    ssim_val = ssim(pred, target, window, window_size=window_size, val_range=1.0)
    
    return 1 - ssim_val


# ============================================================================
# Focal Loss (Optional, for handling class imbalance)
# ============================================================================

def focal_loss(pred, target, alpha=1.0, gamma=2.0, threshold=0.7):
    """
    Focal loss for handling imbalanced data
    
    Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]
        alpha: Weighting factor
        gamma: Focusing parameter
        threshold: Threshold for binary classification
    
    Returns:
        loss: Focal loss value
    """
    # Binarize target based on threshold
    target_binary = (target > threshold).float()
    
    # Compute BCE loss
    bce_loss = F.binary_cross_entropy(pred, target_binary, reduction='none')
    
    # Compute focal weight
    pt = torch.exp(-bce_loss)
    focal_weight = (1 - pt) ** gamma
    
    # Apply focal weight
    loss = alpha * focal_weight * bce_loss
    
    return loss.mean()


# ============================================================================
# Combined Enhanced Loss
# ============================================================================

class EnhancedReconstructionLoss(nn.Module):
    """
    Enhanced reconstruction loss combining multiple terms:
    - Segmented Weighted Loss (pixel-level)
    - Perceptual Loss (LPIPS)
    - Edge Loss (Sobel)
    - SSIM Loss
    - Focal Loss (optional)
    
    Usage:
        criterion = EnhancedReconstructionLoss(
            use_perceptual=True,
            use_edge=True,
            use_ssim=True,
            perceptual_weight=0.1,
            edge_weight=0.5,
            ssim_weight=0.5
        )
        
        loss, loss_dict = criterion(pred, target)
    """
    
    def __init__(self,
                 use_perceptual=True,
                 use_edge=True,
                 use_ssim=True,
                 use_focal=False,
                 segmented_weight=1.0,
                 perceptual_weight=0.1,
                 edge_weight=0.5,
                 ssim_weight=0.5,
                 focal_weight=0.1,
                 focal_alpha=1.0,
                 focal_gamma=2.0,
                 focal_threshold=0.7,
                 lpips_net='alex'):  # 'alex' is faster than 'vgg'
        super().__init__()
        
        self.use_perceptual = use_perceptual
        self.use_edge = use_edge
        self.use_ssim = use_ssim
        self.use_focal = use_focal
        
        self.segmented_weight = segmented_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.focal_weight = focal_weight
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_threshold = focal_threshold
        
        # Initialize LPIPS model if needed
        if self.use_perceptual:
            self.lpips_model = lpips.LPIPS(net=lpips_net, verbose=False)
            # Freeze LPIPS parameters
            for param in self.lpips_model.parameters():
                param.requires_grad = False
    
    def forward(self, pred, target):
        """
        Compute enhanced reconstruction loss
        
        Args:
            pred: Predicted image [B, C, H, W, T] or [B, C, H, W]
            target: Target image, same shape as pred
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss values
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Segmented Weighted Loss (baseline)
        seg_loss = segmented_weighted_loss(pred, target)
        total_loss += self.segmented_weight * seg_loss
        loss_dict['segmented'] = seg_loss.item()
        
        # 2. Perceptual Loss (LPIPS)
        if self.use_perceptual:
            # Handle temporal dimension
            if len(pred.shape) == 5:  # [B, C, H, W, T]
                B, C, H, W, T = pred.shape
                pred_lpips = pred.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)
                target_lpips = target.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)
            else:
                pred_lpips = pred
                target_lpips = target
            
            # LPIPS requires 3-channel input, replicate if needed
            if pred_lpips.size(1) == 1:
                pred_lpips = pred_lpips.repeat(1, 3, 1, 1)
                target_lpips = target_lpips.repeat(1, 3, 1, 1)
            
            perceptual_loss_val = self.lpips_model(pred_lpips, target_lpips).mean()
            total_loss += self.perceptual_weight * perceptual_loss_val
            loss_dict['perceptual'] = perceptual_loss_val.item()
        
        # 3. Edge Loss
        if self.use_edge:
            edge_loss_val = edge_loss(pred, target)
            total_loss += self.edge_weight * edge_loss_val
            loss_dict['edge'] = edge_loss_val.item()
        
        # 4. SSIM Loss
        if self.use_ssim:
            ssim_loss_val = ssim_loss(pred, target)
            total_loss += self.ssim_weight * ssim_loss_val
            loss_dict['ssim'] = ssim_loss_val.item()
        
        # 5. Focal Loss (optional)
        if self.use_focal:
            # Handle temporal dimension
            if len(pred.shape) == 5:
                B, C, H, W, T = pred.shape
                pred_focal = pred.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)
                target_focal = target.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, H, W)
            else:
                pred_focal = pred
                target_focal = target
            
            focal_loss_val = focal_loss(pred_focal, target_focal, 
                                       alpha=self.focal_alpha, 
                                       gamma=self.focal_gamma,
                                       threshold=self.focal_threshold)
            total_loss += self.focal_weight * focal_loss_val
            loss_dict['focal'] = focal_loss_val.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================================================
# Utility Functions
# ============================================================================

def print_loss_weights(criterion):
    """Print current loss weights"""
    print("=" * 60)
    print("Enhanced Loss Configuration:")
    print("=" * 60)
    print(f"Segmented Weight:   {criterion.segmented_weight:.3f}")
    if criterion.use_perceptual:
        print(f"Perceptual Weight:  {criterion.perceptual_weight:.3f} ✓")
    if criterion.use_edge:
        print(f"Edge Weight:        {criterion.edge_weight:.3f} ✓")
    if criterion.use_ssim:
        print(f"SSIM Weight:        {criterion.ssim_weight:.3f} ✓")
    if criterion.use_focal:
        print(f"Focal Weight:       {criterion.focal_weight:.3f} ✓")
    print("=" * 60)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test the loss functions
    print("Testing Enhanced Reconstruction Loss...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    pred = torch.rand(2, 1, 128, 128, 7).to(device)  # [B, C, H, W, T]
    target = torch.rand(2, 1, 128, 128, 7).to(device)
    
    # Test 1: Conservative configuration
    print("\n1. Conservative Configuration:")
    criterion = EnhancedReconstructionLoss(
        use_perceptual=True,
        use_edge=True,
        use_ssim=True,
        perceptual_weight=0.05,
        edge_weight=0.3,
        ssim_weight=0.3
    ).to(device)
    print_loss_weights(criterion)
    
    loss, loss_dict = criterion(pred, target)
    print(f"\nLoss breakdown:")
    for key, val in loss_dict.items():
        print(f"  {key:12s}: {val:.6f}")
    
    # Test 2: Balanced configuration
    print("\n2. Balanced Configuration:")
    criterion = EnhancedReconstructionLoss(
        use_perceptual=True,
        use_edge=True,
        use_ssim=True,
        perceptual_weight=0.1,
        edge_weight=0.5,
        ssim_weight=0.5
    ).to(device)
    print_loss_weights(criterion)
    
    loss, loss_dict = criterion(pred, target)
    print(f"\nLoss breakdown:")
    for key, val in loss_dict.items():
        print(f"  {key:12s}: {val:.6f}")
    
    print("\n✅ All tests passed!")

