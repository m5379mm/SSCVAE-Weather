#!/usr/bin/env python3
"""
compute_skill_scores_mmap.py

在处理大尺寸（例如 384×384）数据时，使用 NumPy memmap 按批次加载 numpy 文件，避免一次性加载过多数据导致 OOM。

Usage:
    python compute_skill_scores_mmap.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import os
from utils.fixedValues import PREPROCESS_SCALE_SEVIR, PREPROCESS_OFFSET_SEVIR
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from properscoring import crps_ensemble  # 标准 CRPS 库

# CRPS (使用标准库 properscoring)
def compute_crps(pred, gt):
    """
    使用 properscoring 库计算标准 CRPS
    对于确定性预测，将其视为单成员集合预报
    
    Args:
        pred: torch.Tensor [B, C, H, W] or [B, C, H, W, T]
        gt: torch.Tensor [B, C, H, W] or [B, C, H, W, T]
    
    Returns:
        float: 平均 CRPS 值（归一化到数据范围）
    """
    # 转换为 numpy
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    
    # 展平所有维度（除了样本维度）
    pred_flat = pred_np.reshape(pred_np.shape[0], -1)  # [B, N]
    gt_flat = gt_np.reshape(gt_np.shape[0], -1)        # [B, N]
    
    # 对于确定性预测，ensemble 只有1个成员
    # crps_ensemble 需要 observations 形状为 [N], forecasts 形状为 [N, ensemble_size]
    crps_values = []
    for i in range(pred_flat.shape[0]):
        # 对每个样本计算 CRPS
        observations = gt_flat[i]                      # [N]
        forecasts = pred_flat[i:i+1].T                 # [N, 1] - 单成员集合（转置）
        
        # 计算 CRPS（返回每个点的 CRPS，然后取平均）
        crps_val = crps_ensemble(observations, forecasts).mean()
        crps_values.append(crps_val)
    
    # 归一化到全局数据范围 [0, 255]
    # 使用固定的数据范围，而不是每个样本的范围
    GLOBAL_DATA_RANGE = 255.0  # VIL 数据的理论最大值
    
    mean_crps = np.mean(crps_values)
    normalized_crps = mean_crps / GLOBAL_DATA_RANGE
    
    return float(normalized_crps)

# SSIM
def compute_ssim_torch(pred, gt):
    pred = pred.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    # 使用固定的全局数据范围 [0, 255]
    return ssim(gt, pred, data_range=255.0)

# HSS
def compute_hss(pred, gt, threshold):
    pred_binary = (pred >= threshold).cpu().numpy()
    gt_binary = (gt >= threshold).cpu().numpy()

    TP = np.logical_and(pred_binary, gt_binary).sum()
    TN = np.logical_and(~pred_binary, ~gt_binary).sum()
    FP = np.logical_and(pred_binary, ~gt_binary).sum()
    FN = np.logical_and(~pred_binary, gt_binary).sum()

    #print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")  # 🔍 打印混淆矩阵元素


    numerator = 2 * (TP * TN - FP * FN)
    denominator = ((TP + FN)*(FN + TN) + (TP + FP)*(FP + TN)) + 1e-8
    return numerator / denominator

 # squeeze if needed

    # return F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)


# ─── 0) USER CONFIG ────────────────────────────────────────────────────────────
# 将 model_name 改成你对应的文件夹名称
model_name = "SimVP_incepu_hko7_255_thr_rain_0.1_rainfall_thr185_002_01_nig3_newsplit"

# 原来保存 preds.npy、trues.npy 的路径
PRED_PATH = f"/root/autodl-tmp/results/sscvae_recon_sevir_gan/images/reconstructed_images_single"
TRUE_PATH = f"/root/autodl-tmp/results/sscvae_recon_sevir_gan/images/true_images_single"



# 结果要保存到的文件
RESULTS_PATH = "csi_scores.npy"

# 每次只加载多少样本到 GPU／内存
BATCH_SIZE = 16

# 要计算的降水阈值列表（mm）
THRESHOLDS = [16,74,133,160,181,219]


# ─── 1) SSIM & PSNR HELPERS (跟之前保持不变) ──────────────────────────────────

@torch.no_grad()
def cal_SSIM(gt, pred, is_img=True):
    drange = float(torch.max(gt) - torch.min(gt))
    metric = StructuralSimilarityIndexMeasure(data_range=drange).to(gt.device)
    if is_img:
        pred = torch.clamp(pred, torch.min(gt), torch.max(gt))
    p = rearrange(pred, 'n t c h w -> (n t) c h w')
    g = rearrange(gt,   'n t c h w -> (n t) c h w')
    print(f"Pred min: {torch.min(pred)}, Pred max: {torch.max(pred)}")
    print(f"GT min: {torch.min(gt)}, GT max: {torch.max(gt)}")
    return float(metric(p, g).cpu())

# 调试阈值
def _threshold(target, pred, T):
    t = (target >= T).float()
    p = (pred   >= T).float()
    print(f"Threshold: {T}, Target bin sum: {t.sum()}, Prediction bin sum: {p.sum()}")
    return t, p


@torch.no_grad()
def cal_PSNR(gt, pred, is_img=True):
    metric = PeakSignalNoiseRatio().to(gt.device)
    if is_img:
        pred = torch.clamp(pred, torch.min(gt), torch.max(gt))
    p = rearrange(pred, 'n t c h w -> (n t) c h w')
    g = rearrange(gt,   'n t c h w -> (n t) c h w')
    total = 0.0
    for i in range(p.shape[0]):
        total += float(metric(p[i], g[i]).cpu())
    return total / p.shape[0]


# ─── 2) THRESHOLDING UTILITY ──────────────────────────────────────────────────

def _threshold(target, pred, T):
    t = (target >= T).float()
    p = (pred   >= T).float()
    # nanmask = torch.isnan(target) | torch.isnan(pred)
    # t[nanmask] = 0.0
    # p[nanmask] = 0.0
    return t, p


# ─── 3) SKILL‐SCORE CLASS ─────────────────────────────────────────────────────

class SEVIRSkillScore:
    def __init__(self, thresholds, preprocess_type="sevir", eps=1e-4):
        self.thresholds  = thresholds
        self.eps         = eps
        self.preproc     = preprocess_type
        shape = (len(self.thresholds),)
        # 1×1 统计
        self.hits   = torch.zeros(shape)
        self.misses = torch.zeros(shape)
        self.fas    = torch.zeros(shape)
        # 4×4 max‐pool
        self.hits4   = torch.zeros(shape)
        self.misses4 = torch.zeros(shape)
        self.fas4    = torch.zeros(shape)
        # 16×16 max‐pool
        self.hits16   = torch.zeros(shape)
        self.misses16 = torch.zeros(shape)
        self.fas16    = torch.zeros(shape)

    def preprocess(self, x):
        # SEVIR-normalization undo → [0,255]
        #x = downsample_to_128x128(x)
        return x

    def preprocess_pool(self, x, k):
        v = F.max_pool2d(x, kernel_size=k, stride=k)
        return v


    def _acc(self, pred, target, pool_k=None):
        if pool_k is None:
            P = self.preprocess(pred)
            T = self.preprocess(target)
        else:
            P = self.preprocess(self.preprocess_pool(pred, pool_k))
            T = self.preprocess(self.preprocess_pool(target, pool_k))

        for i, thr in enumerate(self.thresholds):
            t_bin, p_bin = _threshold(T, P, thr)
            dims = list(range(t_bin.dim()))
            hits   = torch.sum( t_bin * p_bin,    dim=dims).int()
            misses = torch.sum( t_bin * (1-p_bin), dim=dims).int()
            fas    = torch.sum((1-t_bin)* p_bin,   dim=dims).int()

            if pool_k is None:
                self.hits[i]   += hits
                self.misses[i] += misses
                self.fas[i]    += fas
            elif pool_k == 4:
                self.hits4[i]   += hits
                self.misses4[i] += misses
                self.fas4[i]    += fas
            elif pool_k == 16:
                self.hits16[i]   += hits
                self.misses16[i] += misses
                self.fas16[i]    += fas
        # print(self.hits[i])
        # input()

    def update(self, pred, target):
        # 把内部状态移到与 pred 同样的 device
        device = pred.device
        for name in [
            'hits', 'misses', 'fas',
            'hits4', 'misses4', 'fas4',
            'hits16','misses16','fas16'
        ]:
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device))
        # pred, target 都是 [b, t, c, h, w]
        self._acc(pred, target, pool_k=None)
        self._acc(pred, target, pool_k=4)
        self._acc(pred, target, pool_k=16)

    def compute(self):
        out = {}
        for i, thr in enumerate(self.thresholds):
            def csi(h,m,f): return h.float()/(h+m+f+self.eps)
            out[thr] = {
                "CSI@1×1":  float(csi(self.hits[i],   self.misses[i],   self.fas[i]).cpu()),
                "CSI@4×4":  float(csi(self.hits4[i],  self.misses4[i],  self.fas4[i]).cpu()),
                "CSI@16×16":float(csi(self.hits16[i], self.misses16[i], self.fas16[i]).cpu()),
            }
        return out


# ─── 4) MAIN SCRIPT（按批次利用 memmap 加载）────────────────────────────────
def compute_average(npy_directory, pred_directory):
    pred_files = [f for f in os.listdir(pred_directory) if f.endswith('.npy')]
    npy_files = [f for f in os.listdir(npy_directory) if f.endswith('.npy')]

    skill = SEVIRSkillScore(THRESHOLDS)

    total_crps = 0.0
    total_ssim = 0.0
    total_hss = 0.0
    count = 0

    for npy_file, pred_file in zip(npy_files, pred_files):
                # 打印每次配对的文件名
        # print(f"Processing prediction file: {pred_file} and ground truth file: {npy_file}")
        # input()
        pred_data = np.load(os.path.join(pred_directory, pred_file))
        true_data = np.load(os.path.join(npy_directory, npy_file))

        pred_tensor = torch.from_numpy(pred_data).float().cuda()
        true_tensor = torch.from_numpy(true_data).float().cuda()

        pred_tensor = pred_tensor*(4.6395+0.7035)-0.7035
        pred_tensor = pred_tensor/PREPROCESS_SCALE_SEVIR['vil']-PREPROCESS_OFFSET_SEVIR['vil']

        true_tensor = true_tensor*(4.6395+0.7035)-0.7035
        true_tensor = true_tensor/PREPROCESS_SCALE_SEVIR['vil']-PREPROCESS_OFFSET_SEVIR['vil']
        # print(pred_tensor.min(),pred_tensor.max())
        # print(true_tensor.min(),true_tensor.max())
        # input()

        skill.update(pred_tensor, true_tensor)

        # 平均CRPS/SSIM/HSS计算（以每对图为单位）
        crps_val = compute_crps(pred_tensor, true_tensor)
        ssim_val = compute_ssim_torch(pred_tensor, true_tensor)
        hss_val = compute_hss(pred_tensor, true_tensor, threshold=181)

        total_crps += crps_val
        total_ssim += ssim_val
        total_hss  += hss_val

        count += 1

    # 计算平均结果
    avg_crps = total_crps / count
    avg_ssim = total_ssim / count
    avg_hss  = total_hss / count

    # 打印指标
    print(f"平均 CRPS: {avg_crps:.4f}")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均 HSS (181阈值): {avg_hss:.4f}")

    # 原始CSI结果
    csi_results = skill.compute()

    # 合并所有结果
    final_results = {
        "CRPS": avg_crps,
        "SSIM": avg_ssim,
        "HSS@181": avg_hss,
        "CSI": csi_results
    }

    return final_results


def main():
    print("开始计算指标……")
    results = compute_average(TRUE_PATH,PRED_PATH)
    np.save("average_preds_scores.npy", results)
    print("已保存到 average_preds_scores.npy")




if __name__ == "__main__":
    main()
# def main():
#     # 1) 用 memmap 模式只读地打开 .npy，大文件不会一次性读入内存
#     print("正在以 mmap 方式打开大文件……")
#     preds_mmap = np.load(PRED_PATH, mmap_mode='r')   # shape 例如 (N, T, C, H, W)
#     preds_mmap = preds_mmap.squeeze()
#     trues_mmap = np.load(TRUE_PATH, mmap_mode='r') # shape 例如 (N, T, C, H, W) 或 (N, T, H, W)
#     print(f"preds_mmap 形状: {preds_mmap.shape}, trues_mmap 形状: {trues_mmap.shape}")
#     #preds_mmap = preds_mmap[:,:,0,:,:]#取u
#     preds_mmap = preds_mmap[:, :, np.newaxis, :, :]  # 添加通道维度
#     print(f"preds_mmap 形状: {preds_mmap.shape}, trues_mmap 形状: {trues_mmap.shape}")

    
#     N = preds_mmap.shape[0]  # 样本数
#     #print(f"总样本数 N = {N}, 序列长度 T = {T}, 高 H = {H}, 宽 W = {W}")

#     # 2) 建立 CSI 统计器
#     skill = SEVIRSkillScore(THRESHOLDS)

#     # 3) 按 batch 读取和累积
#     for start_idx in range(0, N, BATCH_SIZE):
#         end_idx = min(start_idx + BATCH_SIZE, N)
#         print(f"正在处理 [{start_idx}:{end_idx}] 样本……")

#         # 从 memmap slice 出这一块
#         # preds_slice: numpy 数组，形状 (B, T, C, H, W) 或 (B, T, H, W)
#         preds_slice = preds_mmap[start_idx:end_idx]


#         trues_slice = trues_mmap[start_idx:end_idx]
        
#         # 转为 torch 并搬到 GPU（如果有）
#         P_batch = torch.from_numpy(preds_slice).float()
#         T_batch = torch.from_numpy(trues_slice).float()
#         if torch.cuda.is_available():
#             P_batch = P_batch.cuda()
#             T_batch = T_batch.cuda()

#         # 更新 CSI 统计
#         skill.update(P_batch, T_batch)

#         # 主动释放该 batch 占用的显存
#         del P_batch, T_batch
#         torch.cuda.empty_cache()

#     # 4) 计算最终结果并打印
#     results = skill.compute()
#     print("\n=== CSI scores ===")
#     for thr, d in results.items():
#         print(f"Threshold={thr:3f} mm → 1×1: {d['CSI@1×1']:.5f}, 4×4: {d['CSI@4×4']:.5f}, 16×16: {d['CSI@16×16']:.5f}")

#     # 5) 保存到 .npy
#     np.save(RESULTS_PATH, results)
#     print(f"\n已将完整的结果字典存到：{RESULTS_PATH}")


# if __name__ == "__main__":
#     main()