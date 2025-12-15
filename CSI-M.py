import numpy as np

# === 先定义函数 ===
def compute_csi_m_from_dict(csi_dict):
    # 初始化列表用于分别收集不同尺度的CSI
    csi_1x1_list = []
    csi_4x4_list = []
    csi_16x16_list = []

    # 只处理 'CSI' 键对应的部分
    if 'CSI' in csi_dict:
        csi_dict = csi_dict['CSI']  # 获取到CSI部分的字典

        for threshold, csi_values in csi_dict.items():
            # 如果 csi_values 是字典，则可以正常提取 CSI，否则跳过
            if isinstance(csi_values, dict):
                csi_1x1_list.append(csi_values['CSI@1×1'])
                csi_4x4_list.append(csi_values['CSI@4×4'])
                csi_16x16_list.append(csi_values['CSI@16×16'])
            else:
                print(f"Warning: Expected dictionary for threshold {threshold}, but got {type(csi_values)}")

        # 计算均值
        csi_m_1x1 = sum(csi_1x1_list) / len(csi_1x1_list) if csi_1x1_list else 0
        csi_m_4x4 = sum(csi_4x4_list) / len(csi_4x4_list) if csi_4x4_list else 0
        csi_m_16x16 = sum(csi_16x16_list) / len(csi_16x16_list) if csi_16x16_list else 0

        # 打印结果
        print(f"CSI-M@1×1:   {csi_m_1x1:.4f}")
        print(f"CSI-M@4×4:   {csi_m_4x4:.4f}")
        print(f"CSI-M@16×16: {csi_m_16x16:.4f}")

        return {
            'CSI-M@1×1': csi_m_1x1,
            'CSI-M@4×4': csi_m_4x4,
            'CSI-M@16×16': csi_m_16x16
        }
    else:
        print("Warning: No 'CSI' data found in the provided dictionary.")
        return {}

# === 再加载并调用 ===
csi_data_dict = np.load('average_preds_scores.npy', allow_pickle=True).item()
result = compute_csi_m_from_dict(csi_data_dict)
