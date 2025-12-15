import numpy as np

def read_npy_file(file_path='average_preds_scores.npy'):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"\n✅ 成功读取文件: {file_path}")
        print("📄 文件内容如下：\n")
        print(data)
        return data
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None

# 主程序
if __name__ == "__main__":
    read_npy_file()
