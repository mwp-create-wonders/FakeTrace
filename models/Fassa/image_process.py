import os
import cv2
import numpy as np
import pywt
from tqdm import tqdm  # 进度条，方便查看处理进度

# -------------------------- 配置参数 --------------------------
DATASET_PATH = "autodl-tmp/data/tampered_shuffle"  # 数据集文件夹路径
SAVE_ROOT_PATH = "autodl-tmp/data_shuffle_tensor"  # 单个npy文件保存根目录
WAVELET = 'db1'  # 小波基选择
TARGET_SIZE = (1024, 1024)  # 目标尺寸
FILL_VALUE = 0  # 尺寸恢复时的填充值（0填充）

# -------------------------- 核心工具函数 --------------------------
def binary_encoding(matrix):
    """
    对输入矩阵执行二进制编码（8邻域规则），返回编码矩阵（排除边界）
    :param matrix: 输入矩阵（如512x512/256x256）
    :return: 编码矩阵（510x510/254x254，0-255）
    """
    h, w = matrix.shape
    encode_matrix = np.zeros((h-2, w-2), dtype=np.uint8)
    # 8邻域偏移（A0-A7：上、上右、右、下右、下、下左、左、上左）
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    # 遍历内部区域
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = matrix[i, j]
            binary = []
            for dx, dy in offsets:
                neighbor = matrix[i+dx, j+dy]
                binary.append('0' if neighbor > center else '1')
            # 二进制转十进制
            encode_matrix[i-1, j-1] = int(''.join(binary), 2)
    return encode_matrix

def lbp_to_8channels(matrix):
    """
    将输入矩阵的8邻域LBP编码转换为8通道0/1张量（排除边界）
    :param matrix: 输入矩阵（如512x512）
    :return: 8通道张量（510x510x8），每个通道值为0或1
    """
    h, w = matrix.shape
    # 初始化8通道输出（510x510x8）
    lbp_channels = np.zeros((h-2, w-2, 8), dtype=np.uint8)
    # 8邻域偏移（A0-A7：上、上右、右、下右、下、下左、左、上左）
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    # 遍历内部区域
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = matrix[i, j]
            # 对每个邻域方向生成0/1值
            for idx, (dx, dy) in enumerate(offsets):
                neighbor = matrix[i+dx, j+dy]
                lbp_channels[i-1, j-1, idx] = 0 if neighbor > center else 1
    return lbp_channels

def restore_size(matrix, target_size, fill_value=FILL_VALUE):
    """
    将矩阵（如510x510/510x510x8）恢复为目标尺寸（如512x512/512x512x8）
    方法：在矩阵四周填充一行/列指定值（默认0）
    :param matrix: 输入小矩阵（2D或3D）
    :param target_size: 目标尺寸 (h, w)
    :param fill_value: 填充值
    :return: 恢复后的矩阵
    """
    if len(matrix.shape) == 2:
        # 2D矩阵（原逻辑）
        h, w = matrix.shape
        target_h, target_w = target_size
        restored = np.full((target_h, target_w), fill_value, dtype=matrix.dtype)
        restored[1:-1, 1:-1] = matrix
    elif len(matrix.shape) == 3:
        # 3D张量（新增：处理8通道情况）
        h, w, c = matrix.shape
        target_h, target_w = target_size
        restored = np.full((target_h, target_w, c), fill_value, dtype=matrix.dtype)
        restored[1:-1, 1:-1, :] = matrix
    else:
        raise ValueError(f"不支持的矩阵维度：{matrix.shape}")
    return restored

def normalize_channels(tensor):
    """
    对张量的所有通道执行全局Min-Max归一化（0-1）
    :param tensor: 输入张量 (h, w, c)
    :return: 归一化后的张量
    """
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    if max_val - min_val == 0:
        return np.zeros_like(tensor, dtype=np.float32)
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)

# -------------------------- 单张图像处理函数 --------------------------
def process_single_image(img_path):
    """
    处理单张图像，返回最终拼接并归一化的512x512x64张量
    新增：HVD子带的8通道LBP编码（24通道） + 原有40通道 = 64通道
    """
    # 1. 读取并resize到1024x1024（灰度图）
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    img_float = np.float32(img_resized)

    # -------------------------- 步骤2：一级小波处理 --------------------------
    # 2.1 一级小波变换（得到4个子带：512x512）
    coeffs1 = pywt.dwt2(img_float, WAVELET)
    LL1, (LH1, HL1, HH1) = coeffs1
    A = np.stack([LL1, LH1, HL1, HH1], axis=-1)  # 512x512x4

    # 2.2 对4个子带编码 + 尺寸恢复（原有逻辑）
    B_list = []
    for i in range(4):
        subband = A[..., i]
        encode_mat = binary_encoding(subband)  # 510x510
        restored_mat = restore_size(encode_mat, (512, 512))  # 恢复为512x512
        B_list.append(restored_mat)
    B = np.stack(B_list, axis=-1).astype(np.float32)  # 512x512x4（转为浮点型方便相乘）

    # 2.3 A与B逐通道逐元素相乘得到C（原有逻辑）
    C = A * B  # 512x512x4

    # -------------------------- 新增步骤：HVD子带8通道LBP编码 --------------------------
    # 提取HVD子带（LH1、HL1、HH1）
    hvd_subbands = [LH1, HL1, HH1]
    hvd_8ch_list = []
    for subband in hvd_subbands:
        # 生成8通道LBP编码（510x510x8）
        lbp_8ch = lbp_to_8channels(subband)
        # 恢复尺寸为512x512x8（四周补0）
        restored_8ch = restore_size(lbp_8ch, (512, 512))
        hvd_8ch_list.append(restored_8ch)
    # 拼接3个子带的8通道张量 → 512x512x24
    HVD_24ch = np.concatenate(hvd_8ch_list, axis=-1).astype(np.float32)

    # -------------------------- 步骤3：二级小波处理（原有逻辑） --------------------------
    # 3.1 对一级4个子带分别做小波变换（得到16个子带：256x256）
    D_list = []
    for i in range(4):
        subband_level1 = A[..., i]
        coeffs2 = pywt.dwt2(subband_level1, WAVELET)
        LL2, (LH2, HL2, HH2) = coeffs2
        D_list.extend([LL2, LH2, HL2, HH2])
    D = np.stack(D_list, axis=-1)  # 256x256x16

    # 3.2 对16个子带编码 + 尺寸恢复
    E_list = []
    for i in range(16):
        subband = D[..., i]
        encode_mat = binary_encoding(subband)  # 254x254
        restored_mat = restore_size(encode_mat, (256, 256))  # 恢复为256x256
        E_list.append(restored_mat)
    E = np.stack(E_list, axis=-1).astype(np.float32)  # 256x256x16

    # 3.3 D与E逐通道逐元素相乘得到F
    F = D * E  # 256x256x16

    # -------------------------- 步骤4：上采样 + 拼接 + 归一化 --------------------------
    # 4.1 将E和F上采样到512x512（双线性插值）
    E_up = cv2.resize(E, (512, 512), interpolation=cv2.INTER_LINEAR)  # 512x512x16
    F_up = cv2.resize(F, (512, 512), interpolation=cv2.INTER_LINEAR)  # 512x512x16

    # 4.2 原有拼接（BCEF：4+4+16+16=40通道）
    original_40ch = np.concatenate([B, C, E_up, F_up], axis=-1)  # 512x512x40

    # 4.3 拼接原有40通道 + 新增24通道 → 64通道
    concatenated = np.concatenate([original_40ch, HVD_24ch], axis=-1)  # 512x512x64

    # 4.4 所有通道归一化（0-1）
    normalized_tensor = normalize_channels(concatenated)
    normalized_tensor = normalized_tensor.astype(np.float16)

    return normalized_tensor

# -------------------------- 批量处理数据集（单张保存） --------------------------
def process_dataset_and_save_compressed(dataset_path, save_root_path):
    os.makedirs(save_root_path, exist_ok=True)
    
    img_extensions = ['.jpg', '.jpeg', '.png', '.tif']
    img_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                 if os.path.splitext(f)[1].lower() in img_extensions]
    
    if len(img_paths) == 0:
        raise FileNotFoundError(f"在 {dataset_path} 中未找到任何图像文件")
    
    success_count = 0
    failed_files = []
    
    for img_path in tqdm(img_paths, desc="处理并压缩保存图像"):
        try:
            tensor = process_single_image(img_path)
            file_name = os.path.basename(img_path)
            file_prefix = os.path.splitext(file_name)[0]
            
            # 核心修改：用np.savez_compressed替代np.save，启用无损压缩
            save_path = os.path.join(save_root_path, f"{file_prefix}.npz")  # 后缀改为npz
            np.savez_compressed(save_path, tensor=tensor)  # 以key='tensor'存储
            
            success_count += 1
        except Exception as e:
            failed_files.append((img_path, str(e)))
            print(f"\n处理图像 {img_path} 时出错：{e}，跳过该图像")
    
    print(f"\n===== 处理完成 ======")
    print(f"总计扫描到 {len(img_paths)} 张图像")
    print(f"成功处理并保存 {success_count} 张图像")
    print(f"处理失败 {len(failed_files)} 张图像")
    
    if failed_files:
        print(f"\n失败文件列表：")
        for file_path, error in failed_files:
            print(f"  {file_path}: {error}")

# -------------------------- 读取压缩后的npz文件（训练时用） --------------------------
def load_compressed_tensor(npz_path):
    """读取压缩保存的npz文件，返回原始张量"""
    with np.load(npz_path) as data:
        tensor = data['tensor']  # 对应保存时的key='tensor'
    return tensor

# -------------------------- 主执行 --------------------------
if __name__ == "__main__":
    try:
        process_dataset_and_save_compressed(DATASET_PATH, SAVE_ROOT_PATH)
        print(f"\n所有压缩后的张量文件已保存到：{SAVE_ROOT_PATH}")
        
        # 测试读取（可选）
        sample_npz = os.listdir(SAVE_ROOT_PATH)[0]
        sample_tensor = load_compressed_tensor(os.path.join(SAVE_ROOT_PATH, sample_npz))
        print(f"示例张量形状：{sample_tensor.shape}")  # 仍为(512,512,64)
    except Exception as e:
        print(f"执行出错：{e}")