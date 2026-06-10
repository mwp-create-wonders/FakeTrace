import os
import re
import pandas as pd
import albumentations
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

# 配置日志记录器
logging.basicConfig(
    filename='dataset_errors.log',          # 日志文件名
    level=logging.ERROR,                    # 记录 ERROR 及以上级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'                             # 追加模式
)

def crop_center_by_percentage(image, percentage):
    height, width = image.shape[:2]
    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        cropped_image = image[:, start_x:end_x]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        cropped_image = image[start_y:end_y, :]
    return cropped_image


def get_number_from_filename(filename):
    match = re.match(r'(\d+)', filename)  
    if match:
        return int(match.group(1))  
    return float('inf')  


def read_video(folder_path, trans):
    frames = []
    image_paths = sorted(os.listdir(folder_path), key=get_number_from_filename)
    total_frames = len(image_paths)

    if total_frames < 8:
        raise ValueError(f"No enough frames found in {folder_path}.")

    set_frame = 8 if total_frames < 16 else 16
    max_frame = min(set_frame, total_frames)
    for i in range(max_frame):
        image_path = os.path.join(folder_path, image_paths[i])
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ 图像读取失败: {image_path}")
            continue  # 跳过损坏的图像
        image = crop_center_by_percentage(image, 0.1)
        augmented = trans(image=image)
        image = augmented["image"]
        frames.append(image.transpose(2, 0, 1)[np.newaxis, :])

    frames = np.concatenate(frames, 0)
    frames = torch.tensor(frames[np.newaxis, :]).squeeze(0)
    return frames


def set_preprocessing(aug_type, aug_quality):
    aug_list = []
    aug_list.append(albumentations.Resize(224, 224))
    if aug_type == 'Gaussian_blur':
        aug_list.append(albumentations.GaussianBlur(blur_limit=(3, 7), sigma_limit=(aug_quality, aug_quality), p=1.0)) 
    if aug_type == 'JEPG_compression':
        aug_list.append(albumentations.ImageCompression(quality_lower=aug_quality, quality_upper=aug_quality))
    aug_list.append(albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0))
    return albumentations.Compose(aug_list)


class D3_dataset_AP(Dataset):

    def __init__(self, real_csv, fake_csv, max_len=9999999, aug_type=None, aug_quality=None):
        super(D3_dataset_AP, self).__init__()
        df_real = pd.read_csv(real_csv).head(max_len)
        df_fake = pd.read_csv(fake_csv).head(max_len)
        self.df = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        self.trans = set_preprocessing(aug_type, aug_quality)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 最多尝试5个样本
        for i in range(5):
            idx = (index + i) % len(self.df)
            
            try:
                label = self.df.loc[idx]['label']
                frame_path = self.df.loc[idx]['content_path']
                frames = read_video(frame_path, trans=self.trans)
                
                # 关键：确保有8帧
                if frames.shape[0] >= 8:
                    return frames[:8], label  # 只取前8帧
                else:
                    # 帧数不足，尝试下一个
                    msg = f"帧数不足: {frame_path} (实际帧数 {frames.shape[0]})"
                    print(msg)
                    logging.error(msg)  # 记录到日志文件
                    continue
                    
            except Exception as e:
                # 记录详细异常信息（包括堆栈）
                frame_path = self.df.loc[idx]['content_path'] if 'idx' in locals() else 'unknown'
                logging.exception(f"处理样本时发生异常 | index={index} | 尝试idx={idx} | path={frame_path}")
                print(f"异常: {e}")  # 控制台简单提示
                continue
        
        # 都失败了，返回安全的占位符
        fallback_msg = f"所有尝试失败，返回占位符 | 原始index={index}"
        print(fallback_msg)
        logging.error(fallback_msg)
        return torch.zeros((8, 3, 224, 224)), 0