import os
import shutil
import random
from pathlib import Path

def prepare_joint_dataset():
    # 原始数据集路径
    coco_dir = '/root/autodl-tmp/CocoGlide'
    comofod_dir = '/root/autodl-tmp/CoMoFod/test'
    
    # 输出路径
    output_dir = '/root/autodl-tmp/joint'
    
    # 创建输出目录结构
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [
        os.path.join(train_dir, 'masks'),
        os.path.join(train_dir, 'tampered'),
        os.path.join(train_dir, 'train_tensor'),
        os.path.join(test_dir, 'masks'),
        os.path.join(test_dir, 'tampered'),
        os.path.join(test_dir, 'test_tensor')
    ]:
        os.makedirs(dir_path, exist_ok=True)
    
    def collect_samples(base_dir, tensor_dir, img_ext, mask_ext):
        tampered_dir = os.path.join(base_dir, 'tampered')
        mask_dir = os.path.join(base_dir, 'masks')
        
        samples = []
        for img_name in os.listdir(tampered_dir):
            if not img_name.endswith(img_ext):
                continue
            
            # 获取基础名称
            base_name = img_name[:-len(img_ext)]
            
            # 检查文件是否存在
            mask_path = os.path.join(mask_dir, f'{base_name}_mask{mask_ext}')
            tensor_path = os.path.join(tensor_dir, f'{base_name}.npz')
            
            if os.path.exists(mask_path) and os.path.exists(tensor_path):
                samples.append({
                    'img': os.path.join(tampered_dir, img_name),
                    'mask': mask_path,
                    'tensor': tensor_path,
                    'name': base_name
                })
        
        return samples
    
    # 收集两个数据集的样本
    print("收集 CocoGlide 数据集...")
    coco_tensor_dir = '/root/autodl-tmp/coco_tensor'
    coco_samples = collect_samples(coco_dir, coco_tensor_dir, '.png', '.png')
    print(f"  CocoGlide 样本数: {len(coco_samples)}")
    
    print("收集 CoMoFod 数据集...")
    comofod_tensor_dir = '/root/autodl-tmp/CoMoFod_tensor'
    comofod_samples = collect_samples(comofod_dir, comofod_tensor_dir, '.jpg', '.jpg')
    print(f"  CoMoFod 样本数: {len(comofod_samples)}")
    
    # 按80%/20%划分
    def split_samples(samples, split_ratio=0.8):
        random.seed(42)
        random.shuffle(samples)
        split_idx = int(len(samples) * split_ratio)
        return samples[:split_idx], samples[split_idx:]
    
    coco_train, coco_test = split_samples(coco_samples)
    comofod_train, comofod_test = split_samples(comofod_samples)
    
    # 合并数据集
    train_samples = coco_train + comofod_train
    test_samples = coco_test + comofod_test
    
    print(f"\n训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(test_samples)}")
    
    # 复制文件
    def copy_samples(samples, dest_masks, dest_tampered, dest_tensor):
        total = len(samples)
        for i, sample in enumerate(samples):
            # 复制图片
            shutil.copy(sample['img'], os.path.join(dest_tampered, f"{sample['name']}{os.path.splitext(sample['img'])[1]}"))
            
            # 复制掩码
            mask_ext = os.path.splitext(sample['mask'])[1]
            shutil.copy(sample['mask'], os.path.join(dest_masks, f"{sample['name']}_mask{mask_ext}"))
            
            # 复制张量
            shutil.copy(sample['tensor'], os.path.join(dest_tensor, f"{sample['name']}.npz"))
            
            # 进度条
            if (i + 1) % 50 == 0:
                progress = (i + 1) / total * 100
                print(f"  复制进度: {progress:.1f}%")
    
    print("\n复制训练集文件...")
    copy_samples(train_samples,
                 os.path.join(train_dir, 'masks'),
                 os.path.join(train_dir, 'tampered'),
                 os.path.join(train_dir, 'train_tensor'))
    
    print("\n复制测试集文件...")
    copy_samples(test_samples,
                 os.path.join(test_dir, 'masks'),
                 os.path.join(test_dir, 'tampered'),
                 os.path.join(test_dir, 'test_tensor'))
    
    print("\n联合数据集准备完成！")
    print(f"训练集: {len(train_samples)} 个样本")
    print(f"测试集: {len(test_samples)} 个样本")

if __name__ == '__main__':
    prepare_joint_dataset()