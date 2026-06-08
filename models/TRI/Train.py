import os
import sys
import argparse
import tempfile
import glob
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import time
import json
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A

# ==================== 工具函数 ====================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_lota_model(model_path, device='cuda'):
    """加载 LOTA 模型"""
    lota_path = os.path.join(os.path.dirname(__file__), "LOTA")
    if lota_path not in sys.path:
        sys.path.insert(0, lota_path)
    from test import DeepLearningModel
    
    lota_model = DeepLearningModel()
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        lota_model.load_state_dict(state_dict)
    lota_model = lota_model.to(device)
    lota_model.eval()
    for param in lota_model.parameters():
        param.requires_grad = False
    return lota_model

def prepare_csv_path(path):
    if os.path.isdir(path):
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise ValueError(f"目录 {path} 中没有 CSV 文件")
        print(f"合并目录 {path} 中的 {len(csv_files)} 个 CSV 文件...")
        dfs = [pd.read_csv(f) for f in csv_files]
        merged_df = pd.concat(dfs, ignore_index=True)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        merged_df.to_csv(temp_file.name, index=False)
        print(f"临时合并文件: {temp_file.name}")
        return temp_file.name
    else:
        return path


# ==================== 特征预计算器 ====================
class FeaturePrecomputedCache:
    def __init__(self, cache_dir='feature_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'precomputed_features.pkl')
    
    def exists(self):
        return os.path.exists(self.cache_file)
    
    def save(self, features_dict):
        print(f"\n保存预计算特征到: {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"保存完成，共 {len(features_dict)} 个视频")
    
    def load(self):
        print(f"\n从缓存加载预计算特征: {self.cache_file}")
        with open(self.cache_file, 'rb') as f:
            features_dict = pickle.load(f)
        print(f"加载完成，共 {len(features_dict)} 个视频")
        return features_dict


class FeaturePrecomputer:
    def __init__(self, d3_model, lota_model, device, target_len=8,
                 enable_image_aug=False, image_aug_prob=0.5):
        self.d3_model = d3_model
        self.lota_model = lota_model
        self.device = device
        self.target_len = target_len
        self.enable_image_aug = enable_image_aug
        self.image_aug_prob = image_aug_prob
        
        if enable_image_aug:
            self.image_aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=image_aug_prob),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=image_aug_prob),
                A.GaussNoise(var_limit=(10.0, 50.0), p=image_aug_prob * 0.6),
                A.GaussianBlur(blur_limit=(3, 7), p=image_aug_prob * 0.6),
            ])
        
        self.d3_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def _augment_image(self, img):
        if not self.enable_image_aug or not self.image_aug:
            return img
        img_np = np.array(img)
        augmented = self.image_aug(image=img_np)
        return Image.fromarray(augmented['image'])
    
    def extract_features_for_video(self, video_path):
        """提取 6×3 标量特征：速度标量 + 加速度标量 + LOTA"""
        frame_dir = video_path
        if not os.path.exists(frame_dir):
            return None
        
        frame_files = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if len(frame_files) < self.target_len:
            return None
        
        frame_files = frame_files[:self.target_len]
        
        frames_tensor = []
        for f in frame_files:
            img = Image.open(os.path.join(frame_dir, f)).convert('RGB')
            img = self._augment_image(img)
            frames_tensor.append(self.d3_transform(img))
        
        frames_tensor = torch.stack(frames_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.d3_model(frames_tensor, return_vectors=True)
            frame_features = output['frame_features']

        frame_features = frame_features[0].cpu().numpy()  # [8, 768]
        
        # 速度（一阶差分）
        velocity = frame_features[1:] - frame_features[:-1]  # [7, 768]
        
        # 加速度（二阶差分）
        acceleration = velocity[1:] - velocity[:-1]  # [6, 768]
        
        align_len = self.target_len - 2  # 6
        
        # 速度标量（L2范数）
        velocity = velocity[:align_len]  # [6, 768]
        velocity_norm = np.linalg.norm(velocity, axis=1)  # [6]
        
        # 加速度标量（L2范数）
        acceleration = acceleration[:align_len]  # [6, 768]
        acceleration_norm = np.linalg.norm(acceleration, axis=1)  # [6]
        
        # LOTA 特征
        batch_tensor = frames_tensor.squeeze(0)
        with torch.no_grad():
            logits = self.lota_model(batch_tensor)
            lota_scores = torch.sigmoid(logits).cpu().numpy().flatten()
        lota_scores = lota_scores[:align_len]  # [6]
        
        return {
            'velocity': velocity_norm,      # (6,)
            'acceleration': acceleration_norm,  # (6,)
            'lota': lota_scores             # (6,)
        }
    
    def precompute_all(self, df, max_videos=None):
        """预计算所有视频的特征"""
        features_dict = {}
        if max_videos and len(df) > max_videos:
            df = df.head(max_videos)
        
        print(f"\n开始预计算特征，共 {len(df)} 个视频...")
        print(f"图像增强: {'启用' if self.enable_image_aug else '禁用'}")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="预计算特征"):
            video_path = row['content_path']
            label = row['label']
            
            if not os.path.exists(video_path):
                continue
            
            features = self.extract_features_for_video(video_path)
            if features is not None:
                features_dict[video_path] = {
                    'velocity': features['velocity'],
                    'acceleration': features['acceleration'],
                    'lota': features['lota'],
                    'label': label
                }
        
        print(f"预计算完成，有效视频数: {len(features_dict)}")
        return features_dict


# ==================== 帧级数据集（6×3）====================
class CachedFrameLevelDataset(Dataset):
    def __init__(self, features_dict, video_list, name="dataset",
                 use_velocity=True, use_acceleration=True, use_lota=True,
                 lota_dropout_rate=0.3, lota_noise_std=0.05, lota_weight=0.5,
                 enable_temporal_aug=False, time_mask_ratio=0.2, 
                 time_shift_max=1, feature_dropout=0.1):
        
        self.name = name
        self.video_list = video_list
        self.features_dict = features_dict
        
        self.use_velocity = use_velocity
        self.use_acceleration = use_acceleration
        self.use_lota = use_lota
        
        self.lota_dropout_rate = lota_dropout_rate
        self.lota_noise_std = lota_noise_std
        self.lota_weight = lota_weight
        self.is_training = 'TRAIN' in name
        
        self.enable_temporal_aug = enable_temporal_aug and self.is_training
        self.time_mask_ratio = time_mask_ratio
        self.time_shift_max = time_shift_max
        self.feature_dropout = feature_dropout
        
        # 6×3：速度1 + 加速度1 + LOTA1 = 3
        self.feature_dim = 0
        if use_velocity:
            self.feature_dim += 1
        if use_acceleration:
            self.feature_dim += 1
        if use_lota:
            self.feature_dim += 1
        
        self.data = []
        
        print(f"\n[数据集 {name}] 构建中...")
        print(f"  特征配置: 速度={use_velocity}(1), 加速度={use_acceleration}(1), LOTA={use_lota}(1)")
        print(f"  特征维度: {self.feature_dim}")
        if self.enable_temporal_aug:
            print(f"  时序增强: mask={time_mask_ratio}, shift={time_shift_max}, dropout={feature_dropout}")
        
        for video_path in tqdm(video_list, desc=f"构建 {name}"):
            feat_dict = features_dict.get(video_path)
            if feat_dict is None:
                continue
            
            seq_len = len(feat_dict['velocity'])  # 6
            
            sequence = []
            for t in range(seq_len):
                feat = []
                
                if use_velocity:
                    feat.append(feat_dict['velocity'][t])
                
                if use_acceleration:
                    feat.append(feat_dict['acceleration'][t])
                
                if use_lota:
                    lota_val = feat_dict['lota'][t]
                    if self.is_training:
                        if random.random() < self.lota_dropout_rate:
                            lota_val = 0.5
                        lota_val += random.gauss(0, self.lota_noise_std)
                        lota_val = max(0.0, min(1.0, lota_val))
                        lota_val = lota_val * self.lota_weight + (1 - self.lota_weight) * 0.5
                    feat.append(lota_val)
                
                sequence.append(feat)
            
            sequence = np.array(sequence, dtype=np.float32)
            self.data.append((sequence, feat_dict['label']))
        
        print(f"  有效样本数: {len(self.data)}")
    
    def _apply_temporal_augmentation(self, seq):
        seq_len, feat_dim = seq.shape
        seq = seq.copy()
        
        if self.time_mask_ratio > 0:
            mask_len = int(seq_len * self.time_mask_ratio)
            if mask_len > 0:
                mask_start = random.randint(0, seq_len - mask_len)
                seq[mask_start:mask_start+mask_len, :] = 0
        
        if self.time_shift_max > 0:
            shift = random.randint(-self.time_shift_max, self.time_shift_max)
            if shift != 0:
                seq = np.roll(seq, shift, axis=0)
                if shift > 0:
                    seq[:shift, :] = 0
                else:
                    seq[shift:, :] = 0
        
        if self.feature_dropout > 0:
            mask = np.random.rand(seq_len, feat_dim) > self.feature_dropout
            seq = seq * mask
        
        return seq
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        
        if self.enable_temporal_aug:
            seq = self._apply_temporal_augmentation(seq)
        
        return {'sequence': torch.FloatTensor(seq), 'label': torch.LongTensor([label])[0]}


# ==================== 时序融合网络（6×3输入）====================
class TemporalFusionNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, kernel_size=3, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).squeeze(-1)
        logits = self.classifier(x).squeeze(-1)
        return logits


# ==================== 消融实验管理器 ====================
class AblationExperiment:
    def __init__(self, base_dir='ablation_results'):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'logs'), exist_ok=True)
        self.results = []
        print(f"消融实验结果将保存至: {self.exp_dir}")
    
    def get_config_name(self, config):
        parts = []
        if config.get('use_velocity', True):
            parts.append('vel')
        if config.get('use_acceleration', True):
            parts.append('acc')
        if config.get('use_lota', True):
            parts.append('lota')
        return '_'.join(parts) if parts else 'none'
    
    def train_epoch(self, model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for batch in loader:
            seq = batch['sequence'].to(device)
            labels = batch['label'].float().to(device)
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def evaluate_with_probs(self, model, loader, criterion, device):
        model.eval()
        total_loss = 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                seq = batch['sequence'].to(device)
                labels = batch['label'].float().to(device)
                logits = model(seq)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        metrics = {
            'loss': total_loss / len(loader),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) == 2 else 0.5,
            'ap': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) == 2 else 0.5
        }
        return metrics, all_labels, all_probs
    
    def find_best_threshold(self, labels, probs, num_thresholds=100):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.9, num_thresholds):
            preds = (probs > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh, best_f1
    
    def train_model(self, model, train_loader, val_loader, criterion, optimizer, 
                    scheduler, device, epochs, patience, save_path):
        best_val_f1 = 0
        best_threshold = 0.5
        best_epoch = 0
        counter = 0
        best_state_dict = None
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, device)
            
            val_metrics, val_labels, val_probs = self.evaluate_with_probs(
                model, val_loader, criterion, device
            )
            
            best_th, best_f1 = self.find_best_threshold(val_labels, val_probs)
            preds = (val_probs > best_th).astype(int)
            tn, fp, fn, tp = confusion_matrix(val_labels, preds).ravel()
            
            print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val AUC: {val_metrics['auc']:.4f} | Best Thresh: {best_th:.4f} | F1: {best_f1:.4f}")
            print(f"   Confusion Matrix: TP={tp} TN={tn} FP={fp} FN={fn}")
            
            if scheduler:
                scheduler.step(best_f1)
            
            if best_f1 > best_val_f1:
                best_val_f1 = best_f1
                best_threshold = best_th
                best_epoch = epoch
                counter = 0
                best_state_dict = model.state_dict().copy()
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'best_threshold': best_threshold,
                    'best_val_f1': best_val_f1
                }
                torch.save(checkpoint, save_path)
                print(f"  >> 保存最佳模型 (F1={best_val_f1:.4f}, threshold={best_threshold:.4f})")
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停于 epoch {epoch}")
                    break
            print("-" * 60)
        
        return best_state_dict, best_val_f1, best_threshold, best_epoch
    
    def run_experiment(self, config, train_loader, val_loader, test_loader,
                       model_class, device, args):
        config_name = self.get_config_name(config)
        print(f"\n{'='*70}")
        print(f"运行消融实验: {config_name}")
        print(f"配置: 速度={config['use_velocity']}, "
              f"加速度={config['use_acceleration']}, "
              f"LOTA={config['use_lota']}")
        print(f"{'='*70}")
        
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['sequence'].shape[-1]
        
        model = model_class(
            input_dim=input_dim,
            hidden_dim=64,
            kernel_size=3,
            dropout=0.3
        ).to(device)
        
        model_path = os.path.join(self.exp_dir, 'models', f'{config_name}.pth')
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_state_dict, best_f1, best_threshold, best_epoch = self.train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, args.epochs, args.patience, model_path
        )
        
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        
        test_metrics, test_labels, test_probs = self.evaluate_with_probs(
            model, test_loader, criterion, device
        )
        
        test_preds = (test_probs > best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
        
        def to_python_type(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_python_type(v) for v in obj]
            return obj
        
        result = {
            'config_name': config_name,
            'config': {
                'use_velocity': bool(config['use_velocity']),
                'use_acceleration': bool(config['use_acceleration']),
                'use_lota': bool(config['use_lota'])
            },
            'feature_dim': int(input_dim),
            'best_val_f1': float(best_f1),
            'best_threshold': float(best_threshold),
            'best_epoch': int(best_epoch),
            'test': {
                'loss': float(test_metrics['loss']),
                'auc': float(test_metrics['auc']),
                'ap': float(test_metrics['ap']),
                'accuracy': float((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0),
                'precision': float(tp / (tp + fp) if tp + fp > 0 else 0),
                'recall': float(tp / (tp + fn) if tp + fn > 0 else 0),
                'f1': float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
                'confusion_matrix': {
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                }
            }
        }
        
        self.results.append(result)
        
        log_path = os.path.join(self.exp_dir, 'logs', f'{config_name}.json')
        with open(log_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{config_name} 测试结果:")
        print(f"  AUC: {test_metrics['auc']:.4f}, AP: {test_metrics['ap']:.4f}, F1: {result['test']['f1']:.4f}")
        print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"  模型已保存: {model_path}")
        
        return result
    
    def run_all_experiments(self, configs, train_loaders, val_loaders, test_loaders,
                            model_class, device, args):
        for config in configs:
            config_name = config['name']
            
            if config_name not in train_loaders:
                print(f"警告: 未找到 {config_name} 的数据加载器，跳过")
                continue
            
            sample = next(iter(train_loaders[config_name]))
            actual_dim = sample['sequence'].shape[-1]
            print(f"🔍 {config_name} 使用的 loader 维度: {actual_dim}")
            
            self.run_experiment(
                config,
                train_loaders[config_name],
                val_loaders[config_name],
                test_loaders[config_name],
                model_class,
                device,
                args
            )
        
        self.generate_summary()
        return self.results
    
    def generate_summary(self):
        if not self.results:
            return
        df = pd.DataFrame([{
            'config': r['config_name'],
            'feature_dim': r['feature_dim'],
            'best_val_f1': r['best_val_f1'],
            'test_auc': r['test']['auc'],
            'test_f1': r['test']['f1'],
            'TP': r['test']['confusion_matrix']['tp'],
            'TN': r['test']['confusion_matrix']['tn'],
            'FP': r['test']['confusion_matrix']['fp'],
            'FN': r['test']['confusion_matrix']['fn']
        } for r in self.results])
        df = df.sort_values('test_f1', ascending=False)
        df.to_csv(os.path.join(self.exp_dir, 'summary.csv'), index=False)
        
        print("\n" + "="*80)
        print("消融实验汇总")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


# ==================== 消融实验配置 ====================
def get_ablation_configs():
    """获取消融实验配置"""
    configs = [
        {'name': 'full', 'use_velocity': True, 'use_acceleration': True, 'use_lota': True},
        {'name': 'vel_lota', 'use_velocity': True, 'use_acceleration': False, 'use_lota': True},
    ]
    return configs


# ==================== 创建数据加载器 ====================
def create_data_loaders_from_cache(features_dict, train_videos, val_videos, test_videos,
                                    configs, args):
    from torch.utils.data import DataLoader
    
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    
    for config in configs:
        config_name = config['name']
        
        lota_dropout_rate = getattr(args, 'lota_dropout', 0.3)
        lota_noise_std = getattr(args, 'lota_noise', 0.05)
        lota_weight = getattr(args, 'lota_weight', 0.5)
        
        train_set = CachedFrameLevelDataset(
            features_dict, train_videos, name=f"TRAIN_{config_name}",
            use_velocity=config['use_velocity'],
            use_acceleration=config['use_acceleration'],
            use_lota=config['use_lota'],
            lota_dropout_rate=lota_dropout_rate,
            lota_noise_std=lota_noise_std,
            lota_weight=lota_weight,
            enable_temporal_aug=getattr(args, 'enable_temporal_aug', True),
            time_mask_ratio=getattr(args, 'time_mask_ratio', 0.2),
            time_shift_max=getattr(args, 'time_shift_max', 1),
            feature_dropout=getattr(args, 'feature_dropout', 0.1)
        )
        
        val_set = CachedFrameLevelDataset(
            features_dict, val_videos, name=f"VAL_{config_name}",
            use_velocity=config['use_velocity'],
            use_acceleration=config['use_acceleration'],
            use_lota=config['use_lota'],
            lota_dropout_rate=0.0,
            lota_noise_std=0.0,
            lota_weight=1.0,
            enable_temporal_aug=False
        )
        
        test_set = CachedFrameLevelDataset(
            features_dict, test_videos, name=f"TEST_{config_name}",
            use_velocity=config['use_velocity'],
            use_acceleration=config['use_acceleration'],
            use_lota=config['use_lota'],
            lota_dropout_rate=0.0,
            lota_noise_std=0.0,
            lota_weight=1.0,
            enable_temporal_aug=False
        )
        
        train_loaders[config_name] = DataLoader(
            train_set, args.batch_size, shuffle=True, num_workers=4, 
            pin_memory=(args.device == 'cuda')
        )
        val_loaders[config_name] = DataLoader(
            val_set, args.batch_size, shuffle=False, num_workers=4, 
            pin_memory=(args.device == 'cuda')
        )
        test_loaders[config_name] = DataLoader(
            test_set, args.batch_size, shuffle=False, num_workers=4, 
            pin_memory=(args.device == 'cuda')
        )
        
        print(f"配置 {config_name}: 训练集={len(train_set)}, 验证集={len(val_set)}, 测试集={len(test_set)}")
    
    return train_loaders, val_loaders, test_loaders


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', default=None)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--val-ratio', type=float, default=0.3)
    parser.add_argument('--lota-model', required=True)
    parser.add_argument('--encoder', default='XCLIP-16')
    parser.add_argument('--loss-type', default='l2', choices=['l2', 'cos'])
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target-len', type=int, default=8)
    parser.add_argument('--max-videos', type=int, default=30000)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save-dir', default='ablation_results')
    parser.add_argument('--balance', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存，强制重新计算')
    
    # 图像增强参数
    parser.add_argument('--enable_image_aug', action='store_true', default=True)
    parser.add_argument('--image_aug_prob', type=float, default=0.5)
    
    # 时序增强参数
    parser.add_argument('--enable_temporal_aug', action='store_true', default=True)
    parser.add_argument('--no_temporal_aug', dest='enable_temporal_aug', action='store_false')
    parser.add_argument('--time_mask_ratio', type=float, default=0.2)
    parser.add_argument('--time_shift_max', type=int, default=1)
    parser.add_argument('--feature_dropout', type=float, default=0.1)
    
    # LOTA 正则化参数
    parser.add_argument('--lota_dropout', type=float, default=0.3)
    parser.add_argument('--lota_noise', type=float, default=0.05)
    parser.add_argument('--lota_weight', type=float, default=0.5)
    
    args = parser.parse_args()

    seed_everything(args.seed)
    
    if args.device == 'cuda' and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = torch.device('cuda')
        print(f"使用设备: CUDA (GPU {args.gpu_id})")
    else:
        device = torch.device('cpu')
        print("使用设备: CPU")
    
    print(f"图像增强: {'启用' if args.enable_image_aug else '禁用'} (概率={args.image_aug_prob})")
    print(f"时序增强: {'启用' if args.enable_temporal_aug else '禁用'}")

    from models import D3_model
    d3_model = D3_model(encoder_type=args.encoder, loss_type=args.loss_type).to(device)
    d3_model.eval()
    for param in d3_model.parameters():
        param.requires_grad = False

    lota_model = load_lota_model(args.lota_model, device)
    if lota_model is None:
        print("LOTA 模型加载失败，退出。")
        return

    train_csv_path = prepare_csv_path(args.train_csv)
    
    if args.val_csv is None:
        print(f"\n未提供验证集，将从训练集中划分 {args.val_ratio*100:.0f}% 作为验证集")
        df_all = pd.read_csv(train_csv_path)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df_all, test_size=args.val_ratio, 
            random_state=args.seed, stratify=df_all['label']
        )
        temp_dir = tempfile.mkdtemp(prefix="split_")
        train_temp = os.path.join(temp_dir, "train.csv")
        val_temp = os.path.join(temp_dir, "val.csv")
        train_df.to_csv(train_temp, index=False)
        val_df.to_csv(val_temp, index=False)
        train_csv_path = train_temp
        val_csv_path = val_temp
        self_created_val = True
        print(f"划分完成: 训练集 {len(train_df)} 样本, 验证集 {len(val_df)} 样本")
    else:
        val_csv_path = prepare_csv_path(args.val_csv)
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        self_created_val = False

    test_csv_path = prepare_csv_path(args.test_csv)
    test_df = pd.read_csv(test_csv_path)
    
    print(f"\n原始数据统计:")
    print(f"  训练集: {len(train_df)} 个视频")
    print(f"  验证集: {len(val_df)} 个视频")
    print(f"  测试集: {len(test_df)} 个视频")
    
    if args.max_videos and args.max_videos > 0:
        if len(train_df) > args.max_videos:
            train_df = train_df.sample(n=args.max_videos, random_state=args.seed)
            print(f"训练集采样: {len(train_df)} 个视频")
        if len(val_df) > args.max_videos:
            val_df = val_df.sample(n=args.max_videos, random_state=args.seed)
            print(f"验证集采样: {len(val_df)} 个视频")
        if len(test_df) > args.max_videos:
            test_df = test_df.sample(n=args.max_videos, random_state=args.seed)
            print(f"测试集采样: {len(test_df)} 个视频")
    
    print(f"\n采样后数据统计:")
    print(f"  训练集: {len(train_df)} 个视频")
    print(f"  验证集: {len(val_df)} 个视频")
    print(f"  测试集: {len(test_df)} 个视频")
    
    if len(train_df) == 0:
        print("错误: 训练集为空")
        return
    
    all_df = pd.concat([train_df, val_df, test_df])
    all_videos = all_df['content_path'].tolist()
    total_unique = len(set(all_videos))
    print(f"\n待预计算视频总数: {len(all_df)} (去重后: {total_unique})")
    
    cache = FeaturePrecomputedCache(cache_dir='feature_cache')
    features_dict = None
    if not args.no_cache and cache.exists():
        features_dict = cache.load()
        missing_videos = [v for v in all_videos if v not in features_dict]
        
        if missing_videos:
            print(f"缓存缺失 {len(missing_videos)} 个视频")
            
            if len(missing_videos) < 100:
                print(f"缺失视频较少，直接跳过这些视频，继续使用缓存")
                missing_set = set(missing_videos)
                train_df = train_df[~train_df['content_path'].isin(missing_set)]
                val_df = val_df[~val_df['content_path'].isin(missing_set)]
                test_df = test_df[~test_df['content_path'].isin(missing_set)]
                all_df = pd.concat([train_df, val_df, test_df])
                print(f"移除缺失视频后: 训练集 {len(train_df)}, 验证集 {len(val_df)}, 测试集 {len(test_df)}")
            else:
                print(f"缺失视频较多，需要重新计算特征")
                features_dict = None
    
    if features_dict is None:
        precomputer = FeaturePrecomputer(
            d3_model, lota_model, device, args.target_len,
            enable_image_aug=args.enable_image_aug,
            image_aug_prob=args.image_aug_prob
        )
        features_dict = precomputer.precompute_all(all_df, max_videos=None)
        cache.save(features_dict)
    
    train_videos = [v for v in train_df['content_path'].tolist() if v in features_dict]
    val_videos = [v for v in val_df['content_path'].tolist() if v in features_dict]
    test_videos = [v for v in test_df['content_path'].tolist() if v in features_dict]
    
    print(f"\n特征提取后有效数据统计:")
    print(f"  训练集: {len(train_videos)} 个视频")
    print(f"  验证集: {len(val_videos)} 个视频")
    print(f"  测试集: {len(test_videos)} 个视频")
    
    if len(val_videos) == 0 and len(train_videos) > 0:
        print("警告: 验证集为空，将使用训练集的一部分作为验证集")
        from sklearn.model_selection import train_test_split
        train_videos, val_videos = train_test_split(
            train_videos, test_size=0.2, random_state=args.seed
        )
        print(f"  重新划分后: 训练集 {len(train_videos)}, 验证集 {len(val_videos)}")
    
    if len(test_videos) == 0 and len(train_videos) > 0:
        print("警告: 测试集为空，将使用训练集的一部分作为测试集")
        from sklearn.model_selection import train_test_split
        train_videos, test_videos = train_test_split(
            train_videos, test_size=0.1, random_state=args.seed
        )
        print(f"  重新划分后: 训练集 {len(train_videos)}, 测试集 {len(test_videos)}")
    
    if len(train_videos) == 0:
        print("错误: 没有有效的训练数据")
        return
    
    configs = get_ablation_configs()
    
    train_loaders, val_loaders, test_loaders = create_data_loaders_from_cache(
        features_dict, train_videos, val_videos, test_videos, configs, args
    )
    
    experiment = AblationExperiment(base_dir=args.save_dir)
    experiment.run_all_experiments(
        configs, train_loaders, val_loaders, test_loaders,
        TemporalFusionNet, device, args
    )
    
    if self_created_val and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()