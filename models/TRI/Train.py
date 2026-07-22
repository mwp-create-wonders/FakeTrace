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
    lota_path = os.path.join(os.path.dirname(__file__), "LOTA")
    if lota_path not in sys.path:
        sys.path.insert(0, lota_path)
    from test import DeepLearningModel

    lota_model = DeepLearningModel()
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
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
        return temp_file.name
    else:
        return path


# ==================== 特征预计算器 ====================
class FeaturePrecomputedCache:
    def __init__(self, cache_dir='feature_cache_768'):
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

        frame_features = frame_features[0].cpu().numpy()

        batch_tensor = frames_tensor.squeeze(0)
        with torch.no_grad():
            logits = self.lota_model(batch_tensor)
            lota_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        return {
            'frame_features': frame_features,
            'lota': lota_scores,
            'label': None
        }

    def precompute_all(self, df, max_videos=None):
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
                    'frame_features': features['frame_features'],
                    'lota': features['lota'],
                    'label': label
                }

        print(f"预计算完成，有效视频数: {len(features_dict)}")
        return features_dict


# ==================== 数据集 ====================
class CachedFrameLevelDataset(Dataset):
    def __init__(self, features_dict, video_list, name="dataset",
                 use_lota=True,
                 lota_dropout_rate=0.3, lota_noise_std=0.05, lota_weight=0.5,
                 enable_temporal_aug=False, time_mask_ratio=0.2,
                 time_shift_max=1, feature_dropout=0.1):

        self.name = name
        self.video_list = video_list
        self.features_dict = features_dict

        self.use_lota = use_lota
        self.lota_dropout_rate = lota_dropout_rate
        self.lota_noise_std = lota_noise_std
        self.lota_weight = lota_weight
        self.is_training = 'TRAIN' in name

        self.enable_temporal_aug = enable_temporal_aug and self.is_training
        self.time_mask_ratio = time_mask_ratio
        self.time_shift_max = time_shift_max
        self.feature_dropout = feature_dropout

        self.data = []

        print(f"\n[数据集 {name}] 构建中...")
        print(f"  特征: 768维帧特征 + LOTA(1)")
        if self.enable_temporal_aug:
            print(f"  时序增强: mask={time_mask_ratio}, shift={time_shift_max}, dropout={feature_dropout}")

        for video_path in tqdm(video_list, desc=f"构建 {name}"):
            feat_dict = features_dict.get(video_path)
            if feat_dict is None:
                continue

            frame_features = feat_dict['frame_features']
            lota_scores = feat_dict['lota']
            label = feat_dict['label']

            self.data.append((frame_features, lota_scores, label))

        print(f"  有效样本数: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_features, lota_scores, label = self.data[idx]

        velocity = frame_features[1:] - frame_features[:-1]
        acceleration = velocity[1:] - velocity[:-1]

        align_len = 6
        velocity = velocity[:align_len]
        acceleration = acceleration[:align_len]
        lota = lota_scores[:align_len]

        return {
            'velocity': torch.FloatTensor(velocity),
            'acceleration': torch.FloatTensor(acceleration),
            'lota': torch.FloatTensor(lota),
            'label': torch.LongTensor([label])[0]
        }


# ==================== Adapter（速度和加速度独立残差）====================
class FeatureAdapter1D(nn.Module):
    """
    L2 + 残差，速度和加速度各自独立的残差网络
    """

    def __init__(self, feature_dim=768, hidden_dim=8):
        super().__init__()
        # 速度专用残差网络
        self.velocity_res = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 加速度专用残差网络
        self.acceleration_res = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._zero_initialize()

    def _zero_initialize(self):
        for proj in [self.velocity_res, self.acceleration_res]:
            nn.init.normal_(proj[0].weight, std=0.01)
            nn.init.zeros_(proj[0].bias)
            nn.init.zeros_(proj[2].weight)
            nn.init.zeros_(proj[2].bias)

    def forward(self, velocity, acceleration):
        # L2 范数（固定）
        v_l2 = torch.norm(velocity, dim=-1, keepdim=True)
        a_l2 = torch.norm(acceleration, dim=-1, keepdim=True)

        # 各自独立的残差
        v_res = self.velocity_res(velocity)
        a_res = self.acceleration_res(acceleration)

        # 输出 = L2 + 残差
        v_out = v_l2 + v_res
        a_out = a_l2 + a_res

        return v_out, a_out


# ==================== 完整模型 ====================
class AdapterModel1D(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=8, kernel_size=3, dropout=0.3):
        super().__init__()
        self.adapter = FeatureAdapter1D(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def load_pretrained_cnn(self, ckpt_path):
        print(f"\n加载预训练CNN权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in ckpt:
            pretrained = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            pretrained = ckpt['state_dict']
        else:
            pretrained = ckpt

        own_state = self.state_dict()
        loaded_count = 0

        for name, param in pretrained.items():
            if 'adapter' in name:
                continue
            clean_name = name.replace('module.', '')
            if clean_name in own_state and param.shape == own_state[clean_name].shape:
                own_state[clean_name].copy_(param)
                loaded_count += 1
                print(f"  ✓ 加载 {clean_name}")

        print(f"加载完成: {loaded_count} 层")
        return loaded_count

    def freeze_cnn(self):
        """冻结 CNN，只训练 Adapter"""
        for name, param in self.named_parameters():
            param.requires_grad = 'adapter' in name
        print("✅ CNN 冻结，只训练 Adapter")

    def freeze_adapter(self):
        """冻结 Adapter，只训练 CNN"""
        for name, param in self.named_parameters():
            if 'adapter' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("✅ Adapter 冻结，只训练 CNN")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("✅ 解冻所有参数")

    def forward(self, velocity, acceleration, lota):
        v_feat, a_feat = self.adapter(velocity, acceleration)
        features = torch.cat([v_feat, a_feat, lota.unsqueeze(-1)], dim=-1)

        x = features.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).squeeze(-1)
        logits = self.classifier(x).squeeze(-1)

        return logits, features


# ==================== 实验管理器 ====================
class AblationExperiment:
    def __init__(self, base_dir='ablation_results_l2_residual'):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'logs'), exist_ok=True)
        self.results = []
        print(f"实验结果将保存至: {self.exp_dir}")

    def train_epoch(self, model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for batch in loader:
            velocity = batch['velocity'].to(device)
            acceleration = batch['acceleration'].to(device)
            lota = batch['lota'].to(device)
            labels = batch['label'].float().to(device)

            optimizer.zero_grad()
            logits, _ = model(velocity, acceleration, lota)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, model, loader, criterion, device):
        model.eval()
        total_loss = 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                velocity = batch['velocity'].to(device)
                acceleration = batch['acceleration'].to(device)
                lota = batch['lota'].to(device)
                labels = batch['label'].float().to(device)

                logits, _ = model(velocity, acceleration, lota)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, all_probs)
            ap = average_precision_score(all_labels, all_probs)
        else:
            auc = 0.5
            ap = 0.5

        return {
            'loss': total_loss / len(loader),
            'auc': auc,
            'ap': ap
        }, all_labels, all_probs

    def find_best_threshold(self, labels, probs):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.9, 100):
            preds = (probs > thresh).astype(int)
            try:
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            except:
                continue
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
        save_dir = os.path.dirname(save_path)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, device)

            val_metrics, val_labels, val_probs = self.evaluate(model, val_loader, criterion, device)

            best_th, best_f1 = self.find_best_threshold(val_labels, val_probs)
            try:
                preds = (val_probs > best_th).astype(int)
                tn, fp, fn, tp = confusion_matrix(val_labels, preds).ravel()
            except:
                tn = fp = fn = tp = 0

            # 监控残差权重
            with torch.no_grad():
                v_res_weight = model.adapter.velocity_res[2].weight.norm().item()
                a_res_weight = model.adapter.acceleration_res[2].weight.norm().item()

            print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val AUC: {val_metrics['auc']:.4f} | Best Thresh: {best_th:.4f} | F1: {best_f1:.4f}")
            print(f"   Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
            print(f"   Res Weight: v={v_res_weight:.6f}, a={a_res_weight:.6f}")

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
                    'best_val_f1': best_val_f1,
                    'epoch': epoch
                }
                torch.save(checkpoint, save_path)

                adapter_state = {}
                for name, param in model.named_parameters():
                    if 'adapter' in name:
                        adapter_state[name] = param.data.clone()
                torch.save(adapter_state, os.path.join(save_dir, 'adapter_weights.pth'))

                print(f"  >> 保存最佳模型 (F1={best_val_f1:.4f})")
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停于 epoch {epoch}")
                    break
            print("-" * 60)

        return best_state_dict, best_val_f1, best_threshold, best_epoch

    def run_experiment(self, config_name, train_loader, val_loader, test_loader,
                       model, device, args):
        print(f"\n{'=' * 70}")
        print(f"运行实验: {config_name}")
        print(f"  - Adapter: L2 + 独立残差 (速度/加速度各自专用)")
        print(f"  - CNN: 冻结")
        print(f"  - 学习率: {args.lr}")
        print(f"{'=' * 70}")

        model.freeze_cnn()

        model_path = os.path.join(self.exp_dir, 'models', f'{config_name}.pth')
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_state_dict, best_f1, best_threshold, best_epoch = self.train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, args.epochs, args.patience, model_path
        )

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        test_metrics, test_labels, test_probs = self.evaluate(model, test_loader, criterion, device)

        try:
            preds = (test_probs > best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
        except:
            tn = fp = fn = tp = 0

        result = {
            'config_name': config_name,
            'best_val_f1': float(best_f1),
            'best_threshold': float(best_threshold),
            'best_epoch': int(best_epoch),
            'test': {
                'loss': float(test_metrics['loss']),
                'auc': float(test_metrics['auc']),
                'ap': float(test_metrics['ap']),
                'accuracy': float((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0),
                'f1': float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
                'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
            }
        }

        self.results.append(result)

        with open(os.path.join(self.exp_dir, 'logs', f'{config_name}.json'), 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n测试结果: AUC={test_metrics['auc']:.4f}, AP={test_metrics['ap']:.4f}, F1={result['test']['f1']:.4f}")

        return result

    def generate_summary(self):
        if not self.results:
            return
        df = pd.DataFrame([{
            'config': r['config_name'],
            'best_val_f1': r['best_val_f1'],
            'test_auc': r['test']['auc'],
            'test_f1': r['test']['f1']
        } for r in self.results])
        df.to_csv(os.path.join(self.exp_dir, 'summary.csv'), index=False)
        print("\n" + "=" * 80)
        print("实验汇总")
        print("=" * 80)
        print(df.to_string(index=False))


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', default=None)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--val-ratio', type=float, default=0.3)
    parser.add_argument('--lota-model', required=True)
    parser.add_argument('--pretrained-cnn', default=None)
    parser.add_argument('--encoder', default='XCLIP-16')
    parser.add_argument('--loss-type', default='l2', choices=['l2', 'cos'])
    parser.add_argument('--gpu-id', default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--target-len', type=int, default=8)
    parser.add_argument('--max-videos', type=int, default=30000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save-dir', default='ablation_results_l2_residual')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--no-cache', action='store_true')

    parser.add_argument('--enable_image_aug', action='store_true', default=True)
    parser.add_argument('--image_aug_prob', type=float, default=0.5)
    parser.add_argument('--enable_temporal_aug', action='store_true', default=True)
    parser.add_argument('--no_temporal_aug', dest='enable_temporal_aug', action='store_false')
    parser.add_argument('--time_mask_ratio', type=float, default=0.2)
    parser.add_argument('--time_shift_max', type=int, default=1)
    parser.add_argument('--feature_dropout', type=float, default=0.1)
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

    from models import D3_model
    d3_model = D3_model(encoder_type=args.encoder, loss_type=args.loss_type).to(device)
    d3_model.eval()
    for param in d3_model.parameters():
        param.requires_grad = False

    lota_model = load_lota_model(args.lota_model, device)

    train_csv_path = prepare_csv_path(args.train_csv)

    if args.val_csv is None:
        df_all = pd.read_csv(train_csv_path)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df_all, test_size=args.val_ratio,
            random_state=args.seed, stratify=df_all['label']
        )
        temp_dir = tempfile.mkdtemp(prefix="split_")
        train_df.to_csv(os.path.join(temp_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(temp_dir, "val.csv"), index=False)
        train_csv_path = os.path.join(temp_dir, "train.csv")
        val_csv_path = os.path.join(temp_dir, "val.csv")
        self_created_val = True
    else:
        val_csv_path = prepare_csv_path(args.val_csv)
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        self_created_val = False

    test_csv_path = prepare_csv_path(args.test_csv)
    test_df = pd.read_csv(test_csv_path)

    if args.max_videos and args.max_videos > 0:
        if len(train_df) > args.max_videos:
            train_df = train_df.sample(n=args.max_videos, random_state=args.seed)
        if len(val_df) > args.max_videos:
            val_df = val_df.sample(n=args.max_videos, random_state=args.seed)
        if len(test_df) > args.max_videos:
            test_df = test_df.sample(n=args.max_videos, random_state=args.seed)

    all_df = pd.concat([train_df, val_df, test_df])

    cache = FeaturePrecomputedCache(cache_dir='feature_cache_768')
    features_dict = None

    if not args.no_cache and cache.exists():
        features_dict = cache.load()
        all_videos = all_df['content_path'].tolist()
        missing_videos = [v for v in all_videos if v not in features_dict]

        if missing_videos:
            print(f"\n缓存缺失 {len(missing_videos)} 个视频，只计算缺失部分...")
            missing_df = all_df[all_df['content_path'].isin(missing_videos)]
            print(f"需要计算: {len(missing_df)} 个视频")

            precomputer = FeaturePrecomputer(
                d3_model, lota_model, device,
                target_len=args.target_len,
                enable_image_aug=args.enable_image_aug,
                image_aug_prob=args.image_aug_prob
            )
            new_features = precomputer.precompute_all(missing_df, max_videos=None)
            features_dict.update(new_features)
            cache.save(features_dict)
            print(f"合并完成，总视频数: {len(features_dict)}")

            train_df = train_df[train_df['content_path'].isin(features_dict.keys())]
            val_df = val_df[val_df['content_path'].isin(features_dict.keys())]
            test_df = test_df[test_df['content_path'].isin(features_dict.keys())]
            all_df = pd.concat([train_df, val_df, test_df])
            print(f"更新后: 训练集 {len(train_df)}, 验证集 {len(val_df)}, 测试集 {len(test_df)}")

    if features_dict is None:
        print("\n没有缓存，全部重新计算...")
        precomputer = FeaturePrecomputer(
            d3_model, lota_model, device,
            target_len=args.target_len,
            enable_image_aug=args.enable_image_aug,
            image_aug_prob=args.image_aug_prob
        )
        features_dict = precomputer.precompute_all(all_df, max_videos=None)
        cache.save(features_dict)

    train_videos = [v for v in train_df['content_path'].tolist() if v in features_dict]
    val_videos = [v for v in val_df['content_path'].tolist() if v in features_dict]
    test_videos = [v for v in test_df['content_path'].tolist() if v in features_dict]

    print(f"\n有效数据: 训练={len(train_videos)}, 验证={len(val_videos)}, 测试={len(test_videos)}")

    train_set = CachedFrameLevelDataset(features_dict, train_videos, "TRAIN",
                                        lota_dropout_rate=args.lota_dropout, lota_noise_std=args.lota_noise,
                                        enable_temporal_aug=args.enable_temporal_aug,
                                        time_mask_ratio=args.time_mask_ratio,
                                        time_shift_max=args.time_shift_max,
                                        feature_dropout=args.feature_dropout
                                        )
    val_set = CachedFrameLevelDataset(features_dict, val_videos, "VAL",
                                      lota_dropout_rate=0, lota_noise_std=0, enable_temporal_aug=False
                                      )
    test_set = CachedFrameLevelDataset(features_dict, test_videos, "TEST",
                                       lota_dropout_rate=0, lota_noise_std=0, enable_temporal_aug=False
                                       )

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=(args.device == 'cuda'))
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=4, pin_memory=(args.device == 'cuda'))
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=4,
                             pin_memory=(args.device == 'cuda'))

    model = AdapterModel1D(
        feature_dim=768,
        hidden_dim=4,
        kernel_size=3,
        dropout=0.3
    ).to(device)

    if args.pretrained_cnn:
        model.load_pretrained_cnn(args.pretrained_cnn)

    experiment = AblationExperiment(base_dir=args.save_dir)
    experiment.run_experiment('l2_residual_independent', train_loader, val_loader, test_loader, model, device, args)
    experiment.generate_summary()

    if self_created_val and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()