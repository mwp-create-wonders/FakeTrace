import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from PIL import Image
from torchvision import transforms
import json
import glob
import random
import subprocess
import tempfile
import shutil
from pathlib import Path

# ==================== 稳定路径锚点 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# ==================== 模型路径 ====================
LOTA_MODEL_PATH = SCRIPT_DIR / "LOTA" / "lota_weights" / "Network_best.pth"
MODEL_DIR = SCRIPT_DIR / "weights"


def resolve_path(path_input, base_dir=None):
    if path_input is None:
        return None
    p = Path(path_input)
    if p.is_absolute():
        return p
    base = base_dir if base_dir is not None else SCRIPT_DIR
    return (base / p).resolve()


# ==================== 视频帧提取器 ====================
class VideoFrameExtractor:
    def __init__(self, video_path, target_frames=8, target_size=224):
        self.video_path = Path(video_path)
        self.target_frames = target_frames
        self.target_size = target_size
        self.temp_dir = None

    def get_video_info(self):
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return None

    def extract_frames(self):
        try:
            duration = self.get_video_info()
            if duration is None or duration < 1:
                return None

            self.temp_dir = tempfile.mkdtemp(prefix="video_frames_")

            # 方法1：基于时长均匀采样
            frame_interval = duration / self.target_frames
            cmd = [
                'ffmpeg', '-loglevel', 'error',
                '-i', str(self.video_path),
                '-vf', f'fps=1/{frame_interval},scale={self.target_size}:{self.target_size}',
                '-q:v', '2',
                os.path.join(self.temp_dir, '%d.jpg')
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            frame_files = sorted(glob.glob(os.path.join(self.temp_dir, '*.jpg')))

            # 方法2：固定帧率（备用）
            if len(frame_files) < self.target_frames:
                cmd = [
                    'ffmpeg', '-loglevel', 'error',
                    '-i', str(self.video_path),
                    '-r', str(self.target_frames),
                    '-s', f'{self.target_size}x{self.target_size}',
                    '-q:v', '2',
                    os.path.join(self.temp_dir, '%d.jpg')
                ]
                subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                frame_files = sorted(glob.glob(os.path.join(self.temp_dir, '*.jpg')))

            if len(frame_files) < self.target_frames:
                return None

            # 均匀采样到目标帧数
            if len(frame_files) > self.target_frames:
                indices = np.linspace(0, len(frame_files) - 1, self.target_frames, dtype=int)
                frame_files = [frame_files[i] for i in indices]

            return frame_files

        except Exception as e:
            print(f"帧提取失败: {e}")
            return None

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# ==================== 工具函数 ====================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lota_model(model_path, device='cuda'):
    model_path = resolve_path(model_path)
    lota_test_path = SCRIPT_DIR / "LOTA" / "test.py"
    lota_dir = str(SCRIPT_DIR / "LOTA")

    if not lota_test_path.exists():
        print(f"❌ LOTA test.py 不存在: {lota_test_path}")
        return None

    if lota_dir not in sys.path:
        sys.path.insert(0, lota_dir)

    import importlib.util
    spec = importlib.util.spec_from_file_location("LOTA_test", lota_test_path)
    lota_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lota_module)
    DeepLearningModel = lota_module.DeepLearningModel

    model = DeepLearningModel()
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def compute_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores > threshold).astype(int)
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        if 1 in unique_classes:
            tp = (y_pred == 1).sum()
            fn = (y_pred == 0).sum()
            tn = fp = 0
        else:
            tn = (y_pred == 0).sum()
            fp = (y_pred == 1).sum()
            tp = fn = 0

    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    try:
        if len(unique_classes) == 2:
            ap = average_precision_score(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
        else:
            ap = auc = 0.5
    except Exception:
        ap = auc = 0.0

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'acc': acc,
            'precision': precision, 'recall': recall, 'f1': f1, 'ap': ap, 'auc': auc, 'total': total}


def extract_sequence_features_from_frames(frame_paths, d3_model, lota_model, device, d3_transform, target_len=8):
    if len(frame_paths) < target_len:
        return None

    frame_paths = frame_paths[:target_len]
    frames_tensor = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert('RGB')
            frames_tensor.append(d3_transform(img))
        except Exception:
            break

    if len(frames_tensor) < target_len:
        return None

    frames_tensor = torch.stack(frames_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = d3_model(frames_tensor, return_vectors=True)
        frame_features = output['frame_features']

    frame_features = frame_features[0].cpu().numpy()
    velocity_raw = frame_features[1:] - frame_features[:-1]
    velocity_norm = np.linalg.norm(velocity_raw, axis=1)
    acceleration_raw = velocity_raw[1:] - velocity_raw[:-1]
    acceleration_norm = np.linalg.norm(acceleration_raw, axis=1)

    align_len = target_len - 2
    velocity_norm = velocity_norm[:align_len]
    acceleration_norm = acceleration_norm[:align_len]

    batch_tensor = frames_tensor.squeeze(0)
    with torch.no_grad():
        logits = lota_model(batch_tensor)
        lota_scores = torch.sigmoid(logits).cpu().numpy().flatten()
    lota_scores = lota_scores[:align_len]

    return {'velocity': velocity_norm, 'acceleration': acceleration_norm, 'lota': lota_scores}


def load_model_and_weights(model_dir, device):
    from Train import TemporalFusionNet

    model_dir = resolve_path(model_dir)
    model_path = model_dir / "vel_lota.pth"
    input_dim = 2

    model = TemporalFusionNet(input_dim=input_dim, hidden_dim=64, kernel_size=3, dropout=0.3).to(device)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if list(state_dict['conv1.0.weight'].shape)[1] == 3:
                new_weight = state_dict['conv1.0.weight'][:, :2, :]
                state_dict['conv1.0.weight'] = new_weight
            model.load_state_dict(state_dict, strict=False)
            best_threshold = checkpoint.get('best_threshold', 0.5)
            print(f"  ✅ 加载模型: vel_lota.pth (阈值: {best_threshold:.4f})")
            return model, best_threshold
        else:
            model.load_state_dict(checkpoint, strict=False)
            print(f"  ✅ 加载模型: vel_lota.pth")
            return model, 0.5
    else:
        print(f"  ❌ 模型不存在: {model_path}")
        return None, 0.5


def test_video(video_path, model, d3_model, lota_model, device, d3_transform, threshold=0.5):
    extractor = VideoFrameExtractor(video_path, target_frames=8, target_size=224)
    frame_paths = extractor.extract_frames()

    if frame_paths is None:
        extractor.cleanup()
        return None

    try:
        features = extract_sequence_features_from_frames(
            frame_paths, d3_model, lota_model, device, d3_transform, target_len=8
        )
        if features is None:
            return None

        seq_len = len(features['velocity'])
        sequence = [[features['velocity'][t], features['lota'][t]] for t in range(seq_len)]
        sequence = np.array(sequence)

        X = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(X)
            prob = torch.sigmoid(logits).item()

        prediction = 1 if prob > threshold else 0
        return {'prob': prob, 'prediction': prediction, 'label': 'AI生成' if prediction == 1 else '真实'}
    finally:
        extractor.cleanup()


def main():
    parser = argparse.ArgumentParser(description='视频检测程序')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--encoder', default='XCLIP-16')
    parser.add_argument('--loss', default='l2', choices=['l2', 'cos'])
    parser.add_argument('--video', help='单个视频路径')
    parser.add_argument('--video-dir', help='视频目录')
    parser.add_argument('--output', default=None, help='结果保存路径')
    parser.add_argument('--threshold', type=float, default=None, help='判定阈值')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()
    if not args.video and not args.video_dir:
        parser.error("必须提供 --video 或 --video-dir")

    seed_everything(args.seed)

    if args.device == 'cuda' and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = torch.device('cuda')
        print(f"使用设备: CUDA (GPU {args.gpu_id})")
    else:
        device = torch.device('cpu')
        print("使用设备: CPU")

    print("加载 D3 模型...")
    from models import D3_model
    d3_model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device)
    d3_model.eval()
    for param in d3_model.parameters():
        param.requires_grad = False

    print("加载 LOTA 模型...")
    lota_model = load_lota_model(LOTA_MODEL_PATH, device)
    if lota_model is None:
        print("LOTA 模型加载失败")
        return

    d3_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"\n📂 加载模型权重从: {MODEL_DIR}")
    model, model_threshold = load_model_and_weights(MODEL_DIR, device)
    if model is None:
        return
    model.eval()

    threshold = args.threshold if args.threshold is not None else model_threshold
    print(f"使用阈值: {threshold:.4f}")

    if args.video_dir:
        video_dir = resolve_path(args.video_dir)
        video_files = []
        for ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            video_files.extend(glob.glob(str(video_dir / f'*.{ext}')))
        print(f"\n找到 {len(video_files)} 个视频文件")
        folder_name = video_dir.name
        output_filename = f"detection_results_{folder_name}.json"
    else:
        video_path = resolve_path(args.video)
        video_files = [str(video_path)]
        video_name = video_path.stem
        output_filename = f"detection_results_{video_name}.json"

    if args.output:
        output_path = resolve_path(args.output)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        output_path = RESULTS_DIR / output_filename

    results = []
    for video_path_str in tqdm(video_files, desc="检测进度"):
        video_path = Path(video_path_str)
        video_name = video_path.name
        result = test_video(str(video_path), model, d3_model, lota_model, device, d3_transform, threshold)

        if result is None:
            print(f"  ❌ 处理失败: {video_name}")
            results.append({'video': video_name, 'path': str(video_path), 'success': False})
        else:
            print(f"  ✅ {video_name}: {result['label']} (概率={result['prob']:.4f})")
            results.append({
                'video': video_name,
                'path': str(video_path),
                'success': True,
                'prob': result['prob'],
                'prediction': result['prediction'],
                'label': result['label']
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果保存至: {output_path}")
    success_results = [r for r in results if r['success']]
    if success_results:
        ai_count = sum(1 for r in success_results if r['prediction'] == 1)
        real_count = sum(1 for r in success_results if r['prediction'] == 0)
        print(f"\n统计: 真实={real_count}, AI生成={ai_count}")


if __name__ == '__main__':
    main()