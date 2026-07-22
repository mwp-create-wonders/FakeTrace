import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
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


# ==================== 视频帧提取器（带帧号追踪）====================
class VideoFrameExtractor:
    def __init__(self, video_path, target_frames=8, target_size=224):
        self.video_path = Path(video_path)
        self.target_frames = target_frames
        self.target_size = target_size
        self.temp_dir = None
        self.frame_timestamps = []      # 所有帧的时间戳
        self.frame_numbers = []         # 所有帧的序号
        self.selected_frame_info = []   # 选中的帧信息

    def get_video_info(self):
        """获取视频总时长和总帧数"""
        try:
            # 获取时长
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else None

            # 获取总帧数
            cmd = [
                'ffprobe', '-v', 'error',
                '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            total_frames = int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else None

            # 获取帧率
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            fps_str = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
            if fps_str and '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else None
            else:
                fps = float(fps_str) if fps_str else None

            return duration, total_frames, fps
        except Exception:
            return None, None, None

    def get_all_frame_timestamps(self):
        """获取所有帧的时间戳"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'frame=pts_time',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(self.video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                timestamps = [float(t) for t in result.stdout.strip().split() if t]
                return timestamps
        except Exception:
            pass
        return None

    def _find_nearest_frame(self, timestamps, target_time):
        """在时间戳列表中找最接近目标时间的帧索引"""
        if not timestamps:
            return None
        idx = np.argmin(np.abs(np.array(timestamps) - target_time))
        return idx

    def extract_frames(self):
        try:
            duration, total_frames, fps = self.get_video_info()
            if duration is None or duration < 0.5:
                return None

            self.temp_dir = tempfile.mkdtemp(prefix="video_frames_")

            # ===== 方法1：基于时长均匀采样 =====
            frame_interval = duration / self.target_frames
            sample_times = [i * frame_interval for i in range(self.target_frames)]

            cmd = [
                'ffmpeg', '-loglevel', 'error',
                '-i', str(self.video_path),
                '-vf', f'fps=1/{frame_interval},scale={self.target_size}:{self.target_size}',
                '-q:v', '2',
                os.path.join(self.temp_dir, '%d.jpg')
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            frame_files = sorted(glob.glob(os.path.join(self.temp_dir, '*.jpg')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            # ===== 方法2：固定帧率采样（备用）=====
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
                frame_files = sorted(glob.glob(os.path.join(self.temp_dir, '*.jpg')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                # 取前 target_frames 帧
                frame_files = frame_files[:self.target_frames]
                sample_times = [i / self.target_frames * duration for i in range(len(frame_files))]

            if len(frame_files) < self.target_frames:
                return None

            # 如果多于目标帧数，均匀采样
            if len(frame_files) > self.target_frames:
                indices = np.linspace(0, len(frame_files) - 1, self.target_frames, dtype=int)
                frame_files = [frame_files[i] for i in indices]
                sample_times = [i * duration / self.target_frames for i in range(self.target_frames)]

            # ===== 获取真实帧号 =====
            all_timestamps = self.get_all_frame_timestamps()

            selected_info = []
            for i, frame_path in enumerate(frame_files):
                # 从文件名获取输出序号
                basename = os.path.basename(frame_path)
                output_idx = int(os.path.splitext(basename)[0])

                # 目标时间
                target_time = sample_times[i] if i < len(sample_times) else i * duration / self.target_frames

                # 找真实帧号
                frame_number = None
                actual_time = None
                if all_timestamps:
                    nearest_idx = self._find_nearest_frame(all_timestamps, target_time)
                    if nearest_idx is not None:
                        frame_number = nearest_idx + 1  # 帧号从1开始
                        actual_time = all_timestamps[nearest_idx]

                selected_info.append({
                    'file_path': frame_path,
                    'output_index': output_idx,
                    'frame_number': frame_number,
                    'target_time': target_time,
                    'actual_time': actual_time,
                })

            self.selected_frame_info = selected_info
            return frame_files, selected_info

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
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ==================== 加载 Adapter 模型 ====================
def load_adapter_model(model_dir, device):
    from Train import AdapterModel1D

    model_dir = resolve_path(model_dir)
    model_path = model_dir / "l2_residual_independent.pth"

    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return None, 0.5

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if 'adapter.velocity_res.0.weight' in state_dict:
        hidden_dim = state_dict['adapter.velocity_res.0.weight'].shape[0]
    else:
        hidden_dim = 8

    model = AdapterModel1D(
        feature_dim=768,
        hidden_dim=hidden_dim,
        kernel_size=3,
        dropout=0.3
    ).to(device)

    model.load_state_dict(state_dict)
    best_threshold = checkpoint.get('best_threshold', 0.5)
    print(f"  ✅ 加载模型: l2_residual_independent.pth (阈值: {best_threshold:.4f})")
    return model, best_threshold


# ==================== 特征提取（含帧信息）====================
def extract_sequence_features_from_frames(frame_paths, frame_info_list, d3_model, lota_model, device, d3_transform, target_len=8):
    if len(frame_paths) < target_len:
        return None

    frame_paths = frame_paths[:target_len]
    frame_info_list = frame_info_list[:target_len]

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

    frame_features = frame_features[0].cpu().numpy()  # [8, 768]

    velocity_raw = frame_features[1:] - frame_features[:-1]  # [7, 768]
    acceleration_raw = velocity_raw[1:] - velocity_raw[:-1]  # [6, 768]

    align_len = target_len - 2  # 6
    velocity_raw = velocity_raw[:align_len]
    acceleration_raw = acceleration_raw[:align_len]

    # 帧信息也截取到6帧
    frame_info_list = frame_info_list[:align_len]

    batch_tensor = frames_tensor.squeeze(0)
    with torch.no_grad():
        logits = lota_model(batch_tensor)
        lota_scores = torch.sigmoid(logits).cpu().numpy().flatten()
    lota_scores = lota_scores[:align_len]

    return {
        'velocity_raw': velocity_raw,
        'acceleration_raw': acceleration_raw,
        'lota_scores': lota_scores,
        'velocity_l2': np.linalg.norm(velocity_raw, axis=1),
        'acceleration_l2': np.linalg.norm(acceleration_raw, axis=1),
        'frame_info': frame_info_list,  # 包含帧号和时间的列表
    }


# ==================== 推理函数 ====================
def infer_video(video_path, model, d3_model, lota_model, device, d3_transform, threshold=0.5):
    extractor = VideoFrameExtractor(video_path, target_frames=8, target_size=224)
    result = extractor.extract_frames()

    if result is None:
        extractor.cleanup()
        return None

    frame_paths, frame_info = result

    try:
        features = extract_sequence_features_from_frames(
            frame_paths, frame_info, d3_model, lota_model, device, d3_transform, target_len=8
        )
        if features is None:
            return None

        velocity_raw = features['velocity_raw']
        acceleration_raw = features['acceleration_raw']
        lota_scores = features['lota_scores']

        velocity_t = torch.FloatTensor(velocity_raw).unsqueeze(0).to(device)
        acceleration_t = torch.FloatTensor(acceleration_raw).unsqueeze(0).to(device)
        lota_t = torch.FloatTensor(lota_scores).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(velocity_t, acceleration_t, lota_t)
            prob = torch.sigmoid(logits).item()

        prediction = 1 if prob > threshold else 0

        # 构建证据（用于报告）
        evidence = {
            'frame_info': features['frame_info'],
            'velocity_l2': features['velocity_l2'].tolist(),
            'acceleration_l2': features['acceleration_l2'].tolist(),
            'lota_scores': features['lota_scores'].tolist(),
        }

        return {
            'prob': prob,
            'prediction': prediction,
            'label': 'AI生成' if prediction == 1 else '真实',
            'evidence': evidence
        }
    finally:
        extractor.cleanup()


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description='视频检测程序 (Adapter 版本)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--encoder', default='XCLIP-16')
    parser.add_argument('--loss', default='l2', choices=['l2', 'cos'])
    parser.add_argument('--video', help='单个视频路径')
    parser.add_argument('--video-dir', help='视频目录')
    parser.add_argument('--output', default=None, help='结果保存路径')
    parser.add_argument('--threshold', type=float, default=None, help='判定阈值')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--model-dir', default=str(MODEL_DIR), help='模型目录')
    parser.add_argument('--lota-model', default=str(LOTA_MODEL_PATH), help='LOTA 模型路径')

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

    # 1. 加载 D3 模型
    print("加载 D3 模型...")
    from models import D3_model
    d3_model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device)
    d3_model.eval()
    for param in d3_model.parameters():
        param.requires_grad = False

    # 2. 加载 LOTA 模型
    print("加载 LOTA 模型...")
    lota_model = load_lota_model(args.lota_model, device)
    if lota_model is None:
        print("LOTA 模型加载失败")
        return

    # 3. 图像变换
    d3_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. 加载 Adapter 模型
    print(f"\n📂 加载模型权重从: {args.model_dir}")
    model, model_threshold = load_adapter_model(args.model_dir, device)
    if model is None:
        return
    model.eval()

    threshold = args.threshold if args.threshold is not None else model_threshold
    print(f"使用阈值: {threshold:.4f}")

    # 5. 收集视频文件
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

    # 6. 逐个推理
    results = []
    for video_path_str in tqdm(video_files, desc="检测进度"):
        video_path = Path(video_path_str)
        video_name = video_path.name
        result = infer_video(str(video_path), model, d3_model, lota_model, device, d3_transform, threshold)

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
                'label': result['label'],
                'evidence': result['evidence']
            })

    # 7. 保存结果
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