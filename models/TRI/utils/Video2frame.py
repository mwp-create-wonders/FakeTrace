

import os
import sys
import cv2
import numpy as np
from glob import glob
import multiprocessing
import math
import random
import subprocess
import traceback
from pathlib import Path

class VideoFrameExtractor:
    def __init__(self, dataset_path="GenVideo"):
        self.dataset_path = Path(dataset_path)
        self.video_dir = self.dataset_path / "video"
        self.frames_dir = self.dataset_path / "frames"
        self.log_file = self.dataset_path / "extraction.log"
        
        # 创建目录
        self.frames_dir.mkdir(exist_ok=True)
        
    def get_video_info(self, video_path):
        """获取视频信息"""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=duration,width,height,r_frame_rate,codec_name',
                '-of', 'json',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                if 'streams' in info and info['streams']:
                    stream = info['streams'][0]
                    fps_str = stream.get('r_frame_rate', '30/1')
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den != 0 else 30
                    
                    return {
                        'duration': float(stream.get('duration', 0)),
                        'width': int(stream.get('width', 0)),
                        'height': int(stream.get('height', 0)),
                        'fps': fps,
                        'codec': stream.get('codec_name', 'unknown'),
                        'valid': True
                    }
        except:
            pass
        
        return {'valid': False}
    
    def extract_frames_safe(self, video_path, output_dir, target_frames=24):
        """安全地提取帧"""
        try:
            video_path = Path(video_path)
            output_dir = Path(output_dir)
            
            # 如果已存在足够帧，跳过
            if output_dir.exists():
                existing_frames = len(list(output_dir.glob('*.jpg')))
                if existing_frames >= target_frames * 0.8:  # 80%就认为OK
                    return True, f"已存在{existing_frames}帧"
            
            # 获取视频信息
            info = self.get_video_info(video_path)
            if not info['valid']:
                return False, "无法获取视频信息"
            
            duration = info['duration']
            if duration < 1:
                return False, f"视频过短: {duration:.1f}s"
            
            # 计算提取参数
            fps = 8  # 目标fps
            extract_duration = min(3.0, duration)  # 提取3秒或整个视频
            
            if duration <= extract_duration:
                start_time = 0
            else:
                start_time = random.uniform(0, duration - extract_duration)
            
            # 创建输出目录
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 方法1：标准提取
            cmd = [
                'ffmpeg', '-loglevel', 'error',
                '-ss', str(start_time),
                '-t', str(extract_duration),
                '-i', str(video_path),
                '-vf', f'fps={fps},scale=224:224',
                '-q:v', '2',
                '-pix_fmt', 'yuvj420p',  # 确保兼容性
                str(output_dir / '%d.jpg')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                # 方法2：尝试不同的参数
                cmd = [
                    'ffmpeg', '-loglevel', 'error',
                    '-ss', str(start_time),
                    '-t', str(extract_duration),
                    '-i', str(video_path),
                    '-r', str(fps),
                    '-s', '224x224',
                    '-q:v', '2',
                    str(output_dir / '%d.jpg')
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # 验证提取结果
                extracted = len(list(output_dir.glob('*.jpg')))
                if extracted > 0:
                    return True, f"提取{extracted}帧"
                else:
                    return False, "未生成帧"
            else:
                return False, f"ffmpeg失败: {result.stderr[:100]}"
                
        except Exception as e:
            return False, f"异常: {str(e)}"
    
    def process_video(self, args):
        """处理单个视频"""
        video_path, video_name = args
        success, message = self.extract_frames_safe(
            video_path,
            self.frames_dir / video_name
        )
        
        return video_name, success, message
    
    def run(self, max_workers=8):
        """运行提取"""
        print("🔍 扫描视频文件...")
        
        # 查找所有视频
        video_files = []
        for ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            video_files.extend(self.video_dir.rglob(f'*.{ext}'))
        
        print(f"📊 找到 {len(video_files)} 个视频文件")
        
        if not video_files:
            print("❌ 未找到视频文件")
            return
        
        # 准备参数
        args_list = []
        for video_path in video_files:
            # 生成输出目录名
            rel_path = video_path.relative_to(self.video_dir)
            output_name = rel_path.with_suffix('')  # 去掉扩展名
            args_list.append((video_path, output_name))
        
        # 并行处理
        print(f"🚀 开始处理，使用 {max_workers} 个进程...")
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        with multiprocessing.Pool(processes=min(max_workers, os.cpu_count() or 4)) as pool:
            results = pool.imap_unordered(self.process_video, args_list)
            
            import time
            from tqdm import tqdm
            
            for video_name, success, message in tqdm(results, total=len(args_list), desc="提取进度"):
                if success:
                    if "已存在" in message:
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    # 记录到日志
                    with open(self.log_file, 'a') as f:
                        f.write(f"{video_name}: {message}\n")
        
        # 输出统计
        print("\n" + "="*60)
        print("🎉 处理完成!")
        print("="*60)
        print(f"✅ 成功提取: {success_count}")
        print(f"⏭️  跳过已存在: {skip_count}")
        print(f"❌ 失败: {fail_count}")
        
        if fail_count > 0:
            print(f"\n📋 查看失败详情: cat {self.log_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='健壮的视频帧提取')
    parser.add_argument('--dataset-path', default='GenVideo', help='数据集路径')
    parser.add_argument('--workers', type=int, default=8, help='工作进程数')
    parser.add_argument('--check-only', action='store_true', help='仅检查不处理')
    
    args = parser.parse_args()
    
    extractor = VideoFrameExtractor(args.dataset_path)
    
    if args.check_only:
        # 仅检查
        print("检查帧提取状态...")
        # 实现检查逻辑
    else:
        extractor.run(max_workers=args.workers)