import os
import pandas as pd
from pandas import Series
from glob import glob
import glob as glob_module
import argparse

def clean_checkpoint_files(dataset_path):
    """清理所有checkpoint文件（保持不变）"""
    frames_dir = os.path.join(dataset_path, 'frames')
    checkpoint_files = glob_module.glob(os.path.join(frames_dir, '**', '*checkpoint*'), recursive=True)
    for file in checkpoint_files:
        try:
            os.remove(file)
            print(f"已删除: {file}")
        except:
            pass
    print(f"清理了 {len(checkpoint_files)} 个checkpoint文件")

def count_images_in_folder(folder_path):
    """统计文件夹中的图片数量（保持不变）"""
    image_count = 0
    image_names = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith('.'):
            continue
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_count += 1
            try:
                image_names.append(int(file_name.split('.')[0]))
            except:
                continue
    image_names.sort()
    return image_count, image_names

def get_all_video_dirs(frames_dir):
    """获取所有包含图片的视频目录（即每个视频的帧文件夹）"""
    video_dirs = []
    for root, dirs, files in os.walk(frames_dir):
        has_images = any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files)
        if has_images:
            video_dirs.append(root)
    return video_dirs

def process_subfolder(subfolder_path, is_real, dataset_path):
    """
    处理一个子文件夹：收集其下所有视频帧文件夹，生成 CSV 并保存。
    subfolder_path: frames 下的直接子文件夹的完整路径
    is_real: 是否为真实视频
    dataset_path: 数据集根目录（用于构造 csv 输出路径）
    """
    print(f"\n正在处理子文件夹: {os.path.basename(subfolder_path)}")
    
    # 获取该子文件夹下的所有视频帧文件夹（递归）
    video_dirs = get_all_video_dirs(subfolder_path)
    if not video_dirs:
        print(f"⚠️ 子文件夹 {subfolder_path} 中没有找到任何视频帧文件夹，跳过")
        return

    label = list()
    save_path = list()
    frame_counts = list()
    frame_seq_counts = list()
    content_paths = list()
    str_labels = list()

    for video_dir in video_dirs:
        temp_frame_count, temp_frame_seqs = count_images_in_folder(video_dir)
        if temp_frame_count == 0:
            continue

        frame_files = sorted(glob(os.path.join(video_dir, '*.[jJ][pP][gG]')) +
                            glob(os.path.join(video_dir, '*.[jJ][pP][eE][gG]')) +
                            glob(os.path.join(video_dir, '*.[pP][nN][gG]')))
        if frame_files:
            frame_path = frame_files[0]
            if is_real:
                label.append(str(0))
                str_labels.append('Real Video')
            else:
                label.append(str(1))
                str_labels.append('AI Video')
            frame_counts.append(temp_frame_count)
            frame_seq_counts.append(temp_frame_seqs)
            save_path.append(frame_path)
            content_paths.append(video_dir)

    if not content_paths:
        print(f"⚠️ 子文件夹 {subfolder_path} 中无有效视频数据")
        return

    dic = {
        'content_path': Series(data=content_paths),
        'image_path': Series(data=save_path),
        'type_id': Series(data=str_labels),
        'label': Series(data=label),
        'frame_len': Series(data=frame_counts),
        'frame_seq': Series(data=frame_seq_counts)
    }
    df = pd.DataFrame(dic)

    # 输出 CSV 文件名：子文件夹名.csv
    folder_name = os.path.basename(subfolder_path)
    csv_dir = os.path.join(dataset_path, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{folder_name}.csv")
    df.to_csv(csv_path, encoding='utf-8', index=False)
    print(f"✅ 已保存 {len(df)} 条记录到: {csv_path}")

    # 验证第一行
    if len(df) > 0:
        print(f"  示例: content_path={df['content_path'].iloc[0]}, 存在={os.path.exists(df['content_path'].iloc[0])}")

def main(is_real, dataset_path, folder_paths=None):
    # 清理checkpoint文件（只需一次）
    print("清理checkpoint文件中...")
    clean_checkpoint_files(dataset_path)
    print("清理完成")

    frames_dir = os.path.join(dataset_path, 'frames')
    if not os.path.exists(frames_dir):
        print(f"❌ 帧目录不存在: {frames_dir}")
        return

    # 如果指定了 folder_paths，则按原逻辑处理（可处理单个或多个文件夹，输出合并的 CSV）
    if folder_paths is not None and len(folder_paths) > 0:
        # 原有逻辑：处理指定的子文件夹（可多个），合并成一个 CSV
        all_video_dirs = []
        for folder in folder_paths:
            folder_full = os.path.join(frames_dir, folder)
            if os.path.exists(folder_full):
                all_video_dirs.extend(get_all_video_dirs(folder_full))
            else:
                print(f"⚠️ 目录不存在: {folder_full}")

        if not all_video_dirs:
            print("❌ 没有找到任何视频帧文件夹")
            return

        # 准备数据（与原脚本相同）
        label = list()
        save_path = list()
        frame_counts = list()
        frame_seq_counts = list()
        content_paths = list()
        str_labels = list()

        for video_dir in all_video_dirs:
            temp_frame_count, temp_frame_seqs = count_images_in_folder(video_dir)
            if temp_frame_count == 0:
                continue
            frame_files = sorted(glob(os.path.join(video_dir, '*.[jJ][pP][gG]')) +
                                glob(os.path.join(video_dir, '*.[jJ][pP][eE][gG]')) +
                                glob(os.path.join(video_dir, '*.[pP][nN][gG]')))
            if frame_files:
                frame_path = frame_files[0]
                if is_real:
                    label.append(str(0))
                    str_labels.append('Real Video')
                else:
                    label.append(str(1))
                    str_labels.append('AI Video')
                frame_counts.append(temp_frame_count)
                frame_seq_counts.append(temp_frame_seqs)
                save_path.append(frame_path)
                content_paths.append(video_dir)

        if not content_paths:
            print("❌ 没有找到任何有效视频")
            return

        dic = {
            'content_path': Series(data=content_paths),
            'image_path': Series(data=save_path),
            'type_id': Series(data=str_labels),
            'label': Series(data=label),
            'frame_len': Series(data=frame_counts),
            'frame_seq': Series(data=frame_seq_counts)
        }
        df = pd.DataFrame(dic)

        # 确定输出文件名：若只指定了一个文件夹，则用该文件夹名；否则用 real_all/fake_all
        if len(folder_paths) == 1:
            output_name = f"{folder_paths[0]}.csv"
        else:
            output_name = 'real_all.csv' if is_real else 'fake_all.csv'

        csv_dir = os.path.join(dataset_path, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, output_name)
        df.to_csv(csv_path, encoding='utf-8', index=False)
        print(f"\n✅ 已保存 {len(df)} 条记录到: {csv_path}")

    else:
        # 未指定 folder_paths：自动处理 frames 下的每个直接子文件夹
        print("\n未指定 --folders，将自动为 frames 下的每个直接子文件夹生成 CSV")
        subfolders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
        if not subfolders:
            print("❌ frames 下没有子文件夹")
            return

        for sub in subfolders:
            sub_full = os.path.join(frames_dir, sub)
            process_subfolder(sub_full, is_real, dataset_path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Got: {v}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为 frames 下每个子文件夹生成单独的 CSV')
    parser.add_argument('--is-real', type=str2bool, required=True,
                        help="视频是否为真实：--is-real True 或 False")
    parser.add_argument('--dataset-path', type=str, default='GenVideo',
                        help="数据集根目录路径")
    parser.add_argument('--folders', nargs='*', default=None,
                        help="可选：指定要处理的子文件夹列表（不指定则自动处理所有直接子文件夹）")
    args = parser.parse_args()
    main(args.is_real, args.dataset_path, args.folders)