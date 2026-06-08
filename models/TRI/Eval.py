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

# ==================== 工具函数 ====================
def find_all_csv_files(csv_dir):
    """递归查找所有 CSV 文件"""
    all_csv_files = []
    if os.path.isdir(csv_dir):
        for root, dirs, files in os.walk(csv_dir):
            for file in files:
                if file.endswith('.csv') and not file.startswith('.'):
                    all_csv_files.append(os.path.join(root, file))
    elif os.path.isfile(csv_dir) and csv_dir.endswith('.csv'):
        all_csv_files = [csv_dir]
    return sorted(all_csv_files)


def load_lota_model(model_path, device='cuda'):
    """加载 LOTA 模型"""
    lota_path = os.path.join(os.path.dirname(__file__), "LOTA")
    if lota_path not in sys.path:
        sys.path.insert(0, lota_path)
    from test import DeepLearningModel
    
    model = DeepLearningModel()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def compute_metrics(y_true, y_scores, threshold=0.5):
    """计算各项指标"""
    y_pred = (y_scores > threshold).astype(int)
    
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        if 1 in unique_classes:
            tp = (y_pred == 1).sum()
            fn = (y_pred == 0).sum()
            tn = 0
            fp = 0
        else:
            tn = (y_pred == 0).sum()
            fp = (y_pred == 1).sum()
            tp = 0
            fn = 0
    
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    try:
        if len(unique_classes) == 2:
            ap = average_precision_score(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
        else:
            ap = 0.5
            auc = 0.5
    except:
        ap = 0.0
        auc = 0.0
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
        'ap': ap, 'auc': auc, 'total': total
    }


def extract_sequence_features(video_path, args, d3_model, lota_model, device, d3_transform):
    """提取 6×3 特征：速度标量 + 加速度标量 + LOTA"""
    if not os.path.exists(video_path):
        return None, False
    
    frame_files = sorted([
        f for f in os.listdir(video_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    if len(frame_files) < args.target_len:
        return None, False
    
    frame_files = frame_files[:args.target_len]
    frame_paths = [os.path.join(video_path, f) for f in frame_files]
    
    frames_tensor = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert('RGB')
            frames_tensor.append(d3_transform(img))
        except Exception:
            break
    if len(frames_tensor) < args.target_len:
        return None, False
    
    frames_tensor = torch.stack(frames_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = d3_model(frames_tensor, return_vectors=True)
        frame_features = output['frame_features']
    
    frame_features = frame_features[0].cpu().numpy()
    
    # 速度（一阶差分）
    velocity_raw = frame_features[1:] - frame_features[:-1]
    # 速度标量 = L2范数
    velocity_norm = np.linalg.norm(velocity_raw, axis=1)  # (7,)
    
    # 加速度（二阶差分）
    acceleration_raw = velocity_raw[1:] - velocity_raw[:-1]
    acceleration_norm = np.linalg.norm(acceleration_raw, axis=1)  # (6,)
    
    align_len = args.target_len - 2
    velocity_norm = velocity_norm[:align_len]  # (6,)
    acceleration_norm = acceleration_norm[:align_len]  # (6,)
    
    # LOTA 分数
    batch_tensor = frames_tensor.squeeze(0)
    with torch.no_grad():
        logits = lota_model(batch_tensor)
        lota_scores = torch.sigmoid(logits).cpu().numpy().flatten()
    lota_scores = lota_scores[:align_len]  # (6,)
    
    return {
        'velocity': velocity_norm,      # (6,)
        'acceleration': acceleration_norm,  # (6,)
        'lota': lota_scores              # (6,)
    }, True


def extract_all_features(df_real, df_fake, args, d3_model, lota_model, device, d3_transform):
    """提取所有视频的特征"""
    all_features = []
    all_labels = []
    skipped_real = 0
    skipped_fake = 0
    
    print("\n提取真实视频特征...")
    for idx, row in tqdm(df_real.iterrows(), total=len(df_real), desc="真实视频"):
        video_path = row['content_path']
        features, success = extract_sequence_features(
            video_path, args, d3_model, lota_model, device, d3_transform
        )
        if success and features is not None:
            all_features.append(features)
            all_labels.append(0)
        else:
            skipped_real += 1
    
    print("\n提取AI视频特征...")
    for idx, row in tqdm(df_fake.iterrows(), total=len(df_fake), desc="AI视频"):
        video_path = row['content_path']
        features, success = extract_sequence_features(
            video_path, args, d3_model, lota_model, device, d3_transform
        )
        if success and features is not None:
            all_features.append(features)
            all_labels.append(1)
        else:
            skipped_fake += 1
    
    print(f"\n特征提取完成: 成功={len(all_features)} (真实={all_labels.count(0)}, AI={all_labels.count(1)})")
    print(f"跳过: 真实={skipped_real}, AI={skipped_fake}")
    
    return all_features, all_labels


def build_sequences_from_features(features_list, use_velocity, use_acceleration, use_lota):
    """构建 6×3 序列"""
    sequences = []
    for feat_dict in features_list:
        seq_len = len(feat_dict['velocity'])
        seq = []
        for t in range(seq_len):
            feat = []
            if use_velocity:
                feat.append(feat_dict['velocity'][t])
            if use_acceleration:
                feat.append(feat_dict['acceleration'][t])
            if use_lota:
                feat.append(feat_dict['lota'][t])
            seq.append(feat)
        sequences.append(np.array(seq, dtype=np.float32))
    return sequences


def load_model_and_weights(model_dir, config_name, device):
    """加载模型权重（6×3版本）"""
    from LASTONE import TemporalFusionNet
    
    model_name_map = {
        'full': 'vel_acc_lota.pth',
        'motion_only': 'vel_acc.pth',
        'vel_lota': 'vel_lota.pth',
        'acc_lota': 'acc_lota.pth',
        'lota_only': 'lota.pth',
    }
    
    # 6×3 版本：速度1 + 加速度1 + LOTA1 = 3
    input_dim_map = {
        'full': 3,
        'motion_only': 2,
        'vel_lota': 2,
        'acc_lota': 2,
        'lota_only': 1,
    }
    
    input_dim = input_dim_map.get(config_name, 3)
    model_file = model_name_map.get(config_name, f"{config_name}.pth")
    model_path = os.path.join(model_dir, model_file)
    
    model = TemporalFusionNet(
        input_dim=input_dim,
        hidden_dim=64,
        kernel_size=3,
        dropout=0.3
    ).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 获取模型状态字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤掉不在模型中的键
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                               if k in model_state_dict and v.shape == model_state_dict[k].shape}
        
        # 加载过滤后的权重
        model.load_state_dict(filtered_state_dict, strict=False)
        
        best_threshold = checkpoint.get('best_threshold', 0.5)
        print(f"  ✅ 加载 {config_name} 模型: {model_file} (阈值: {best_threshold:.4f})")
        return model, best_threshold
    else:
        print(f"  ❌ 模型不存在: {model_path}")
        return None, 0.5


def test_with_features(features_list, labels, model, config_name, device, threshold=0.5):
    """使用已提取的特征进行测试"""
    use_velocity = True
    use_acceleration = True
    use_lota = True
    
    if config_name == 'motion_only':
        use_lota = False
    elif config_name == 'vel_lota':
        use_acceleration = False
    elif config_name == 'acc_lota':
        use_velocity = False
    elif config_name == 'lota_only':
        use_velocity = False
        use_acceleration = False
        use_lota = True
    
    sequences = build_sequences_from_features(features_list, use_velocity, use_acceleration, use_lota)
    
    X = np.array(sequences)
    y_true = np.array(labels)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        logits = model(X_tensor)
        y_scores = torch.sigmoid(logits).cpu().numpy().flatten()
    
    metrics = compute_metrics(y_true, y_scores, threshold=threshold)
    
    return {
        'config_name': config_name,
        'threshold': threshold,
        'n_total': len(sequences),
        'ap': metrics['ap'],
        'auc': metrics['auc'],
        'acc': metrics['acc'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'tp': metrics['tp'], 'tn': metrics['tn'], 'fp': metrics['fp'], 'fn': metrics['fn']
    }


def save_results(all_results, output_dir, method_name="Ours"):
    """保存结果表格"""
    if not all_results:
        return
    
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj
    
    all_results_native = convert_to_native(all_results)
    
    config_order = ['full', 'motion_only', 'vel_lota', 'acc_lota', 'lota_only']
    config_names = [c for c in config_order if c in [r['config_name'] for r in all_results_native]]
    
    config_display = {
        'full': 'Full',
        'motion_only': 'Motion Only',
        'vel_lota': 'Velocity + LOTA',
        'acc_lota': 'Acceleration + LOTA',
        'lota_only': 'LOTA Only',
    }
    
    dataset_names = sorted(set(r['test_name'] for r in all_results_native))
    
    data = {}
    for r in all_results_native:
        config_name = r['config_name']
        dataset_name = r['test_name']
        if config_name not in data:
            data[config_name] = {}
        data[config_name][dataset_name] = (r['ap'], r['auc'], r['acc'])
    
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    csv_dir = os.path.join(tables_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    combined_df = pd.DataFrame({'Config': [config_display.get(c, c) for c in config_names]})
    for ds in dataset_names:
        values = []
        for cfg in config_names:
            if ds in data[cfg]:
                ap, auc, acc = data[cfg][ds]
                values.append(f"{ap:.4f} / {auc:.4f} / {acc:.4f}")
            else:
                values.append("- / - / -")
        combined_df[ds] = values
    combined_df.to_csv(os.path.join(csv_dir, "combined_table.csv"), index=False)
    
    latex_path = os.path.join(tables_dir, "paper_table.tex")
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("% 消融实验结果\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{" + method_name + "消融实验结果}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{l" + "c" * len(dataset_names) + "}\n")
        f.write("\\hline\n")
        header = "Method & " + " & ".join(dataset_names) + " \\\\\n"
        f.write(header)
        f.write("\\hline\n")
        
        for cfg in config_names:
            row = config_display.get(cfg, cfg)
            for ds in dataset_names:
                if ds in data[cfg]:
                    ap, auc, acc = data[cfg][ds]
                    row += f" & {ap:.4f} / {auc:.4f} / {acc:.4f}"
                else:
                    row += " & - / - / -"
            row += " \\\\\n"
            f.write(row)
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    md_path = os.path.join(tables_dir, "paper_table.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {method_name} 消融实验结果\n\n")
        header = "| Method | " + " | ".join(dataset_names) + " |\n"
        separator = "|" + "---|" * (1 + len(dataset_names)) + "\n"
        f.write(header)
        f.write(separator)
        
        for cfg in config_names:
            row = f"| {config_display.get(cfg, cfg)} "
            for ds in dataset_names:
                if ds in data[cfg]:
                    ap, auc, acc = data[cfg][ds]
                    row += f" | {ap:.4f} / {auc:.4f} / {acc:.4f}"
                else:
                    row += " | - / - / -"
            row += " |\n"
            f.write(row)
    
    avg_results = []
    for cfg in config_names:
        ap_vals = []
        auc_vals = []
        acc_vals = []
        for ds in dataset_names:
            if ds in data[cfg]:
                ap, auc, acc = data[cfg][ds]
                ap_vals.append(ap)
                auc_vals.append(auc)
                acc_vals.append(acc)
        if ap_vals:
            avg_results.append({
                'Config': config_display.get(cfg, cfg),
                'mAP': np.mean(ap_vals),
                'mAUC': np.mean(auc_vals),
                'mACC': np.mean(acc_vals)
            })
    
    avg_df = pd.DataFrame(avg_results)
    avg_df.to_csv(os.path.join(tables_dir, "average_metrics.csv"), index=False)
    
    json_path = os.path.join(tables_dir, "all_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results_native, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果保存至: {tables_dir}")
    print(f"  - 合并表格CSV: {os.path.join(csv_dir, 'combined_table.csv')}")
    print(f"  - LaTeX表格: {latex_path}")
    print(f"  - Markdown表格: {md_path}")
    print(f"  - 平均指标: {os.path.join(tables_dir, 'average_metrics.csv')}")
    
    return tables_dir


def main():
    parser = argparse.ArgumentParser(description='消融实验评估程序')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--loss', default='l2', choices=['l2', 'cos'])
    parser.add_argument('--encoder', default='XCLIP-16')
    parser.add_argument('--method-name', default='Ours')
    
    parser.add_argument('--real-csv', required=True)
    parser.add_argument('--fake-csv', help='与--csv-dir二选一')
    parser.add_argument('--csv-dir', help='目录下所有CSV作为AI视频')
    
    parser.add_argument('--model-dir', required=True, help='训练保存的模型目录')
    
    parser.add_argument('--lota-model', required=True)
    
    parser.add_argument('--output-dir', default='ablation_results')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--balance', action='store_true', default=True)
    parser.add_argument('--target-len', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    if not args.fake_csv and not args.csv_dir:
        parser.error("必须提供 --fake-csv 或 --csv-dir")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.device == 'cuda' and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = torch.device('cuda')
        print(f"使用设备: CUDA (GPU {args.gpu_id})")
    else:
        device = torch.device('cpu')
        print("使用设备: CPU")
    
    from models import D3_model
    d3_model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device)
    d3_model.eval()
    for param in d3_model.parameters():
        param.requires_grad = False
    
    lota_model = load_lota_model(args.lota_model, device)
    if lota_model is None:
        print("LOTA 模型加载失败，退出。")
        return
    
    d3_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    if args.csv_dir:
        fake_csv_files = find_all_csv_files(args.csv_dir)
    else:
        fake_csv_files = [args.fake_csv]
    
    print(f"\n找到 {len(fake_csv_files)} 个AI视频CSV文件")
    
    ablation_configs = [
        {'name': 'full', 'use_velocity': True, 'use_acceleration': True, 'use_lota': True},
        {'name': 'motion_only', 'use_velocity': True, 'use_acceleration': True, 'use_lota': False},
        {'name': 'vel_lota', 'use_velocity': True, 'use_acceleration': False, 'use_lota': True},
        {'name': 'acc_lota', 'use_velocity': False, 'use_acceleration': True, 'use_lota': True},
        {'name': 'lota_only', 'use_velocity': False, 'use_acceleration': False, 'use_lota': True},
    ]
    
    print(f"\n📂 加载模型权重从: {args.model_dir}")
    models = {}
    thresholds = {}
    for cfg in ablation_configs:
        model, thresh = load_model_and_weights(args.model_dir, cfg['name'], device)
        if model:
            models[cfg['name']] = model
            thresholds[cfg['name']] = thresh
    
    if not models:
        print("❌ 没有成功加载任何模型")
        return
    
    print(f"\n✅ 成功加载 {len(models)} 个模型: {list(models.keys())}")
    
    all_results = []
    
    for fake_csv in fake_csv_files:
        fake_name = os.path.splitext(os.path.basename(fake_csv))[0]
        print(f"\n{'='*70}")
        print(f"处理数据集: {fake_name}")
        print(f"{'='*70}")
        
        df_real = pd.read_csv(args.real_csv)
        df_fake = pd.read_csv(fake_csv)
        
        print(f"原始: 真实={len(df_real)}, AI={len(df_fake)}")
        
        if args.balance:
            min_samples = min(len(df_real), len(df_fake))
            df_real = df_real.sample(n=min_samples, random_state=args.seed)
            df_fake = df_fake.sample(n=min_samples, random_state=args.seed)
        elif args.sample:
            if args.sample < len(df_real):
                df_real = df_real.sample(n=args.sample, random_state=args.seed)
            if args.sample < len(df_fake):
                df_fake = df_fake.sample(n=args.sample, random_state=args.seed)
        
        print(f"测试: 真实={len(df_real)}, AI={len(df_fake)}")
        print("开始提取特征...")
        
        features_list, labels = extract_all_features(
            df_real, df_fake, args, d3_model, lota_model, device, d3_transform
        )
        
        print(f"特征提取完成，有效样本数: {len(features_list)}")
        
        if len(features_list) == 0:
            print(f"⚠️ 跳过 {fake_name}: 无有效特征")
            continue
        
        for config_name in ['full', 'motion_only', 'vel_lota', 'acc_lota', 'lota_only']:
            if config_name not in models:
                continue
            
            print(f"\n测试配置: {config_name}")
            result = test_with_features(
                features_list, labels, models[config_name], config_name, device, 
                threshold=thresholds.get(config_name, 0.5)
            )
            result['test_name'] = fake_name
            all_results.append(result)
            
            print(f"  AP: {result['ap']:.4f}, AUC: {result['auc']:.4f}, ACC: {result['acc']:.4f}, F1: {result['f1']:.4f}")
    
    if all_results:
        save_results(all_results, args.output_dir, args.method_name)
        print(f"\n✅ 完成！结果保存至: {args.output_dir}")
    else:
        print("❌ 没有成功处理任何测试")


if __name__ == '__main__':
    main()