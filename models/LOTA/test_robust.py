import os
import csv
import json
import io
import sys
import argparse
import hashlib
import random
import shutil
from datetime import datetime

import torch
from PIL import Image, ImageFile, ImageFilter
from sklearn.metrics import average_precision_score

from util import set_random_seed as seed_generator
from config import ConfigurationManager as Configurator
from model import model as DeepLearningModel
from loader import get_single_loader as acquire_single_loader


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def parse_robust_args_and_clean_argv():
    """
    单独解析鲁棒性参数，然后从 sys.argv 中移除，
    避免原始 ConfigurationManager 不认识新增参数时报错。
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--name", default="baseline", type=str)
    parser.add_argument(
        "--robust_mode",
        default="none",
        type=str,
        choices=["none", "jpeg", "blur", "resize"]
    )
    parser.add_argument("--jpeg_quality", default=95, type=int)
    parser.add_argument("--blur_radius", default=0.0, type=float)
    parser.add_argument("--resize_scale", default=1.0, type=float)

    parser.add_argument(
        "--robust_csv",
        default=None,
        type=str,
        help="Merged CSV path for all robustness results."
    )
    parser.add_argument(
        "--save_json",
        default=None,
        type=str,
        help="Save current run results to a JSON file."
    )

    parser.add_argument(
        "--robust_cache_root",
        default=None,
        type=str,
        help="Root directory for cached robustness datasets."
    )
    parser.add_argument(
        "--robust_sample_seed",
        default=2026,
        type=int,
        help="Seed for deterministic image sampling."
    )
    parser.add_argument(
        "--rebuild_robust_cache",
        action="store_true",
        help="Rebuild cached robustness images even if cache already exists."
    )

    robust_args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv

    return robust_args


def configure_computation_device(device_id):
    """Set computational hardware environment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"Selected computation device: GPU {device_id}")


def parse_path_list(x):
    if x is None:
        return []

    if isinstance(x, (list, tuple)):
        return [str(v).strip() for v in x if v is not None and str(v).strip()]

    if isinstance(x, str):
        x = x.strip()
        if len(x) == 0:
            return []
        return [p.strip() for p in x.split(",") if p.strip()]

    return []


def collect_images_from_dirs(dir_list):
    image_paths = []

    for root_dir in dir_list:
        if not os.path.isdir(root_dir):
            raise ValueError(f"Directory does not exist: {root_dir}")

        for root, _, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(IMG_EXTENSIONS):
                    image_paths.append(os.path.join(root, fname))

    return sorted(image_paths)


def get_test_num_from_config(config):
    test_num = getattr(config, "test_num", 0)
    try:
        test_num = int(test_num)
    except Exception:
        test_num = 0
    return max(0, test_num)


def get_robust_value(robust_args):
    if robust_args.robust_mode == "jpeg":
        return robust_args.jpeg_quality
    if robust_args.robust_mode == "blur":
        return robust_args.blur_radius
    if robust_args.robust_mode == "resize":
        return robust_args.resize_scale
    return None


def safe_condition_key(robust_args):
    if robust_args.robust_mode == "jpeg":
        return f"jpeg_q{robust_args.jpeg_quality}"
    if robust_args.robust_mode == "blur":
        return f"blur_r{robust_args.blur_radius}"
    if robust_args.robust_mode == "resize":
        return f"resize_s{robust_args.resize_scale}"
    return "baseline"


def safe_dataset_key(dir_list):
    """
    为一个或多个输入目录生成稳定缓存名。
    防止不同目录 basename 相同导致缓存冲突。
    """
    dir_list = parse_path_list(dir_list)
    joined = "|".join([os.path.abspath(x).rstrip("/") for x in dir_list])

    if len(dir_list) == 1:
        norm = os.path.abspath(dir_list[0]).rstrip("/")
        tail = "_".join(norm.split(os.sep)[-5:])
    else:
        tail = f"multi_{len(dir_list)}_dirs"

    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()[:10]
    safe_tail = "".join([c if c.isalnum() or c in "._-" else "_" for c in tail])

    return f"{safe_tail}_{digest}"


def deterministic_sample_paths(image_paths, sample_num, seed, source_key):
    """
    稳定随机采样。
    同一个 source_key + sample_num + seed 会得到同一批图像。
    robust_mode 不参与采样 seed，保证所有扰动强度使用同一批样本。
    """
    if sample_num is None or sample_num <= 0:
        return image_paths

    if len(image_paths) <= sample_num:
        print(
            f"[WARN] 样本不足，期望 {sample_num} 张，实际只有 {len(image_paths)} 张，改为取全部。"
        )
        return image_paths

    seed_key = f"{source_key}::{sample_num}::{seed}"
    local_seed = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest()[:8], 16)

    rng = random.Random(local_seed)
    sampled = image_paths[:]
    rng.shuffle(sampled)
    return sampled[:sample_num]


def cache_is_ready(cache_dir):
    manifest_path = os.path.join(cache_dir, "manifest.json")
    return os.path.isdir(cache_dir) and os.path.isfile(manifest_path)


def apply_jpeg(img, quality):
    """
    JPEG recompression.
    quality=100 仍然会经历 JPEG encode/decode，只是扰动很轻。
    subsampling=0 表示 4:4:4，避免额外色度子采样。
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=0
    )
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_blur(img, radius):
    radius = float(radius)
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_resize_recover(img, scale):
    scale = float(scale)
    if abs(scale - 1.0) < 1e-8:
        return img

    w, h = img.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    img = img.resize((w, h), resample=Image.BICUBIC)
    return img


def apply_robust_transform(src_path, dst_path, robust_args):
    with Image.open(src_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        if robust_args.robust_mode == "jpeg":
            img = apply_jpeg(img, robust_args.jpeg_quality)
        elif robust_args.robust_mode == "blur":
            img = apply_blur(img, robust_args.blur_radius)
        elif robust_args.robust_mode == "resize":
            img = apply_resize_recover(img, robust_args.resize_scale)
        elif robust_args.robust_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported robust_mode: {robust_args.robust_mode}")

        # 统一保存为 PNG，避免源文件格式不一致造成保存失败。
        img.save(dst_path, format="PNG")


def build_cached_dir_for_dirs(dir_list, robust_args, cache_root, prefix, sample_num=0):
    """
    对一个或多个目录进行稳定采样，并构建/复用鲁棒性缓存目录。

    返回值是一个 list，便于直接传给 acquire_single_loader。
    """
    dir_list = parse_path_list(dir_list)
    if len(dir_list) == 0:
        raise ValueError("Empty dir_list for cache building.")

    source_key = "|".join([os.path.abspath(x).rstrip("/") for x in dir_list])
    dataset_key = safe_dataset_key(dir_list)
    condition_key = safe_condition_key(robust_args)

    cache_dir = os.path.join(
        cache_root,
        condition_key,
        f"{prefix}_{dataset_key}_n{sample_num}_seed{robust_args.robust_sample_seed}"
    )

    if cache_is_ready(cache_dir) and not robust_args.rebuild_robust_cache:
        print(f"[Cache] Reuse cached dir: {cache_dir}")
        return [cache_dir]

    if os.path.exists(cache_dir) and robust_args.rebuild_robust_cache:
        print(f"[Cache] Rebuild enabled, removing old cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.makedirs(cache_dir, exist_ok=True)

    image_paths = collect_images_from_dirs(dir_list)
    selected_paths = deterministic_sample_paths(
        image_paths=image_paths,
        sample_num=sample_num,
        seed=robust_args.robust_sample_seed,
        source_key=source_key
    )

    print(
        f"[Cache] Building cache | prefix: {prefix} | condition: {condition_key} | "
        f"selected: {len(selected_paths)} | cache: {cache_dir}"
    )

    manifest = {
        "dir_list": dir_list,
        "cache_dir": cache_dir,
        "prefix": prefix,
        "condition": condition_key,
        "robust_mode": robust_args.robust_mode,
        "robust_value": get_robust_value(robust_args),
        "sample_num": sample_num,
        "seed": robust_args.robust_sample_seed,
        "num_selected": len(selected_paths),
        "selected_files": selected_paths,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for idx, src_path in enumerate(selected_paths):
        base_name = os.path.basename(src_path)
        name_wo_ext = os.path.splitext(base_name)[0]

        if robust_args.robust_mode == "none":
            dst_name = f"{idx:08d}_{base_name}"
            dst_path = os.path.join(cache_dir, dst_name)

            try:
                os.symlink(os.path.abspath(src_path), dst_path)
            except Exception:
                shutil.copy2(src_path, dst_path)
        else:
            dst_name = f"{idx:08d}_{name_wo_ext}.png"
            dst_path = os.path.join(cache_dir, dst_name)
            apply_robust_transform(src_path, dst_path, robust_args)

    manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[Cache] Finished cache: {cache_dir}")
    return [cache_dir]


def evaluate_one_fake_dataset(test_loader, neural_network, fake_name="unknown"):
    """
    在固定 real 数据集 + 当前 fake 数据集 上进行一次评估。

    标签约定：
        real = 1
        fake = 0
    """
    neural_network.eval()

    total_correct = 0
    total_samples = 0

    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for image_batch, target_labels in test_loader:
            image_batch = image_batch.cuda(non_blocking=True)
            target_labels = target_labels.cuda(non_blocking=True).float()

            logits = neural_network(image_batch).flatten()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct = (preds == target_labels).float()

            total_correct += correct.sum().item()
            total_samples += target_labels.numel()

            real_mask = (target_labels == 1)
            fake_mask = (target_labels == 0)

            if real_mask.any():
                real_correct += (preds[real_mask] == target_labels[real_mask]).float().sum().item()
                real_total += real_mask.sum().item()

            if fake_mask.any():
                fake_correct += (preds[fake_mask] == target_labels[fake_mask]).float().sum().item()
                fake_total += fake_mask.sum().item()

            all_targets.extend(target_labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    real_acc = real_correct / real_total if real_total > 0 else 0.0
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0.0

    # real=1 作为正类 AP
    real_ap = average_precision_score(all_targets, all_probs) if len(set(all_targets)) > 1 else None

    # fake=1 作为正类 AP
    fake_targets = [1 - int(x) for x in all_targets]
    fake_probs = [1.0 - float(x) for x in all_probs]
    fake_ap = average_precision_score(fake_targets, fake_probs) if len(set(fake_targets)) > 1 else None

    print(f"\n[Evaluation Result] Fake Dataset: {fake_name}")
    print(f"  Overall Accuracy : {overall_acc:.4f}")
    print(f"  Real Accuracy    : {real_acc:.4f}")
    print(f"  Fake Accuracy    : {fake_acc:.4f}")
    print(f"  Real AP          : {real_ap:.4f}" if real_ap is not None else "  Real AP          : None")
    print(f"  Fake AP          : {fake_ap:.4f}" if fake_ap is not None else "  Fake AP          : None")

    return {
        "fake_dataset": fake_name,
        "overall_acc": overall_acc,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "real_ap": real_ap,
        "fake_ap": fake_ap,
        "num_real": real_total,
        "num_fake": fake_total,
        "num_total": total_samples,
    }


def save_results_to_csv(results, save_csv_path):
    """
    Append all evaluation results into one CSV file.
    """
    csv_dir = os.path.dirname(save_csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    fieldnames = [
        "timestamp",
        "name",
        "robust_mode",
        "robust_value",
        "jpeg_quality",
        "blur_radius",
        "resize_scale",
        "fake_dataset",
        "fake_dir",
        "actual_real_dirs",
        "actual_fake_dirs",
        "test_num",
        "cache_root",
        "overall_acc",
        "real_acc",
        "fake_acc",
        "real_ap",
        "fake_ap",
        "num_real",
        "num_fake",
        "num_total",
    ]

    file_exists = os.path.isfile(save_csv_path)

    with open(save_csv_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"\nSaved evaluation results to: {save_csv_path}")


def save_results_to_json(results, save_json_path):
    if save_json_path is None or len(str(save_json_path).strip()) == 0:
        return

    save_json_path = str(save_json_path).strip()
    json_dir = os.path.dirname(save_json_path)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON results to: {save_json_path}")


def execute_evaluation_procedure():
    """
    Main evaluation workflow.
    """
    seed_generator()

    robust_args = parse_robust_args_and_clean_argv()

    config = Configurator().parse()
    config.isTrain = False
    config.isVal = True

    configure_computation_device(config.gpu_id)

    if not hasattr(config, "test_real_dirs") or len(config.test_real_dirs) == 0:
        raise ValueError("Please provide --test_real_dirs")
    if not hasattr(config, "test_fake_dirs") or len(config.test_fake_dirs) == 0:
        raise ValueError("Please provide --test_fake_dirs")

    if robust_args.robust_cache_root is None or len(str(robust_args.robust_cache_root).strip()) == 0:
        robust_args.robust_cache_root = os.path.join(config.save_path, "robust_cache")

    robust_args.robust_cache_root = str(robust_args.robust_cache_root).strip()
    os.makedirs(robust_args.robust_cache_root, exist_ok=True)

    if robust_args.robust_csv is None or len(str(robust_args.robust_csv).strip()) == 0:
        robust_args.robust_csv = os.path.join(config.save_path, "robust_results.csv")

    print("=" * 100)
    print("Robustness Evaluation Settings")
    print(f"Name                 : {robust_args.name}")
    print(f"Robust mode          : {robust_args.robust_mode}")
    print(f"Robust value         : {get_robust_value(robust_args)}")
    print(f"JPEG quality         : {robust_args.jpeg_quality}")
    print(f"Blur radius          : {robust_args.blur_radius}")
    print(f"Resize scale         : {robust_args.resize_scale}")
    print(f"Robust CSV           : {robust_args.robust_csv}")
    print(f"Save JSON            : {robust_args.save_json}")
    print(f"Robust cache root    : {robust_args.robust_cache_root}")
    print(f"Robust sample seed   : {robust_args.robust_sample_seed}")
    print(f"Rebuild robust cache : {robust_args.rebuild_robust_cache}")
    print("=" * 100)

    network_instance = DeepLearningModel().cuda()

    if config.load is not None:
        network_instance.load_state_dict(torch.load(config.load))
        print(f"Loaded model parameters from {config.load}")
    else:
        raise ValueError("Please provide model weights using --load")

    results_path = config.save_path
    os.makedirs(results_path, exist_ok=True)

    all_results = []
    test_num = get_test_num_from_config(config)

    actual_real_dirs = build_cached_dir_for_dirs(
        dir_list=config.test_real_dirs,
        robust_args=robust_args,
        cache_root=robust_args.robust_cache_root,
        prefix="real",
        sample_num=test_num
    )

    for fake_dir in config.test_fake_dirs:
        fake_name = os.path.basename(os.path.normpath(fake_dir))

        print("\n" + "=" * 80)
        print(f"Testing fake dataset: {fake_dir}")
        print(f"Using at most {test_num} real samples and {test_num} fake samples")
        print(f"Robust mode: {robust_args.robust_mode}")
        print("=" * 80)

        actual_fake_dirs = build_cached_dir_for_dirs(
            dir_list=[fake_dir],
            robust_args=robust_args,
            cache_root=robust_args.robust_cache_root,
            prefix="fake",
            sample_num=test_num
        )

        test_loader, dataset_size = acquire_single_loader(
            config,
            real_dirs=actual_real_dirs,
            fake_dirs=actual_fake_dirs
        )

        result = evaluate_one_fake_dataset(
            test_loader=test_loader,
            neural_network=network_instance,
            fake_name=fake_name
        )

        result.update({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": robust_args.name,
            "robust_mode": robust_args.robust_mode,
            "robust_value": get_robust_value(robust_args),
            "jpeg_quality": robust_args.jpeg_quality,
            "blur_radius": robust_args.blur_radius,
            "resize_scale": robust_args.resize_scale,
            "fake_dir": fake_dir,
            "actual_real_dirs": " | ".join(actual_real_dirs),
            "actual_fake_dirs": " | ".join(actual_fake_dirs),
            "test_num": test_num,
            "cache_root": robust_args.robust_cache_root,
        })

        all_results.append(result)

    save_results_to_csv(all_results, robust_args.robust_csv)
    save_results_to_json(all_results, robust_args.save_json)

    print("\nAll evaluations completed.")


if __name__ == '__main__':
    execute_evaluation_procedure()