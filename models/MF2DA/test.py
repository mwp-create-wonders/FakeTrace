import os
import csv
import time
import random
import argparse
import types

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from io import BytesIO
from scipy.ndimage import gaussian_filter

from models import get_model
from validate import validate
from copy import deepcopy
from utils import collect_efficiency_metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 0


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}


def parse_dirs(dir_input):
    if dir_input is None:
        return []

    if isinstance(dir_input, (list, tuple)):
        return [str(x).strip() for x in dir_input if str(x).strip()]

    if isinstance(dir_input, str):
        return [x.strip() for x in dir_input.split(',') if x.strip()]

    raise TypeError(f"Unsupported dir_input type: {type(dir_input)}")


def collect_images_from_dirs(dir_list, max_num=None, exts=None):
    """
    从多个目录递归收集图片。
    """
    if exts is None:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    all_paths = []
    for folder in dir_list:
        if not os.path.isdir(folder):
            print(f"[Warning] directory does not exist: {folder}")
            continue

        for root, _, files in os.walk(folder):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    all_paths.append(os.path.join(root, name))

    random.shuffle(all_paths)

    if max_num is not None and max_num > 0:
        all_paths = all_paths[:min(max_num, len(all_paths))]

    return all_paths


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out).convert("RGB")
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
    return Image.fromarray(img)


class RealFakeDataset(Dataset):
    def __init__(self, opt, jpeg_quality=None, gaussian_sigma=None):
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        real_dirs = parse_dirs(opt.real_dirs)
        fake_dirs = parse_dirs(opt.fake_dirs)

        if len(real_dirs) == 0:
            raise ValueError("real_dirs is empty.")
        if len(fake_dirs) == 0:
            raise ValueError("fake_dirs is empty.")

        real_num = opt.real_num if opt.real_num > 0 else None
        fake_num = opt.fake_num if opt.fake_num > 0 else None

        real_list = collect_images_from_dirs(real_dirs, max_num=real_num)
        fake_list = collect_images_from_dirs(fake_dirs, max_num=fake_num)

        if len(real_list) == 0:
            raise ValueError("No real images found in real_dirs.")
        if len(fake_list) == 0:
            raise ValueError("No fake images found in fake_dirs.")

        print(f"[Test Dataset] real dirs: {len(real_dirs)}, real images: {len(real_list)}")
        print(f"[Test Dataset] fake dirs: {len(fake_dirs)}, fake images: {len(fake_list)}")

        self.total_list = real_list + fake_list
        self.labels_dict = {}

        for p in real_list:
            self.labels_dict[p] = 0
        for p in fake_list:
            self.labels_dict[p] = 1

        random.shuffle(self.total_list)
        self.targets = [self.labels_dict[p] for p in self.total_list]

        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"
        print("mean and std stats are from:", stat_from)

        transform_list = []
        if not opt.no_resize:
            transform_list.append(transforms.Resize((opt.loadSize, opt.loadSize)))
        if not opt.no_crop:
            transform_list.append(transforms.CenterCrop(opt.cropSize))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]

        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label


def create_dataloader(opt):
    dataset = RealFakeDataset(opt)

    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_threads,
        pin_memory=opt.pin_memory
    )
    return data_loader


def _infer_fc_from_weights(model, model_weights, device):
    if 'fc.0.weight' not in model_weights or 'fc.3.weight' not in model_weights:
        return

    fc0_weight = model_weights['fc.0.weight']
    fc3_weight = model_weights['fc.3.weight']

    input_dim = fc0_weight.shape[1]
    hidden_dim = fc0_weight.shape[0]
    output_dim = fc3_weight.shape[0]

    print(f"Inferred FC shape: {input_dim} -> {hidden_dim} -> {output_dim}")

    def init_fc_custom(this_model, in_dim, hid_dim, out_dim):
        this_model.feature_dim = in_dim
        this_model.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, out_dim)
        )
        this_model.fc = this_model.fc.to(device)

    model._init_fc = lambda in_dim, hid_dim, out_dim: init_fc_custom(model, in_dim, hid_dim, out_dim)
    model._init_fc(input_dim, hidden_dim, output_dim)


def enable_resnet_only_inference(model):
    if not hasattr(model, "model"):
        raise ValueError("The current model wrapper does not expose an inner dual-branch model.")

    dual_branch_model = model.model
    required_attrs = ["resnet_branch", "fusion_network", "clip_branch"]
    missing_attrs = [name for name in required_attrs if not hasattr(dual_branch_model, name)]
    if missing_attrs:
        raise ValueError(
            f"ResNet-only inference requires a dual-branch CLIP model, but missing: {missing_attrs}"
        )

    clip_visual_dim = dual_branch_model.clip_branch.visual.output_dim
    text_feature_dim = dual_branch_model.clip_branch.text_projection.shape[1]

    def forward_resnet_only(self, x, return_feature=False):
        resnet_features_dict = self.model.resnet_branch(x)
        resnet_features = resnet_features_dict['Resnet_feature']

        batch_size = x.size(0)
        clip_features = torch.zeros(
            batch_size,
            clip_visual_dim,
            device=x.device,
            dtype=resnet_features.dtype
        )
        text_features = torch.zeros(
            batch_size,
            text_feature_dim,
            device=x.device,
            dtype=resnet_features.dtype
        )

        combined_visual_features = torch.cat([clip_features, resnet_features], dim=1)
        fused_features = self.model.fusion_network(combined_visual_features, text_features)

        if self.fc is None:
            self._init_fc(fused_features.shape[1])

        logits = self.fc(fused_features)

        if return_feature:
            feature_dict = {
                'clip_features': clip_features,
                'resnet_features': resnet_features,
                'combined_visual_features': combined_visual_features,
                'text_features': text_features,
                'fused_features': fused_features,
                'logits': logits,
            }
            return logits, feature_dict

        return logits

    model.forward = types.MethodType(forward_resnet_only, model)
    print("Enabled EPF-ResNet-only inference: CLIP-ViT image features are skipped during testing.")
    print("Note: fusion head and classifier weights are still used to keep checkpoint compatibility.")
    return model


def build_model(opt, device):
    model = get_model(opt.arch)

    state_dict = torch.load(opt.ckpt, map_location='cpu', weights_only=False)
    model_weights = state_dict['model'] if 'model' in state_dict else state_dict
    _infer_fc_from_weights(model, model_weights, device)

    # 如果权重里有 fc.*，则按权重推断分类头维度
    if 'fc.0.weight' in model_weights and 'fc.3.weight' in model_weights:
        fc0_weight = model_weights['fc.0.weight']
        fc3_weight = model_weights['fc.3.weight']

        input_dim = fc0_weight.shape[1]
        hidden_dim = fc0_weight.shape[0]
        output_dim = fc3_weight.shape[0]

        print(f"Inferred FC shape: {input_dim} -> {hidden_dim} -> {output_dim}")

        def init_fc_custom(this_model, in_dim, hid_dim, out_dim):
            this_model.feature_dim = in_dim
            this_model.fc = nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hid_dim, out_dim)
            )
            this_model.fc = this_model.fc.to(device)

        model._init_fc = lambda in_dim, hid_dim, out_dim: init_fc_custom(model, in_dim, hid_dim, out_dim)
        model._init_fc(input_dim, hidden_dim, output_dim)

    model.load_state_dict(model_weights, strict=False)
    if getattr(opt, "resnet_only", False):
        if not opt.arch.startswith("CLIP:"):
            raise ValueError("--resnet_only only supports the current dual-branch CLIP architecture.")
        model = enable_resnet_only_inference(model)
    model = model.to(device)
    model.eval()
    return model


def format_metric(value, fmt=".6f"):
    if value is None:
        return "N/A"
    return format(value, fmt)


def select_report_metrics(metrics):
    metrics = metrics or {}
    return {
        "trainable_params_m": metrics.get("trainable_params_m"),
        "gflops": metrics.get("gflops"),
        "inference_time_ms_per_img": metrics.get("inference_time_ms_per_img"),
        "peak_gpu_memory_gb": metrics.get("peak_gpu_memory_gb"),
        "model_size_mb": metrics.get("model_size_mb"),
    }


def save_results(opt, ap, auc, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres, efficiency_metrics=None):
    os.makedirs(opt.output_dir, exist_ok=True)
    efficiency_metrics = select_report_metrics(efficiency_metrics)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    txt_path = os.path.join(opt.output_dir, "test_result.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {opt.arch}\n")
        f.write(f"Weights: {opt.ckpt}\n")
        f.write(f"Real Dirs: {opt.real_dirs}\n")
        f.write(f"Fake Dirs: {opt.fake_dirs}\n")
        f.write(f"Real Num: {opt.real_num}\n")
        f.write(f"Fake Num: {opt.fake_num}\n")
        f.write(f"AP: {ap if ap is None else f'{ap:.6f}'}\n")
        f.write(f"AUC: {auc if auc is None else f'{auc:.6f}'}\n")
        f.write(f"Real Accuracy 0: {r_acc0:.6f}\n")
        f.write(f"Fake Accuracy 0: {f_acc0:.6f}\n")
        f.write(f"Overall Accuracy 0: {acc0:.6f}\n")
        f.write(f"Real Accuracy 1: {r_acc1:.6f}\n")
        f.write(f"Fake Accuracy 1: {f_acc1:.6f}\n")
        f.write(f"Overall Accuracy 1: {acc1:.6f}\n")
        f.write(f"Best Threshold: {best_thres:.6f}\n")
        f.write(f"Trainable Params (M): {format_metric(efficiency_metrics.get('trainable_params_m'))}\n")
        f.write(f"GFLOPs: {format_metric(efficiency_metrics.get('gflops'))}\n")
        f.write(
            f"Inference Time/img (ms): "
            f"{format_metric(efficiency_metrics.get('inference_time_ms_per_img'))}\n"
        )
        f.write(f"Peak GPU Memory (GB): {format_metric(efficiency_metrics.get('peak_gpu_memory_gb'))}\n")
        f.write(f"Model Size (MB): {format_metric(efficiency_metrics.get('model_size_mb'))}\n")

    csv_path = os.path.join(opt.output_dir, "test_results.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "arch", "ckpt", "real_dirs", "fake_dirs",
                "real_num", "fake_num", "ap", "auc",
                "r_acc0", "f_acc0", "acc0",
                "r_acc1", "f_acc1", "acc1", "best_thres",
                "trainable_params_m", "gflops", "inference_time_ms_per_img",
                "peak_gpu_memory_gb", "model_size_mb"
            ])
        writer.writerow([
            timestamp,
            opt.arch,
            opt.ckpt,
            "|".join(parse_dirs(opt.real_dirs)),
            "|".join(parse_dirs(opt.fake_dirs)),
            opt.real_num,
            opt.fake_num,
            "" if ap is None else f"{ap:.6f}",
            "" if auc is None else f"{auc:.6f}",
            f"{r_acc0:.6f}",
            f"{f_acc0:.6f}",
            f"{acc0:.6f}",
            f"{r_acc1:.6f}",
            f"{f_acc1:.6f}",
            f"{acc1:.6f}",
            f"{best_thres:.6f}",
            "" if efficiency_metrics.get("trainable_params_m") is None else f"{efficiency_metrics['trainable_params_m']:.6f}",
            "" if efficiency_metrics.get("gflops") is None else f"{efficiency_metrics['gflops']:.6f}",
            "" if efficiency_metrics.get("inference_time_ms_per_img") is None else f"{efficiency_metrics['inference_time_ms_per_img']:.6f}",
            "" if efficiency_metrics.get("peak_gpu_memory_gb") is None else f"{efficiency_metrics['peak_gpu_memory_gb']:.6f}",
            "" if efficiency_metrics.get("model_size_mb") is None else f"{efficiency_metrics['model_size_mb']:.6f}",
        ])

    print(f"Saved txt result to: {txt_path}")
    print(f"Saved csv result to: {csv_path}")


def print_efficiency_metrics(metrics):
    metrics = select_report_metrics(metrics)
    print("\n===== Efficiency Metrics =====")
    print(f"Trainable Params (M): {format_metric(metrics.get('trainable_params_m'))}")
    print(f"GFLOPs: {format_metric(metrics.get('gflops'))}")
    print(f"Inference Time/img (ms): {format_metric(metrics.get('inference_time_ms_per_img'))}")
    print(f"Peak GPU Memory (GB): {format_metric(metrics.get('peak_gpu_memory_gb'))}")
    print(f"Model Size (MB): {format_metric(metrics.get('model_size_mb'))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test binary real/fake detection with multiple real dirs and multiple fake dirs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--real_dirs', type=str, required=True,
                        help='comma-separated real image directories')
    parser.add_argument('--fake_dirs', type=str, required=True,
                        help='comma-separated fake image directories')

    parser.add_argument('--real_num', type=int, default=500,
                        help='max number of real images, 0 means all')
    parser.add_argument('--fake_num', type=int, default=500,
                        help='max number of fake images, 0 means all')

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14',
                        help='model architecture')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='model checkpoint path')

    parser.add_argument('--output_dir', type=str, default='./result',
                        help='directory to save test results')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for testing')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--pin_memory', action='store_true',
                        help='use pin_memory in dataloader')

    parser.add_argument('--loadSize', type=int, default=256,
                        help='resize image to this size before crop')
    parser.add_argument('--cropSize', type=int, default=224,
                        help='center crop size')
    parser.add_argument('--no_resize', action='store_true',
                        help='do not resize image')
    parser.add_argument('--no_crop', action='store_true',
                        help='do not crop image')

    parser.add_argument('--merge_fake_dirs', action='store_true',
                        help='if set, merge all fake dirs into one test set; otherwise test one fake dir at a time')
    parser.add_argument('--benchmark_warmup', type=int, default=10,
                        help='number of warmup steps for inference benchmarking')
    parser.add_argument('--benchmark_steps', type=int, default=30,
                        help='number of timed steps for inference benchmarking')
    parser.add_argument('--resnet_only', action='store_true',
                        help='load the full checkpoint but skip CLIP-ViT image branch during inference')

    opt = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(opt, device)

    fake_dir_list = parse_dirs(opt.fake_dirs)

    if opt.merge_fake_dirs:
        test_loader = create_dataloader(opt)
        efficiency_metrics = collect_efficiency_metrics(
            model,
            test_loader,
            device=device,
            ckpt_path=opt.ckpt,
            warmup_steps=opt.benchmark_warmup,
            benchmark_steps=opt.benchmark_steps,
        )
        print_efficiency_metrics(efficiency_metrics)
        ap, auc, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(
            model, test_loader, device=device, find_thres=True
        )

        save_results(
            opt,
            ap, auc,
            r_acc0, f_acc0, acc0,
            r_acc1, f_acc1, acc1,
            best_thres,
            efficiency_metrics=efficiency_metrics
        )

        print(f"(Merged Test) acc: {acc0:.4f}; auc: {auc if auc is None else f'{auc:.4f}'}; ap: {ap if ap is None else f'{ap:.4f}'}")

    else:
        for fake_dir in fake_dir_list:
            cur_opt = deepcopy(opt)
            cur_opt.fake_dirs = fake_dir

            fake_name = os.path.basename(os.path.normpath(fake_dir))

            if fake_name in ["0_real", "1_fake", "fake", "real"]:
                fake_name = os.path.basename(os.path.dirname(os.path.normpath(fake_dir)))

            cur_opt.output_dir = os.path.join(opt.output_dir, fake_name)

            print(f"\n===== Testing fake dataset: {fake_dir} =====")

            test_loader = create_dataloader(cur_opt)
            efficiency_metrics = collect_efficiency_metrics(
                model,
                test_loader,
                device=device,
                ckpt_path=opt.ckpt,
                warmup_steps=opt.benchmark_warmup,
                benchmark_steps=opt.benchmark_steps,
            )
            print_efficiency_metrics(efficiency_metrics)
            ap, auc, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(
                model, test_loader, device=device, find_thres=True
            )

            save_results(
                cur_opt,
                ap, auc,
                r_acc0, f_acc0, acc0,
                r_acc1, f_acc1, acc1,
                best_thres,
                efficiency_metrics=efficiency_metrics
            )

            print(f"(Test @ {fake_name}) acc: {acc0:.4f}; auc: {auc if auc is None else f'{auc:.4f}'}; ap: {ap if ap is None else f'{ap:.4f}'}")
