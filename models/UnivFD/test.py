import argparse
import csv
import os
import random
from collections import OrderedDict
from io import BytesIO
from time import perf_counter

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

from models import get_model
from data.datasets import collect_images_from_dirs

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 0

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}
STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_dir_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = s.strip()
    if s == "":
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def parse_int_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return [int(x) for x in s]
    s = s.strip()
    if s == "":
        return None
    return [int(item.strip()) for item in s.split(",") if item.strip()]


def find_best_threshold(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    best_acc = 0.0
    best_thres = 0.5
    for thres in np.unique(y_pred):
        pred_bin = (y_pred >= thres).astype(np.int32)
        acc = (pred_bin == y_true).mean()
        if acc >= best_acc:
            best_acc = acc
            best_thres = float(thres)
    return best_thres


def calculate_acc(y_true, y_pred, thres):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_bin = (y_pred >= thres).astype(np.int32)

    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)

    r_acc = accuracy_score(y_true[real_mask], y_bin[real_mask]) if real_mask.sum() > 0 else 0.0
    f_acc = accuracy_score(y_true[fake_mask], y_bin[fake_mask]) if fake_mask.sum() > 0 else 0.0
    acc = accuracy_score(y_true, y_bin)
    return r_acc, f_acc, acc


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)
    out.seek(0)
    img = Image.open(out).convert("RGB")
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def apply_gaussian_blur(img, sigma):
    img = np.array(img).copy()
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
    return Image.fromarray(img)


class EvalRealFakeDataset(Dataset):
    def __init__(
        self,
        real_dirs,
        fake_dirs,
        arch,
        real_num_per_dir=None,
        fake_num_per_dir=None,
        jpeg_quality=None,
        gaussian_sigma=None,
        image_size=224,
    ):
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        real_list = collect_images_from_dirs(real_dirs, real_num_per_dir)
        fake_list = collect_images_from_dirs(fake_dirs, fake_num_per_dir)

        if len(real_list) == 0:
            raise ValueError("real_list is empty.")
        if len(fake_list) == 0:
            raise ValueError("fake_list is empty.")

        self.total_list = real_list + fake_list
        self.labels_dict = {p: 0 for p in real_list}
        self.labels_dict.update({p: 1 for p in fake_list})

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

        print(f"[Dataset] real: {len(real_list)}, fake: {len(fake_list)}, total: {len(self.total_list)}")

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        for _ in range(10):
            img_path = self.total_list[idx]
            label = self.labels_dict[img_path]
            try:
                img = Image.open(img_path).convert("RGB")

                if self.gaussian_sigma is not None:
                    img = apply_gaussian_blur(img, self.gaussian_sigma)

                if self.jpeg_quality is not None:
                    img = png2jpg(img, self.jpeg_quality)

                img = self.transform(img)
                return img, label
            except Exception as e:
                print(f"[Warning] Failed to read image: {img_path}, error: {e}")
                idx = np.random.randint(0, len(self.total_list))

        raise RuntimeError("Too many failed image reads.")


@torch.no_grad()
def validate(model, loader, device=None, find_thres=False):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    y_true, y_pred = [], []

    print("Length of dataloader: %d" % len(loader))

    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(img)

        if logits.ndim == 1:
            prob = torch.sigmoid(logits)
        elif logits.ndim == 2 and logits.shape[1] == 1:
            prob = torch.sigmoid(logits[:, 0])
        elif logits.ndim == 2 and logits.shape[1] == 2:
            prob = torch.softmax(logits, dim=1)[:, 1]
        else:
            raise ValueError(f"Unsupported model output shape: {logits.shape}")

        y_pred.extend(prob.detach().cpu().numpy().tolist())
        y_true.extend(label.detach().cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auroc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)

    if not find_thres:
        return auroc, ap, r_acc0, f_acc0, acc0

    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return auroc, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


def load_checkpoint_to_model(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')

    if isinstance(state_dict, dict) and 'model' in state_dict:
        inner = state_dict['model']

        filtered_state_dict = OrderedDict()
        if isinstance(inner, dict) and 'fc.weight' in inner:
            filtered_state_dict['weight'] = inner['fc.weight']
        if isinstance(inner, dict) and 'fc.bias' in inner:
            filtered_state_dict['bias'] = inner['fc.bias']

        if len(filtered_state_dict) > 0 and hasattr(model, 'fc'):
            model.fc.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded fc layer from checkpoint.")
            return model

        missing, unexpected = model.load_state_dict(inner, strict=False)
        print("Loaded checkpoint from state_dict['model'].")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        return model

    if isinstance(state_dict, dict):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint from raw state_dict.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        return model

    raise ValueError("Unsupported checkpoint format.")


def get_trainable_params_m(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params / 1e6


def get_model_size_mb(model):
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 ** 2)
    buffer.close()
    return size_mb


def get_gflops(model, example_input, device):
    if not hasattr(torch, "profiler"):
        return None

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_flops=True,
            ) as prof:
                model(example_input)
        total_flops = 0
        for event in prof.key_averages():
            event_flops = getattr(event, "flops", 0)
            if event_flops is not None:
                total_flops += event_flops
        if total_flops <= 0:
            return None
        return total_flops / 1e9
    except Exception as e:
        print(f"[Warning] Failed to compute FLOPs: {e}")
        return None
    finally:
        model.train(was_training)


@torch.no_grad()
def benchmark_model(model, device, image_size, warmup_iters=10, bench_iters=30):
    example_input = torch.randn(1, 3, image_size, image_size, device=device)
    was_training = model.training
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    try:
        for _ in range(warmup_iters):
            model(example_input)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = perf_counter()
        for _ in range(bench_iters):
            model(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start_time

        inference_time_ms = elapsed * 1000.0 / bench_iters
        peak_gpu_memory_mb = None
        if device.type == "cuda":
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        gflops = get_gflops(model, example_input, device)

        return {
            "trainable_params_m": get_trainable_params_m(model),
            "gflops": gflops,
            "inference_time_ms": inference_time_ms,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "model_size_mb": get_model_size_mb(model),
        }
    finally:
        model.train(was_training)


def build_dataset_configs(opt):
    dataset_configs = []

    if opt.parent_folder is not None and len(opt.subfolders) > 0:
        for subfolder in opt.subfolders:
            fake_num_per_dir = opt.fake_num_per_dir

            # parent_folder + subfolders 模式下，每个 subfolder 会单独构成一个 fake_dirs
            # 因此如果只传了一个 fake_num_per_dir，例如 1000，则对每个 subfolder 都使用 1000
            if fake_num_per_dir is not None:
                if len(fake_num_per_dir) == 1:
                    fake_num_per_dir = [fake_num_per_dir[0]]
                else:
                    raise ValueError(
                        "在 --parent_folder + --subfolders 模式下，"
                        "--fake_num_per_dir 建议只传一个数，例如 --fake_num_per_dir 1000"
                    )

            dataset_configs.append({
                'key': subfolder,
                'fake_dirs': [os.path.join(opt.parent_folder, subfolder)],
                'fake_num_per_dir': fake_num_per_dir
            })

    elif opt.fake_dirs is not None and len(opt.fake_dirs) > 0:
        dataset_configs.append({
            'key': opt.key if opt.key is not None else 'custom_test',
            'fake_dirs': opt.fake_dirs,
            'fake_num_per_dir': opt.fake_num_per_dir
        })
    else:
        raise ValueError("必须提供 --fake_dirs，或提供 --parent_folder 与 --subfolders")

    return dataset_configs


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--real_dirs', type=str, required=True)
    parser.add_argument('--real_num_per_dir', type=str, default=None)

    parser.add_argument('--fake_dirs', type=str, default=None)
    parser.add_argument('--fake_num_per_dir', type=str, default=None)

    parser.add_argument('--parent_folder', type=str, default=None)
    parser.add_argument('--subfolders', type=str, nargs='+', default=[])

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--result_folder', type=str, default='result')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--jpeg_quality', type=int, default=None)
    parser.add_argument('--gaussian_sigma', type=float, default=None)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--profile_warmup_iters', type=int, default=10)
    parser.add_argument('--profile_bench_iters', type=int, default=30)

    opt = parser.parse_args()

    set_seed()

    opt.real_dirs = parse_dir_list(opt.real_dirs)
    opt.fake_dirs = parse_dir_list(opt.fake_dirs)
    opt.real_num_per_dir = parse_int_list(opt.real_num_per_dir)
    opt.fake_num_per_dir = parse_int_list(opt.fake_num_per_dir)

    if opt.real_dirs is None or len(opt.real_dirs) == 0:
        raise ValueError("real_dirs 不能为空")

    if opt.real_num_per_dir is not None and len(opt.real_dirs) != len(opt.real_num_per_dir):
        raise ValueError("real_dirs 和 real_num_per_dir 长度不一致")

    if opt.fake_dirs is not None and opt.fake_num_per_dir is not None:
        if len(opt.fake_dirs) != len(opt.fake_num_per_dir):
            raise ValueError("fake_dirs 和 fake_num_per_dir 长度不一致")

    os.makedirs(opt.result_folder, exist_ok=True)

    model = get_model(opt.arch)
    model = load_checkpoint_to_model(model, opt.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    perf_metrics = benchmark_model(
        model,
        device=device,
        image_size=opt.image_size,
        warmup_iters=opt.profile_warmup_iters,
        bench_iters=opt.profile_bench_iters,
    )
    gflops_str = 'N/A' if perf_metrics['gflops'] is None else f"{perf_metrics['gflops']:.3f}"
    peak_gpu_memory_str = (
        'N/A'
        if perf_metrics['peak_gpu_memory_mb'] is None
        else f"{perf_metrics['peak_gpu_memory_mb']:.3f}"
    )

    print(
        "[Profile] "
        f"Trainable Params: {perf_metrics['trainable_params_m']:.3f} M, "
        f"GFLOPs: {gflops_str}, "
        f"Inference Time/img: {perf_metrics['inference_time_ms']:.3f} ms, "
        f"Peak GPU Memory: {peak_gpu_memory_str} MB, "
        f"Model Size: {perf_metrics['model_size_mb']:.3f} MB"
    )

    dataset_configs = build_dataset_configs(opt)

    csv_path = os.path.join(opt.result_folder, 'results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Dataset', 'AUROC', 'AP',
            'R_Acc_0.5', 'F_Acc_0.5', 'Acc_0.5',
            'Best_Threshold', 'R_Acc_Best', 'F_Acc_Best', 'Acc_Best',
            'Trainable Params (M)', 'GFLOPs', 'Inference Time/img (ms)',
            'Peak GPU Memory (MB)', 'Model Size (MB)'
        ])

    all_results = []

    for cfg in dataset_configs:
        set_seed()

        dataset = EvalRealFakeDataset(
            real_dirs=opt.real_dirs,
            fake_dirs=cfg['fake_dirs'],
            arch=opt.arch,
            real_num_per_dir=opt.real_num_per_dir,
            fake_num_per_dir=cfg['fake_num_per_dir'],
            jpeg_quality=opt.jpeg_quality,
            gaussian_sigma=opt.gaussian_sigma,
            image_size=opt.image_size,
        )

        loader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

        auroc, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(
            model, loader, device=device, find_thres=True
        )

        result = [
            cfg['key'],
            round(auroc * 100, 6),
            round(ap * 100, 6),
            round(r_acc0 * 100, 6),
            round(f_acc0 * 100, 6),
            round(acc0 * 100, 6),
            round(best_thres, 6),
            round(r_acc1 * 100, 6),
            round(f_acc1 * 100, 6),
            round(acc1 * 100, 6),
            round(perf_metrics['trainable_params_m'], 6),
            'N/A' if perf_metrics['gflops'] is None else round(perf_metrics['gflops'], 6),
            round(perf_metrics['inference_time_ms'], 6),
            'N/A' if perf_metrics['peak_gpu_memory_mb'] is None else round(perf_metrics['peak_gpu_memory_mb'], 6),
            round(perf_metrics['model_size_mb'], 6),
        ]
        all_results.append(result)

        print(
            f"完成: {cfg['key']}, "
            f"AUROC: {auroc * 100:.2f}%, "
            f"AP: {ap * 100:.2f}%, "
            f"Acc@0.5: {acc0 * 100:.2f}%, "
            f"Acc@Best: {acc1 * 100:.2f}%"
        )

    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(all_results)

    print(f"\n所有测试完成，结果已保存到: {csv_path}")
    print(f"共测试了 {len(all_results)} 个数据集")


if __name__ == '__main__':
    main()
