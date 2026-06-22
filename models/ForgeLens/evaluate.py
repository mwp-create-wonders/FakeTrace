import os
import time
import torch
import numpy as np
import random

from tqdm import tqdm

from models.network.net_stage1 import net_stage1
from models.network.net_stage2 import net_stage2
from options.options import Options
from util import Logger, read_yaml
from util import  get_dataset_test
from util import get_bal_sampler
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score, average_precision_score
from torch.cuda.amp import autocast

# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake', 'seeingdark', 'san', 'crn', 'imle', 'guided', 'ldm_200', 'ldm_200_cfg', 'ldm_100', 'glide_100_27', 'glide_50_27', 'glide_100_10', 'dalle']
# multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

vals = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']
multiclass = [0, 0, 0, 0, 0, 0, 0, 0]

def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def forward_model(model, data, eval_stage):
    if eval_stage == 1:
        pre, _ = model(data)
    else:
        pre = model(data)
    return pre


def get_trainable_params_m(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def estimate_gflops(model, data, eval_stage, device):
    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception:
        return None

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    try:
        with torch.no_grad():
            for _ in range(3):
                with autocast(enabled=device.type == 'cuda'):
                    _ = forward_model(model, data, eval_stage)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

            with profile(activities=activities, with_flops=True) as prof:
                with autocast(enabled=device.type == 'cuda'):
                    _ = forward_model(model, data, eval_stage)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        total_flops = 0
        for event in prof.key_averages():
            event_flops = getattr(event, 'flops', 0)
            if event_flops is not None:
                total_flops += event_flops

        if total_flops <= 0:
            return None
        return total_flops / data.size(0) / 1e9
    except Exception:
        return None


def benchmark_inference(model, data, eval_stage, device, warmup=5, iters=20):
    with torch.no_grad():
        for _ in range(warmup):
            with autocast(enabled=device.type == 'cuda'):
                _ = forward_model(model, data, eval_stage)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        if device.type == 'cuda':
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(iters):
                with autocast(enabled=True):
                    _ = forward_model(model, data, eval_stage)
            ender.record()
            torch.cuda.synchronize(device)
            elapsed_ms = starter.elapsed_time(ender)
        else:
            start_time = time.perf_counter()
            for _ in range(iters):
                _ = forward_model(model, data, eval_stage)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

    return elapsed_ms / (iters * data.size(0))


def get_peak_gpu_memory_mb(model, data, eval_stage, device):
    if device.type != 'cuda':
        return None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        with autocast(enabled=True):
            _ = forward_model(model, data, eval_stage)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def collect_efficiency_metrics(model, data, eval_stage, device):
    return {
        'Trainable Params (M)': get_trainable_params_m(model),
        'GFLOPs': estimate_gflops(model, data, eval_stage, device),
        'Inference Time/img (ms)': benchmark_inference(model, data, eval_stage, device),
        'Peak GPU Memory (MB)': get_peak_gpu_memory_mb(model, data, eval_stage, device),
        'Model Size (MB)': get_model_size_mb(model),
    }


def print_efficiency_metrics(metrics):
    print("Efficiency Metrics" + "-" * 43)
    for key, value in metrics.items():
        if value is None:
            print(f"{key}: N/A")
        else:
            print(f"{key}: {value:.2f}")
    print("-" * 60)


def limit_dataset_size(dataset, max_samples, seed):
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset, False

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    limited_dataset = torch.utils.data.Subset(dataset, indices)
    return limited_dataset, True


def compute_subset_metrics(all_targets, all_pre_probs):
    if len(all_targets) == 0:
        return None, None

    all_targets = np.asarray(all_targets)
    all_pre_probs = np.asarray(all_pre_probs).reshape(-1)

    acc = accuracy_score(all_targets, all_pre_probs > 0.5)
    if np.unique(all_targets).size < 2:
        ap = np.nan
    else:
        ap = average_precision_score(all_targets, all_pre_probs)

    return acc, ap



if __name__ == '__main__':
    seed_torch(3407)

    # Options
    options = Options()
    opt = options.parse()
    log_dir = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    Logger(os.path.join(log_dir, 'evaluation_log.log'))

    classes = ''

    # loading
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if opt.eval_stage == 1:
        model = net_stage1()
    else:
        model = net_stage2(opt, train=False)
    model_load = torch.load(opt.weights, map_location=device)
    model.load_state_dict(model_load['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"LOAD {opt.weights}!!!!!!")

    accs = []
    aps = []
    efficiency_metrics_printed = False

    # test
    for val_id, val in enumerate(vals):
        sub_test_data_root = '{}/{}'.format(opt.eval_data_root, val)
        if multiclass[val_id] == 1:
            classes = os.listdir(sub_test_data_root)
        else:
            classes = ['']

        val_dataset = get_dataset_test(sub_test_data_root, classes)
        val_dataset, is_limited = limit_dataset_size(val_dataset, opt.max_eval_samples, opt.seed + val_id)
        if is_limited:
            print(f"({val_id + 1} {val:10}) using a sampled subset of {len(val_dataset)} images for fast evaluation")
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=opt.batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=opt.num_workers)
        else:
            sampler = get_bal_sampler(val_dataset)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=opt.batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     sampler=sampler,
                                                     num_workers=opt.num_workers)

        all_targets = []
        all_pre_probs = []

        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            if not efficiency_metrics_printed:
                metrics = collect_efficiency_metrics(model, data, opt.eval_stage, device)
                print_efficiency_metrics(metrics)
                efficiency_metrics_printed = True

            with autocast(enabled=device.type == 'cuda'):

                with torch.no_grad():
                    pre = forward_model(model, data, opt.eval_stage)

                    pre_prob = torch.sigmoid(pre).detach().cpu().view(-1).numpy()
                    target = target.detach().cpu().view(-1).numpy()

                    all_targets.extend(target.tolist())
                    all_pre_probs.extend(pre_prob.tolist())

        val_mean_acc, val_mean_ap = compute_subset_metrics(all_targets, all_pre_probs)
        if val_mean_acc is None:
            print(f"({val_id + 1} {val:10}) no valid samples were loaded, skipping metric computation")
            continue
        if np.isnan(val_mean_ap):
            print(f"({val_id + 1} {val:10}) warning: sampled targets contain only one class, AP is undefined")
        print(
            "({} {:10}) acc: {:.2f}; ap: {:.2f}".format(val_id + 1, val, val_mean_acc * 100, val_mean_ap * 100))
        accs.append(val_mean_acc)
        aps.append(val_mean_ap)

    mean_acc = np.mean(accs) * 100 if len(accs) > 0 else np.nan
    mean_ap = np.nanmean(aps) * 100 if len(aps) > 0 else np.nan
    print("({} {:10}) acc: {:.2f}; ap: {:.2f}".format('*', 'Mean', mean_acc, mean_ap))
    print('*'*60)
