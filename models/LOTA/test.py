import os
import csv
import time
import torch
import torch.nn as nn
from PIL import ImageFile

from util import set_random_seed as seed_generator
from config import ConfigurationManager as Configurator
from model import model as DeepLearningModel
from loader import get_single_loader as acquire_single_loader
from sklearn.metrics import average_precision_score

ImageFile.LOAD_TRUNCATED_IMAGES = True


def count_trainable_params_million(neural_network):
    trainable_params = sum(
        parameter.numel() for parameter in neural_network.parameters() if parameter.requires_grad
    )
    return trainable_params / 1e6


def estimate_model_size_mb(neural_network, weight_path=None):
    if weight_path is not None and os.path.isfile(weight_path):
        return os.path.getsize(weight_path) / (1024 ** 2)

    total_bytes = 0
    for tensor in neural_network.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def estimate_model_gflops(neural_network, input_shape, device):
    flops_holder = {"total": 0}
    hooks = []

    def conv_hook(module, inputs, output):
        input_tensor = inputs[0]
        batch_size = input_tensor.shape[0]
        output_height, output_width = output.shape[2], output.shape[3]
        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        filters_per_channel = out_channels // groups
        conv_macs = (
            batch_size
            * output_height
            * output_width
            * kernel_height
            * kernel_width
            * in_channels
            * filters_per_channel
        )
        if module.bias is not None:
            conv_macs += batch_size * output_height * output_width * out_channels

        flops_holder["total"] += 2 * conv_macs

    def linear_hook(module, inputs, output):
        input_tensor = inputs[0]
        batch_size = input_tensor.shape[0] if input_tensor.dim() > 1 else 1
        linear_macs = batch_size * module.in_features * module.out_features
        if module.bias is not None:
            linear_macs += batch_size * module.out_features

        flops_holder["total"] += 2 * linear_macs

    def batchnorm_hook(module, inputs, output):
        flops_holder["total"] += 2 * output.numel()

    def relu_hook(module, inputs, output):
        flops_holder["total"] += output.numel()

    def pool_hook(module, inputs, output):
        flops_holder["total"] += output.numel()

    for module in neural_network.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(batchnorm_hook))
        elif isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(relu_hook))
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            hooks.append(module.register_forward_hook(pool_hook))

    was_training = neural_network.training
    neural_network.eval()

    with torch.no_grad():
        dummy_input = torch.randn(*input_shape, device=device)
        neural_network(dummy_input)

    for hook in hooks:
        hook.remove()

    if was_training:
        neural_network.train()

    return flops_holder["total"] / 1e9


def measure_inference_performance(neural_network, input_shape, device, warmup_iters=20, timed_iters=50):
    was_training = neural_network.training
    neural_network.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            neural_network(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.perf_counter()
        for _ in range(timed_iters):
            neural_network(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed_time = time.perf_counter() - start_time

    if was_training:
        neural_network.train()

    inference_time_ms = (elapsed_time / timed_iters) * 1000.0 / input_shape[0]
    peak_gpu_memory_mb = 0.0
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return inference_time_ms, peak_gpu_memory_mb


def collect_model_metrics(neural_network, config):
    device = next(neural_network.parameters()).device
    input_shape = (1, 3, config.img_height, config.img_height)

    metrics = {
        "Trainable Params (M)": count_trainable_params_million(neural_network),
        "GFLOPs": estimate_model_gflops(neural_network, input_shape, device),
        "Model Size (MB)": estimate_model_size_mb(neural_network, getattr(config, "load", None)),
    }

    inference_time_ms, peak_gpu_memory_mb = measure_inference_performance(
        neural_network, input_shape, device
    )
    metrics["Inference Time/img (ms)"] = inference_time_ms
    metrics["Peak GPU Memory (MB)"] = peak_gpu_memory_mb
    return metrics


def print_model_metrics(metrics):
    print("\n[Model Efficiency Metrics]")
    print(f"  Trainable Params (M)    : {metrics['Trainable Params (M)']:.4f}")
    print(f"  GFLOPs                  : {metrics['GFLOPs']:.4f}")
    print(f"  Inference Time/img (ms) : {metrics['Inference Time/img (ms)']:.4f}")
    print(f"  Peak GPU Memory (MB)    : {metrics['Peak GPU Memory (MB)']:.2f}")
    print(f"  Model Size (MB)         : {metrics['Model Size (MB)']:.2f}")


def configure_computation_device(device_id):
    """Set computational hardware environment"""
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"Selected computation device: GPU {device_id}")


def evaluate_one_fake_dataset(test_loader, neural_network, fake_name="unknown"):
    """
    在固定 real 数据集 + 当前 fake 数据集 上进行一次评估
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

            # 保存用于 AP 计算的概率和标签
            all_targets.extend(target_labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    real_acc = real_correct / real_total if real_total > 0 else 0.0
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0.0

    # real=1 作为正类的 AP
    real_ap = average_precision_score(all_targets, all_probs) if len(set(all_targets)) > 1 else None

    # fake=1 作为正类的 AP
    fake_targets = [1 - int(x) for x in all_targets]
    fake_probs = [1.0 - float(x) for x in all_probs]
    fake_ap = average_precision_score(fake_targets, fake_probs) if len(set(fake_targets)) > 1 else None
    mean_ap = None
    if real_ap is not None and fake_ap is not None:
        mean_ap = (real_ap + fake_ap) / 2.0

    print(f"\n[Evaluation Result] Fake Dataset: {fake_name}")
    print(f"  Overall Accuracy : {overall_acc:.4f}")
    print(f"  Real Accuracy    : {real_acc:.4f}")
    print(f"  Fake Accuracy    : {fake_acc:.4f}")
    print(f"  Real AP          : {real_ap:.4f}" if real_ap is not None else "  Real AP          : None")
    print(f"  Fake AP          : {fake_ap:.4f}" if fake_ap is not None else "  Fake AP          : None")
    print(f"  Mean AP          : {mean_ap:.4f}" if mean_ap is not None else "  Mean AP          : None")

    return {
        "fake_dataset": fake_name,
        "overall_acc": overall_acc,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "real_ap": real_ap,
        "fake_ap": fake_ap,
        "mean_ap": mean_ap,
        "num_real": real_total,
        "num_fake": fake_total,
        "num_total": total_samples,
    }


def save_results_to_csv(results, save_csv_path):
    """Save all evaluation results into a CSV file"""
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

    file_exists = os.path.isfile(save_csv_path)

    with open(save_csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fake_dataset",
                "fake_dir",
                "test_num",
                "overall_acc",
                "real_acc",
                "fake_acc",
                "real_ap",
                "fake_ap",
                "mean_ap",
                "num_real",
                "num_fake",
                "num_total",
                "Trainable Params (M)",
                "GFLOPs",
                "Inference Time/img (ms)",
                "Peak GPU Memory (MB)",
                "Model Size (MB)",
            ]
        )

        if not file_exists:
            writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"\nSaved evaluation results to: {save_csv_path}")


def execute_evaluation_procedure():
    """Main evaluation workflow"""
    seed_generator()

    # 读取配置
    config = Configurator().parse()
    config.isTrain = False
    config.isVal = True

    # 先设置 GPU
    configure_computation_device(config.gpu_id)

    # 检查测试目录
    if not hasattr(config, "test_real_dirs") or len(config.test_real_dirs) == 0:
        raise ValueError("Please provide --test_real_dirs")
    if not hasattr(config, "test_fake_dirs") or len(config.test_fake_dirs) == 0:
        raise ValueError("Please provide --test_fake_dirs")

    # 初始化模型
    network_instance = DeepLearningModel().cuda()

    if config.load is not None:
        network_instance.load_state_dict(torch.load(config.load))
        print(f"Loaded model parameters from {config.load}")
    else:
        raise ValueError("Please provide model weights using --load")

    model_metrics = collect_model_metrics(network_instance, config)
    print_model_metrics(model_metrics)

    # 创建结果目录
    results_path = config.save_path
    os.makedirs(results_path, exist_ok=True)

    # 结果保存路径
    save_csv_path = os.path.join(results_path, "test_results.csv")

    all_results = []

    if getattr(config, "test_real_parent_dir", ""):
        print(f"Resolved test real directories from parent: {config.test_real_parent_dir}")
        for resolved_real_dir in config.test_real_dirs:
            print(f"  Real Dir -> {resolved_real_dir}")

    if getattr(config, "test_fake_parent_dir", ""):
        print(f"Resolved test fake directories from parent: {config.test_fake_parent_dir}")
        for resolved_fake_dir in config.test_fake_dirs:
            print(f"  Fake Dir -> {resolved_fake_dir}")

    # 固定 real，逐个 fake 测试
    for fake_dir in config.test_fake_dirs:
        fake_name = os.path.basename(os.path.normpath(fake_dir))
    
        print("\n" + "=" * 80)
        print(f"Testing fake dataset: {fake_dir}")
        print(f"Using at most {config.test_num} real samples and {config.test_num} fake samples")
        print("=" * 80)
    
        test_loader, dataset_size = acquire_single_loader(
            config,
            real_dirs=config.test_real_dirs,
            fake_dirs=[fake_dir]
        )
    
        result = evaluate_one_fake_dataset(
            test_loader=test_loader,
            neural_network=network_instance,
            fake_name=fake_name
        )
    
        result["fake_dir"] = fake_dir
        result["test_num"] = config.test_num
        result.update(model_metrics)
        all_results.append(result)

    # 保存所有结果
    save_results_to_csv(all_results, save_csv_path)

    print("\nAll evaluations completed.")


if __name__ == '__main__':
    execute_evaluation_procedure()
