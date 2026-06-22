from __future__ import print_function

import argparse
import csv
import os
import random
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from models.moe_adapter_clip import MoE_Adapter_CLIP


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


preprocess = Compose([
    CenterCrop((224, 224)),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

resize_preprocess = Compose([
    Resize(256),
    CenterCrop((224, 224)),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def read_and_preprocess_image(image_path):
    temp_img = Image.open(image_path)
    try:
        image = preprocess(temp_img)
    except (ValueError, RuntimeError):
        print("size can not crop to 224x224 in {}".format(image_path))
        image = resize_preprocess(temp_img)
    return image


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError("Path does not exist: {}".format(root_dir))

    image_paths = [path for path in root_dir.rglob("*") if is_image_file(path)]
    image_paths.sort()
    return image_paths


def maybe_sample_paths(paths, max_count, seed):
    if max_count is None or max_count <= 0 or len(paths) <= max_count:
        return paths

    rng = random.Random(seed)
    sampled_paths = rng.sample(paths, max_count)
    sampled_paths.sort()
    return sampled_paths


def build_samples(real_dir, fake_root, max_real_count=None, max_fake_count=None, seed=42):
    real_dir = Path(real_dir)
    fake_root = Path(fake_root)

    real_paths = collect_image_paths(real_dir)
    if not real_paths:
        raise ValueError("No real images found in {}".format(real_dir))
    real_paths = maybe_sample_paths(real_paths, max_real_count, seed)

    fake_subdirs = [path for path in fake_root.iterdir() if path.is_dir()]
    fake_subdirs.sort()
    if not fake_subdirs:
        raise ValueError("No fake generator sub-directories found in {}".format(fake_root))

    samples = []
    for real_path in real_paths:
        samples.append({
            "path": str(real_path),
            "label": 0,
            "generator": "real",
        })

    generator_to_fake_count = {}
    for generator_idx, generator_dir in enumerate(fake_subdirs):
        fake_paths = collect_image_paths(generator_dir)
        if not fake_paths:
            print("Skip empty generator directory: {}".format(generator_dir))
            continue
        fake_paths = maybe_sample_paths(fake_paths, max_fake_count, seed + generator_idx + 1)

        generator_name = generator_dir.name
        generator_to_fake_count[generator_name] = len(fake_paths)
        for fake_path in fake_paths:
            samples.append({
                "path": str(fake_path),
                "label": 1,
                "generator": generator_name,
            })

    if not generator_to_fake_count:
        raise ValueError("No fake images found under {}".format(fake_root))

    return samples, real_paths, generator_to_fake_count


class ForgeryBenchmarkDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = read_and_preprocess_image(sample["path"])
        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "path": sample["path"],
            "generator": sample["generator"],
        }


def benchmark_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    paths = [item["path"] for item in batch]
    generators = [item["generator"] for item in batch]
    return {
        "image": images,
        "label": labels,
        "path": paths,
        "generator": generators,
    }


def load_model(model_path, device):
    model = MoE_Adapter_CLIP(device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def get_parameter_count_m(model):
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def get_model_file_size_mb(model_path, model):
    if model_path and os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return (parameter_bytes + buffer_bytes) / (1024 * 1024)


def estimate_gflops(model, device, input_shape=(1, 3, 224, 224)):
    dummy_input = torch.randn(*input_shape, device=device)

    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return float(macs) * 2.0 / 1e9
    except Exception:
        pass

    try:
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                model(dummy_input)

        total_flops = 0
        for event in prof.key_averages():
            event_flops = getattr(event, "flops", 0)
            if event_flops:
                total_flops += event_flops

        if total_flops > 0:
            return float(total_flops) / 1e9
    except Exception:
        pass

    return None


def format_metric(value, precision=3, missing="N/A"):
    if value is None:
        return missing
    return round(float(value), precision)


def measure_latency_and_memory(model, device, batch_size, warmup=10, repeats=30):
    if device.type != "cuda":
        return None, None

    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    latency_samples = []
    peak_memory_samples = []

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize(device)

        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize(device)
            end = time.perf_counter()

            latency_samples.append((end - start) * 1000.0 / batch_size)
            peak_memory_samples.append(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    return statistics.mean(latency_samples), max(peak_memory_samples)


def evaluate(model, dataloader, device, threshold):
    sigmoid = torch.nn.Sigmoid()
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(int).tolist()
            generators = batch["generator"]
            paths = batch["path"]

            scores = sigmoid(model(images)).squeeze(1).detach().cpu().numpy().tolist()
            preds = [1 if score > threshold else 0 for score in scores]

            for path, generator, label, score, pred in zip(paths, generators, labels, scores, preds):
                rows.append({
                    "path": path,
                    "generator": generator,
                    "label": label,
                    "score": float(score),
                    "pred": pred,
                })

    return rows


def summarize_rows(rows):
    labels = [row["label"] for row in rows]
    preds = [row["pred"] for row in rows]
    scores = [row["score"] for row in rows]

    real_count = sum(1 for row in rows if row["label"] == 0)
    fake_count = sum(1 for row in rows if row["label"] == 1)

    return {
        "num_samples": len(rows),
        "num_real": real_count,
        "num_fake": fake_count,
        "accuracy": accuracy_score(labels, preds) if rows else 0.0,
        "avg_score": sum(scores) / len(scores) if rows else 0.0,
    }


def summarize_by_generator(rows, real_rows):
    generator_names = sorted({row["generator"] for row in rows if row["generator"] != "real"})
    summaries = []

    for generator_name in generator_names:
        fake_rows = [row for row in rows if row["generator"] == generator_name]
        generator_rows = real_rows + fake_rows
        summary = summarize_rows(generator_rows)
        summary["generator"] = generator_name
        summaries.append(summary)

    return summaries


def save_predictions(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "generator", "label", "score", "pred"])
        writer.writeheader()
        writer.writerows(rows)


def print_dataset_overview(real_paths, generator_to_fake_count):
    print("=== Dataset Overview ===")
    print("Real images: {}".format(len(real_paths)))
    for generator_name, fake_count in generator_to_fake_count.items():
        print("Fake images [{}]: {}".format(generator_name, fake_count))


def print_model_stats(stats):
    print("=== Model Stats ===")
    print("Params (M): {}".format(format_metric(stats["params_m"])))
    print("GFLOPs (G): {}".format(format_metric(stats["gflops_g"])))
    print("Latency (ms/img): {}".format(format_metric(stats["latency_ms_per_img"])))
    print("Memory (MB): {}".format(format_metric(stats["memory_mb"])))
    print("Size (MB): {}".format(format_metric(stats["size_mb"])))


def print_result_table(overall_summary, generator_summaries):
    print("=== Evaluation Summary ===")
    print(
        "Overall | samples: {num_samples} | real: {num_real} | fake: {num_fake} | acc: {accuracy:.4f} | avg_score: {avg_score:.4f}".format(
            **overall_summary
        )
    )
    for summary in generator_summaries:
        print(
            "{generator} | samples: {num_samples} | real: {num_real} | fake: {num_fake} | acc: {accuracy:.4f} | avg_score: {avg_score:.4f}".format(
                **summary
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Forensic-MoE on folder-structured datasets.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--real-dir", type=str, required=True, help="Directory containing all real images.")
    parser.add_argument(
        "--fake-root",
        type=str,
        required=True,
        help="Parent directory whose each sub-directory corresponds to one generator.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--max-real-samples",
        type=int,
        default=None,
        help="Maximum number of real images to use in this evaluation run. Use all when omitted.",
    )
    parser.add_argument(
        "--max-fake-samples",
        type=int,
        default=None,
        help="Maximum number of fake images to use per generator sub-directory. Use all when omitted.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sampling subsets.")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to save per-image predictions.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available. This project currently expects a CUDA runtime.")

    samples, real_paths, generator_to_fake_count = build_samples(
        args.real_dir,
        args.fake_root,
        max_real_count=args.max_real_samples,
        max_fake_count=args.max_fake_samples,
        seed=args.seed,
    )
    dataset = ForgeryBenchmarkDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=benchmark_collate_fn,
    )

    print_dataset_overview(real_paths, generator_to_fake_count)

    model = load_model(args.model_path, device)

    model_stats = {
        "params_m": get_parameter_count_m(model),
        "gflops_g": estimate_gflops(model, device),
        "size_mb": get_model_file_size_mb(args.model_path, model),
    }
    latency_ms_per_img, memory_mb = measure_latency_and_memory(
        model,
        device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    model_stats["latency_ms_per_img"] = latency_ms_per_img
    model_stats["memory_mb"] = memory_mb
    print_model_stats(model_stats)

    rows = evaluate(model, dataloader, device, args.threshold)
    overall_summary = summarize_rows(rows)
    real_rows = [row for row in rows if row["generator"] == "real"]
    generator_summaries = summarize_by_generator(rows, real_rows)
    print_result_table(overall_summary, generator_summaries)

    if args.save_csv:
        save_predictions(rows, args.save_csv)
        print("Predictions saved to {}".format(args.save_csv))
