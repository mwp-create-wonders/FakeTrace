import os
import json
import csv
import random
import warnings
import argparse

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from networks.trainer import Trainer

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in VALID_EXTS


def pil_loader(path):
    with Image.open(path) as img:
        return img.convert("RGB")


class BinaryTestDataset(Dataset):
    """
    测试集只包含:
        - real_dir
        - fake_dir

    标签定义:
        real -> 1
        fake -> 0

    若设置 test_num:
        - 从 real 中随机抽 test_num 张
        - 从 fake 中随机抽 test_num 张
    """

    def __init__(self, real_dir, fake_dir, transform=None, test_num=None, seed=2026):
        self.samples = []
        self.transform = transform
        self.test_num = test_num
        self.seed = seed

        real_files = [
            os.path.join(real_dir, name)
            for name in os.listdir(real_dir)
            if is_image_file(name)
        ]
        fake_files = [
            os.path.join(fake_dir, name)
            for name in os.listdir(fake_dir)
            if is_image_file(name)
        ]

        real_files = sorted(real_files)
        fake_files = sorted(fake_files)

        rng = random.Random(seed)

        if test_num is not None:
            if len(real_files) > test_num:
                real_files = rng.sample(real_files, test_num)
            if len(fake_files) > test_num:
                fake_files = rng.sample(fake_files, test_num)

        for path in real_files:
            self.samples.append((path, 1, "real"))

        for path in fake_files:
            self.samples.append((path, 0, "fake"))

        print(
            f"[BinaryTestDataset] Loaded {len(real_files)} real samples and "
            f"{len(fake_files)} fake samples. Total: {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, kind = self.samples[idx]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "kind": kind,
            "path": path,
        }


def binary_test_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    kinds = [b["kind"] for b in batch]
    paths = [b["path"] for b in batch]

    return {
        "image": images,
        "label": labels,
        "kind": kinds,
        "path": paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Binary test for real/fake classification")

    parser.add_argument("--name", type=str, default="test_run")
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2026)

    # 测试数据
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--test_num", type=int, default=None, help="Number of real and fake images to sample for testing")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    # 预处理
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--norm_mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--norm_std", type=float, nargs=3, default=[0.229, 0.224, 0.225])

    # 模型配置（与训练保持一致）
    parser.add_argument("--backbone_name", type=str, default="dinov2_vitl14")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--return_tokens", action="store_true")

    # Trainer 初始化仍需要这些参数
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--t_max", type=int, default=1000)
    parser.add_argument("--accumulation_steps", type=int, default=1)

    parser.add_argument("--lambda_bin", type=float, default=1.0)
    parser.add_argument("--lambda_src", type=float, default=0.5)
    parser.add_argument("--lambda_pair", type=float, default=0.5)
    parser.add_argument("--lambda_rf", type=float, default=0.5)
    parser.add_argument("--lambda_dmar", type=float, default=0.2)
    parser.add_argument("--lambda_con", type=float, default=0.2)

    parser.add_argument("--margin_real_fake", type=float, default=0.4)
    parser.add_argument("--margin_dm_ar", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--resume_strict", type=int, default=1)

    args = parser.parse_args()

    str_ids = args.gpu_ids.split(",")
    args.gpu_ids = [int(x) for x in str_ids if int(x) >= 0]

    if len(args.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_ids[0])
        args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
    else:
        args.device = torch.device("cpu")

    return args


def build_test_dataloader(opt):
    transform = transforms.Compose([
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=opt.norm_mean, std=opt.norm_std),
    ])

    dataset = BinaryTestDataset(
        real_dir=opt.real_dir,
        fake_dir=opt.fake_dir,
        transform=transform,
        test_num=opt.test_num,
        seed=opt.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_threads,
        pin_memory=opt.pin_memory,
        drop_last=False,
        collate_fn=binary_test_collate_fn,
    )
    return loader


def compute_average_precision(y_true, y_score):
    """
    手写 AP 计算，正类定义为 label=1（real）
    y_true: list/np.ndarray of {0,1}
    y_score: list/np.ndarray, 越大越偏向正类
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    pos_count = int((y_true == 1).sum())
    if pos_count == 0:
        return None

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tp = (y_true_sorted == 1).astype(np.float64)
    fp = (y_true_sorted == 0).astype(np.float64)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / max(pos_count, 1)

    # AP = sum over points where label == positive of precision@k / num_positive
    ap = float((precision * tp).sum() / pos_count)
    return ap


def evaluate(model, data_loader, device):
    model.set_model_eval()
    bce_loss = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_batches = 0

    total_correct = 0
    total_num = 0

    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0

    all_labels = []
    all_probs = []

    # confusion matrix: rows=true, cols=pred
    # label: real=1, fake=0
    confusion = torch.zeros(2, 2, dtype=torch.long)

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            outputs = model.model(images, return_feature=False, return_tokens=False)
            logits = outputs["binary_logits"].squeeze(1)

            loss = bce_loss(logits, labels)
            total_loss += loss.item()
            total_batches += 1

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            labels_long = labels.long()

            all_labels.extend(labels_long.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            total_correct += (preds == labels_long).sum().item()
            total_num += labels_long.numel()

            real_mask = labels_long == 1
            fake_mask = labels_long == 0

            if real_mask.any():
                real_correct += (preds[real_mask] == labels_long[real_mask]).sum().item()
                real_total += real_mask.sum().item()

            if fake_mask.any():
                fake_correct += (preds[fake_mask] == labels_long[fake_mask]).sum().item()
                fake_total += fake_mask.sum().item()

            for p, y in zip(preds.cpu().tolist(), labels_long.cpu().tolist()):
                confusion[y][p] += 1

    ap = compute_average_precision(all_labels, all_probs)

    results = {
        "loss_bin": total_loss / max(total_batches, 1),
        "binary_acc": total_correct / max(total_num, 1),
        "real_acc": real_correct / max(real_total, 1),
        "fake_acc": fake_correct / max(fake_total, 1),
        "ap": ap,
        "num_samples": total_num,
        "num_real": real_total,
        "num_fake": fake_total,
        "confusion_matrix": confusion.tolist(),  # row=true, col=pred
    }
    return results


def pretty_print_results(results):
    print("\n==================== TEST RESULTS ====================")
    print(f"{'loss_bin':15s}: {results['loss_bin']:.6f}")
    print(f"{'binary_acc':15s}: {results['binary_acc']:.6f}")
    print(f"{'real_acc':15s}: {results['real_acc']:.6f}")
    print(f"{'fake_acc':15s}: {results['fake_acc']:.6f}")
    ap_str = "None" if results["ap"] is None else f"{results['ap']:.6f}"
    print(f"{'ap':15s}: {ap_str}")
    print("------------------------------------------------------")
    print("confusion_matrix (rows=true, cols=pred)")
    print("              pred_fake   pred_real")
    print(f"true_fake     {results['confusion_matrix'][0][0]:9d} {results['confusion_matrix'][0][1]:10d}")
    print(f"true_real     {results['confusion_matrix'][1][0]:9d} {results['confusion_matrix'][1][1]:10d}")
    print("======================================================\n")


def append_results_to_csv(opt, results, checkpoint):
    csv_path = opt.save_csv
    if not csv_path:
        return

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    total_steps = checkpoint.get("total_steps", "")
    epoch = checkpoint.get("epoch", "")

    row = {
        "name": opt.name,
        "checkpoint_path": opt.checkpoint_path,
        "epoch": epoch,
        "total_steps": total_steps,
        "real_dir": opt.real_dir,
        "fake_dir": opt.fake_dir,
        "test_num": opt.test_num if opt.test_num is not None else "",
        "batch_size": opt.batch_size,
        "image_size": opt.image_size,
        "seed": opt.seed,
        "num_samples": results["num_samples"],
        "num_real": results["num_real"],
        "num_fake": results["num_fake"],
        "loss_bin": results["loss_bin"],
        "binary_acc": results["binary_acc"],
        "real_acc": results["real_acc"],
        "fake_acc": results["fake_acc"],
        "ap": results["ap"] if results["ap"] is not None else "",
        "tn_true_fake_pred_fake": results["confusion_matrix"][0][0],
        "fp_true_fake_pred_real": results["confusion_matrix"][0][1],
        "fn_true_real_pred_fake": results["confusion_matrix"][1][0],
        "tp_true_real_pred_real": results["confusion_matrix"][1][1],
    }

    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended test results to CSV: {csv_path}")


def main():
    opt = parse_args()
    set_seed(opt.seed)

    data_loader = build_test_dataloader(opt)
    print(f"Length of test data loader: {len(data_loader)}")

    model = Trainer(opt)

    print(f"Loading checkpoint from: {opt.checkpoint_path}")
    checkpoint = torch.load(opt.checkpoint_path, map_location=opt.device)
    model.model.load_state_dict(checkpoint["model"], strict=bool(opt.resume_strict))
    if "total_steps" in checkpoint:
        model.total_steps = checkpoint["total_steps"]

    results = evaluate(model, data_loader, opt.device)
    pretty_print_results(results)

    if opt.save_json:
        if os.path.dirname(opt.save_json):
            os.makedirs(os.path.dirname(opt.save_json), exist_ok=True)
        with open(opt.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved test results to: {opt.save_json}")

    if opt.save_csv:
        append_results_to_csv(opt, results, checkpoint)


if __name__ == "__main__":
    main()