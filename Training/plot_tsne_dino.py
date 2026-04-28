import os
import random
import warnings
import argparse

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


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


class TSNEDataset(Dataset):
    """
    三类标签:
        0 -> Real
        1 -> DM
        2 -> AR
    """

    def __init__(
        self,
        real_dir,
        dm_dir,
        ar_dir,
        transform=None,
        num_per_class=300,
        seed=2026
    ):
        self.samples = []
        self.transform = transform
        rng = random.Random(seed)

        label_map = {
            0: ("Real", real_dir),
            1: ("DM", dm_dir),
            2: ("AR", ar_dir),
        }

        for label, (name, folder) in label_map.items():
            if not folder or not os.path.isdir(folder):
                raise ValueError(f"{name} dir is invalid: {folder}")

            files = [
                os.path.join(folder, x)
                for x in os.listdir(folder)
                if is_image_file(x)
            ]
            files = sorted(files)

            if len(files) == 0:
                raise ValueError(f"{name} dir has no images: {folder}")

            if num_per_class is not None and len(files) > num_per_class:
                files = rng.sample(files, num_per_class)

            for path in files:
                self.samples.append((path, label, name))

            print(f"[{name}] {len(files)} samples")

        print(f"[TSNEDataset] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, name = self.samples[idx]
        img = pil_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "name": name,
            "path": path,
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    names = [b["name"] for b in batch]
    paths = [b["path"] for b in batch]

    return {
        "image": images,
        "label": labels,
        "name": names,
        "path": paths,
    }


def parse_args():
    parser = argparse.ArgumentParser("Plot t-SNE with pretrained ResNet50 features")

    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--dm_dir", type=str, required=True)
    parser.add_argument("--ar_dir", type=str, required=True)

    parser.add_argument("--num_per_class", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--save_path", type=str, default="./tsne_pretrained_3class.png")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--marker_size", type=float, default=10)

    args = parser.parse_args()

    str_ids = args.gpu_ids.split(",")
    args.gpu_ids = [int(x) for x in str_ids if int(x) >= 0]

    if len(args.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_ids[0])
        args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
    else:
        args.device = torch.device("cpu")

    return args


def build_transform(image_size):
    # 用 torchvision 官方预训练权重自带的标准预处理更稳
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    # 这里把 resize/crop 固定到你指定尺寸
    # 若你想完全使用官方默认流程，可直接 return preprocess
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocess.mean,
            std=preprocess.std
        ),
    ])
    return transform


def build_dataloader(opt):
    transform = build_transform(opt.image_size)

    dataset = TSNEDataset(
        real_dir=opt.real_dir,
        dm_dir=opt.dm_dir,
        ar_dir=opt.ar_dir,
        transform=transform,
        num_per_class=opt.num_per_class,
        seed=opt.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_threads,
        pin_memory=opt.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return loader


def build_pretrained_model(device):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()   # 去掉分类层，直接输出全局特征
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, data_loader, device):
    all_features = []
    all_labels = []

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()

        feats = model(images)  # [B, 2048]

        if feats.dim() > 2:
            feats = feats.flatten(1)

        feats = feats.detach().cpu().numpy()

        all_features.append(feats)
        all_labels.append(labels)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def plot_tsne(features, labels, save_path, perplexity=30, pca_dim=50, marker_size=10, seed=2026):
    label_names = {
        0: "Real",
        1: "DM",
        2: "AR"
    }

    features = StandardScaler().fit_transform(features)

    if features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)

    perplexity = min(perplexity, features.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed
    )
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))

    for lab in sorted(np.unique(labels)):
        idx = labels == lab
        plt.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=marker_size,
            alpha=0.7,
            label=label_names[int(lab)]
        )

    plt.xticks([])
    plt.yticks([])
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE figure to: {save_path}")


def main():
    opt = parse_args()
    set_seed(opt.seed)

    print("Building dataloader...")
    data_loader = build_dataloader(opt)

    print("Building pretrained model...")
    model = build_pretrained_model(opt.device)

    print("Extracting pretrained features...")
    features, labels = extract_features(model, data_loader, opt.device)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    print("Plotting t-SNE...")
    plot_tsne(
        features=features,
        labels=labels,
        save_path=opt.save_path,
        perplexity=opt.perplexity,
        pca_dim=opt.pca_dim,
        marker_size=opt.marker_size,
        seed=opt.seed,
    )


if __name__ == "__main__":
    main()