import os
import random
import warnings
import argparse

warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    parser = argparse.ArgumentParser("Plot t-SNE for Real / DM / AR")

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--dm_dir", type=str, required=True)
    parser.add_argument("--ar_dir", type=str, required=True)

    parser.add_argument("--num_per_class", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--norm_mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--norm_std", type=float, nargs=3, default=[0.229, 0.224, 0.225])

    parser.add_argument("--save_path", type=str, default="./tsne_3class.png")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--marker_size", type=float, default=10)

    # 保持和训练时 Trainer 初始化一致
    parser.add_argument("--name", type=str, default="tsne_vis")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--backbone_name", type=str, default="dinov2_vitl14")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--proj_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--return_tokens", action="store_true")

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


def build_dataloader(opt):
    transform = transforms.Compose([
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=opt.norm_mean, std=opt.norm_std),
    ])

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


def extract_feature_from_outputs(outputs):
    """
    根据你的模型输出格式自动找特征。
    如果第一次运行报错，就把打印出的 keys 看一下，
    然后手动改成对应的字段即可。
    """
    candidate_keys = [
        "proj_features",
        "proj_feature",
        "features",
        "feature",
        "feat",
        "embedding",
        "embeddings",
        "global_feat",
        "global_feature",
        "cls_feature",
        "cls_token",
    ]

    if isinstance(outputs, dict):
        for key in candidate_keys:
            if key in outputs and torch.is_tensor(outputs[key]):
                return outputs[key]

        print("Available output keys:", list(outputs.keys()))

    raise KeyError(
        "Cannot find feature tensor in model outputs. "
        "Please modify 'extract_feature_from_outputs()' according to your model."
    )


@torch.no_grad()
def extract_features(model, data_loader, device):
    model.set_model_eval()

    all_features = []
    all_labels = []

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()

        outputs = model.model(images, return_feature=True, return_tokens=False)
        feats = extract_feature_from_outputs(outputs)

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

    print("Building model...")
    model = Trainer(opt)

    print(f"Loading checkpoint from: {opt.checkpoint_path}")
    checkpoint = torch.load(opt.checkpoint_path, map_location=opt.device)
    model.model.load_state_dict(checkpoint["model"], strict=bool(opt.resume_strict))

    print("Extracting features...")
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