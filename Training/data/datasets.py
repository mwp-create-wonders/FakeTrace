import os
import random
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in VALID_EXTS


def list_images(folder: str) -> Dict[str, str]:
    files = {}
    for name in os.listdir(folder):
        if is_image_file(name):
            files[name] = os.path.join(folder, name)
    return files


def pil_loader(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


# =========================================================
# 同步增强器
# 作用：让同一组相关图像（6张）使用完全相同的随机增强参数
# =========================================================

class SyncPairAugmenter:
    """
    对一个 sample dict 中的所有图像执行同步随机增强。
    sample 格式:
        {
            "real": PIL.Image,
            "real_processed": PIL.Image,
            "dm": PIL.Image,
            "dm_processed": PIL.Image,
            "ar": PIL.Image,
            "ar_processed": PIL.Image,
        }

    适合你的场景：
    - 同名 real / dm / ar 图像保持空间和几何对应
    - 为 pair loss / contrastive / guided fusion 做准备
    """

    def __init__(
        self,
        resize: Optional[Tuple[int, int]] = None,
        hflip_prob: float = 0.5,
        crop_size: Optional[Tuple[int, int]] = None,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        gamma_range: Optional[Tuple[float, float]] = None,
        blur_prob: float = 0.0,
        blur_radius_range: Tuple[float, float] = (0.3, 1.0),
    ):
        self.resize = resize
        self.hflip_prob = hflip_prob
        self.crop_size = crop_size
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma_range = gamma_range
        self.blur_prob = blur_prob
        self.blur_radius_range = blur_radius_range

    def _sample_params(self, img: Image.Image) -> Dict:
        w, h = img.size
        params = {}

        # resize
        params["resize"] = self.resize

        # hflip
        params["do_hflip"] = random.random() < self.hflip_prob

        # crop
        if self.crop_size is not None:
            crop_w, crop_h = self.crop_size
            if crop_w > w or crop_h > h:
                raise ValueError(
                    f"crop_size={self.crop_size} 大于当前图像尺寸 {(w, h)}"
                )
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            params["crop_box"] = (left, top, left + crop_w, top + crop_h)
        else:
            params["crop_box"] = None

        # color jitter
        def sample_factor(strength: float) -> float:
            if strength <= 0:
                return 1.0
            low = max(0.0, 1.0 - strength)
            high = 1.0 + strength
            return random.uniform(low, high)

        params["brightness_factor"] = sample_factor(self.brightness)
        params["contrast_factor"] = sample_factor(self.contrast)
        params["saturation_factor"] = sample_factor(self.saturation)

        # gamma
        if self.gamma_range is not None:
            params["gamma"] = random.uniform(*self.gamma_range)
        else:
            params["gamma"] = None

        # blur
        params["do_blur"] = random.random() < self.blur_prob
        if params["do_blur"]:
            params["blur_radius"] = random.uniform(*self.blur_radius_range)
        else:
            params["blur_radius"] = 0.0

        return params

    @staticmethod
    def _apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
        if gamma is None:
            return img
        x = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        x = x.float().view(img.size[1], img.size[0], 3) / 255.0
        x = x.pow(gamma).clamp(0, 1) * 255.0
        x = x.byte().numpy()
        return Image.fromarray(x)

    def _apply_one(self, img: Image.Image, params: Dict) -> Image.Image:
        # resize
        if params["resize"] is not None:
            img = img.resize(params["resize"], Image.BICUBIC)

        # crop
        if params["crop_box"] is not None:
            img = img.crop(params["crop_box"])

        # flip
        if params["do_hflip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # brightness / contrast / saturation
        if params["brightness_factor"] != 1.0:
            img = ImageEnhance.Brightness(img).enhance(params["brightness_factor"])

        if params["contrast_factor"] != 1.0:
            img = ImageEnhance.Contrast(img).enhance(params["contrast_factor"])

        if params["saturation_factor"] != 1.0:
            img = ImageEnhance.Color(img).enhance(params["saturation_factor"])

        # gamma
        if params["gamma"] is not None:
            img = self._apply_gamma(img, params["gamma"])

        # blur
        if params["do_blur"]:
            img = img.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))

        return img

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        if not isinstance(sample, dict):
            raise TypeError("SyncPairAugmenter 输入必须是 dict")

        first_key = next(iter(sample.keys()))
        params = self._sample_params(sample[first_key])

        out = {}
        for k, img in sample.items():
            out[k] = self._apply_one(img, params)
        return out


# =========================================================
# 主 Dataset
# =========================================================

class SixFolderForgeryDataset(Dataset):
    """
    读取 6 个文件夹，并按同名文件组织成一个样本组。

    文件夹含义:
        - real
        - real_processed
        - dm
        - dm_processed
        - ar
        - ar_processed

    返回内容已经为后续多损失训练准备好:
        - binary_label: real=1, fake=0
        - source_label: real=0, dm=1, ar=2
        - pair: (原始, 后处理)
        - identity/group 信息
    """

    def __init__(
        self,
        real_dir: str,
        real_processed_dir: str,
        dm_dir: str,
        dm_processed_dir: str,
        ar_dir: str,
        ar_processed_dir: str,
        transform=None,
        sync_transform=None,
        return_pil: bool = False,
        strict_check: bool = True,
        limit: Optional[int] = None,
        shuffle_filenames: bool = False,
        seed: int = 2026,
    ):
        super().__init__()

        self.real_dir = real_dir
        self.real_processed_dir = real_processed_dir
        self.dm_dir = dm_dir
        self.dm_processed_dir = dm_processed_dir
        self.ar_dir = ar_dir
        self.ar_processed_dir = ar_processed_dir

        self.transform = transform
        self.sync_transform = sync_transform
        self.return_pil = return_pil
        self.strict_check = strict_check

        self.real_files = list_images(real_dir)
        self.real_processed_files = list_images(real_processed_dir)
        self.dm_files = list_images(dm_dir)
        self.dm_processed_files = list_images(dm_processed_dir)
        self.ar_files = list_images(ar_dir)
        self.ar_processed_files = list_images(ar_processed_dir)

        sets = [
            set(self.real_files.keys()),
            set(self.real_processed_files.keys()),
            set(self.dm_files.keys()),
            set(self.dm_processed_files.keys()),
            set(self.ar_files.keys()),
            set(self.ar_processed_files.keys()),
        ]

        if strict_check:
            base = sets[0]
            for i, s in enumerate(sets[1:], start=1):
                if s != base:
                    raise ValueError(f"文件名不一致: folder[0] 与 folder[{i}] 不匹配")
            filenames = sorted(list(base))
        else:
            filenames = sorted(list(set.intersection(*sets)))

        if shuffle_filenames:
            rng = random.Random(seed)
            rng.shuffle(filenames)

        if limit is not None:
            filenames = filenames[:limit]

        self.filenames = filenames

        self.source_to_id = {
            "real": 0,
            "dm": 1,
            "ar": 2,
        }

        self.binary_real = 1
        self.binary_fake = 0

        print(f"[SixFolderForgeryDataset] Loaded {len(self.filenames)} matched groups.")

    def __len__(self):
        return len(self.filenames)

    def _load_group(self, filename: str) -> Dict[str, Image.Image]:
        return {
            "real": pil_loader(self.real_files[filename]),
            "real_processed": pil_loader(self.real_processed_files[filename]),
            "dm": pil_loader(self.dm_files[filename]),
            "dm_processed": pil_loader(self.dm_processed_files[filename]),
            "ar": pil_loader(self.ar_files[filename]),
            "ar_processed": pil_loader(self.ar_processed_files[filename]),
        }

    def _apply_transforms(self, sample: Dict[str, Image.Image]) -> Dict[str, Union[Image.Image, torch.Tensor]]:
        # 先做同步增强
        if self.sync_transform is not None:
            sample = self.sync_transform(sample)

        # 再逐张做普通 transform
        if self.transform is not None:
            out = {}
            for k, v in sample.items():
                out[k] = self.transform(v)
            return out

        return sample

    def __getitem__(self, idx: int) -> Dict:
        filename = self.filenames[idx]

        sample_pil = self._load_group(filename)
        sample = self._apply_transforms(sample_pil)

        item = {
            "filename": filename,
            "group_id": idx,

            # ========== 六路图像 ==========
            "real": sample["real"],
            "real_processed": sample["real_processed"],
            "dm": sample["dm"],
            "dm_processed": sample["dm_processed"],
            "ar": sample["ar"],
            "ar_processed": sample["ar_processed"],

            # ========== pair ==========
            "pair_real": (sample["real"], sample["real_processed"]),
            "pair_dm": (sample["dm"], sample["dm_processed"]),
            "pair_ar": (sample["ar"], sample["ar_processed"]),

            # ========== binary labels ==========
            "binary_label_real": torch.tensor(self.binary_real, dtype=torch.long),
            "binary_label_real_processed": torch.tensor(self.binary_real, dtype=torch.long),
            "binary_label_dm": torch.tensor(self.binary_fake, dtype=torch.long),
            "binary_label_dm_processed": torch.tensor(self.binary_fake, dtype=torch.long),
            "binary_label_ar": torch.tensor(self.binary_fake, dtype=torch.long),
            "binary_label_ar_processed": torch.tensor(self.binary_fake, dtype=torch.long),

            # ========== source labels ==========
            "source_label_real": torch.tensor(self.source_to_id["real"], dtype=torch.long),
            "source_label_real_processed": torch.tensor(self.source_to_id["real"], dtype=torch.long),
            "source_label_dm": torch.tensor(self.source_to_id["dm"], dtype=torch.long),
            "source_label_dm_processed": torch.tensor(self.source_to_id["dm"], dtype=torch.long),
            "source_label_ar": torch.tensor(self.source_to_id["ar"], dtype=torch.long),
            "source_label_ar_processed": torch.tensor(self.source_to_id["ar"], dtype=torch.long),

            # ========== identity ==========
            "identity_real": f"{filename}_real",
            "identity_dm": f"{filename}_dm",
            "identity_ar": f"{filename}_ar",
        }

        if self.return_pil:
            item["pil"] = sample_pil

        return item


# =========================================================
# collate_fn
# 作用：把多个样本组组织成一个训练 batch
# =========================================================

def six_folder_collate_fn(batch: List[Dict]) -> Dict:
    out = {}

    # 非张量字段
    out["filename"] = [b["filename"] for b in batch]
    out["group_id"] = torch.tensor([b["group_id"] for b in batch], dtype=torch.long)
    out["identity_real"] = [b["identity_real"] for b in batch]
    out["identity_dm"] = [b["identity_dm"] for b in batch]
    out["identity_ar"] = [b["identity_ar"] for b in batch]

    # 六路图像
    img_keys = [
        "real", "real_processed",
        "dm", "dm_processed",
        "ar", "ar_processed"
    ]
    for k in img_keys:
        if torch.is_tensor(batch[0][k]):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]

    # pair
    out["pair_real"] = [(b["pair_real"][0], b["pair_real"][1]) for b in batch]
    out["pair_dm"] = [(b["pair_dm"][0], b["pair_dm"][1]) for b in batch]
    out["pair_ar"] = [(b["pair_ar"][0], b["pair_ar"][1]) for b in batch]

    # binary labels
    binary_keys = [
        "binary_label_real",
        "binary_label_real_processed",
        "binary_label_dm",
        "binary_label_dm_processed",
        "binary_label_ar",
        "binary_label_ar_processed",
    ]
    for k in binary_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    # source labels
    source_keys = [
        "source_label_real",
        "source_label_real_processed",
        "source_label_dm",
        "source_label_dm_processed",
        "source_label_ar",
        "source_label_ar_processed",
    ]
    for k in source_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    # 可选调试字段
    if "pil" in batch[0]:
        out["pil"] = [b["pil"] for b in batch]

    return out


# =========================================================
# 一个便捷构造函数
# 方便你在外部直接调用
# =========================================================

def build_six_folder_dataset(
    real_dir: str,
    real_processed_dir: str,
    dm_dir: str,
    dm_processed_dir: str,
    ar_dir: str,
    ar_processed_dir: str,
    transform=None,
    sync_transform=None,
    return_pil: bool = False,
    strict_check: bool = True,
    limit: Optional[int] = None,
    shuffle_filenames: bool = False,
    seed: int = 2026,
) -> SixFolderForgeryDataset:
    return SixFolderForgeryDataset(
        real_dir=real_dir,
        real_processed_dir=real_processed_dir,
        dm_dir=dm_dir,
        dm_processed_dir=dm_processed_dir,
        ar_dir=ar_dir,
        ar_processed_dir=ar_processed_dir,
        transform=transform,
        sync_transform=sync_transform,
        return_pil=return_pil,
        strict_check=strict_check,
        limit=limit,
        shuffle_filenames=shuffle_filenames,
        seed=seed,
    )