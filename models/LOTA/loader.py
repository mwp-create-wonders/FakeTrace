import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from pathlib import Path
from typing import List, Union
import random

from bit_patch import bit_patch as bit_patch_process


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def collate_batch(batch):
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch]).float()
    return inputs, labels


def create_preprocessing_pipeline(options):
    if options.isPatch:
        transform_func = transforms.Lambda(
            lambda img: bit_patch_process(
                img, options.img_height, options.bit_mode,
                options.patch_size, options.patch_mode
            )
        )
    else:
        transform_func = transforms.Resize((options.img_height, options.img_height))

    return transforms.Compose([
        transform_func,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def apply_preprocessing(image, options):
    pipeline = create_preprocessing_pipeline(options)
    return pipeline(image)


def _to_dir_list(dir_input):
    if dir_input is None:
        return []

    if isinstance(dir_input, str):
        dir_input = dir_input.strip()
        if not dir_input:
            return []
        if ',' in dir_input:
            return [x.strip() for x in dir_input.split(',') if x.strip()]
        return [dir_input]

    if isinstance(dir_input, (list, tuple)):
        return [str(x).strip() for x in dir_input if str(x).strip()]

    raise TypeError(f"Unsupported dir_input type: {type(dir_input)}")


def _collect_images_from_dirs(dir_list, recursive=True):
    dir_list = _to_dir_list(dir_list)
    all_images = []

    for d in dir_list:
        if not os.path.isdir(d):
            print(f"[Warning] Directory not found: {d}")
            continue

        if recursive:
            for root, _, files in os.walk(d):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in IMG_EXTENSIONS:
                        all_images.append(os.path.join(root, f))
        else:
            for f in os.listdir(d):
                full_path = os.path.join(d, f)
                if os.path.isfile(full_path):
                    ext = Path(f).suffix.lower()
                    if ext in IMG_EXTENSIONS:
                        all_images.append(full_path)

    return sorted(all_images)


class MultiSourceRealFakeDataset(Dataset):
    def __init__(
        self,
        real_dirs,
        fake_dirs,
        options,
        recursive=True,
        max_real=None,
        max_fake=None,
        random_sample=True,
    ):
        super().__init__()
        self.options = options

        self.real_paths = _collect_images_from_dirs(real_dirs, recursive=recursive)
        self.fake_paths = _collect_images_from_dirs(fake_dirs, recursive=recursive)

        if len(self.real_paths) == 0:
            raise ValueError(f"No real images found in: {_to_dir_list(real_dirs)}")
        if len(self.fake_paths) == 0:
            raise ValueError(f"No fake images found in: {_to_dir_list(fake_dirs)}")

        # 限制 real 数量
        if max_real is not None and max_real > 0 and len(self.real_paths) > max_real:
            if random_sample:
                self.real_paths = random.sample(self.real_paths, max_real)
            else:
                self.real_paths = self.real_paths[:max_real]

        # 限制 fake 数量
        if max_fake is not None and max_fake > 0 and len(self.fake_paths) > max_fake:
            if random_sample:
                self.fake_paths = random.sample(self.fake_paths, max_fake)
            else:
                self.fake_paths = self.fake_paths[:max_fake]

        self.all_images = self.real_paths + self.fake_paths
        self.labels = torch.cat([
            torch.ones(len(self.real_paths), dtype=torch.float32),
            torch.zeros(len(self.fake_paths), dtype=torch.float32)
        ])

        print(f"[Dataset] real images: {len(self.real_paths)}")
        print(f"[Dataset] fake images: {len(self.fake_paths)}")
        print(f"[Dataset] total images: {len(self.all_images)}")

    def _load_rgb(self, img_path):
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as e:
            print(f"Image Loading Error {img_path}: {str(e)}")
            return Image.new('RGB', (256, 256), (0, 0, 0))

    def __getitem__(self, index):
        try:
            img = self._load_rgb(self.all_images[index])
            label = self.labels[index]
        except Exception:
            prev_index = max(0, index - 1)
            img = self._load_rgb(self.all_images[prev_index])
            label = self.labels[prev_index]

        processed_img = apply_preprocessing(img, self.options)
        return processed_img, label

    def __len__(self):
        return len(self.all_images)


def create_training_loader(options):
    """
    训练集接口：
    options.train_real_dirs
    options.train_fake_dirs
    """
    train_dataset = MultiSourceRealFakeDataset(
        real_dirs=options.train_real_dirs,
        fake_dirs=options.train_fake_dirs,
        options=options,
        recursive=getattr(options, "recursive", True)
    )

    return DataLoader(
        train_dataset,
        batch_size=options.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


def create_validation_loader(options, real_dirs, fake_dirs, max_real=None, max_fake=None):
    val_dataset = MultiSourceRealFakeDataset(
        real_dirs=real_dirs,
        fake_dirs=fake_dirs,
        options=options,
        recursive=getattr(options, "recursive", True),
        max_real=max_real,
        max_fake=max_fake,
        random_sample=True
    )

    return DataLoader(
        val_dataset,
        batch_size=options.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch
    ), len(val_dataset)


def get_loader(opt):
    return create_training_loader(opt)


def get_val_loader(opt):
    """
    默认验证集
    """
    return create_validation_loader(
        opt,
        real_dirs=opt.val_real_dirs,
        fake_dirs=opt.val_fake_dirs
    )


def get_single_loader(opt, real_dirs, fake_dirs):
    test_num = getattr(opt, "test_num", None)

    return create_validation_loader(
        opt,
        real_dirs=real_dirs,
        fake_dirs=fake_dirs,
        max_real=test_num,
        max_fake=test_num
    )
