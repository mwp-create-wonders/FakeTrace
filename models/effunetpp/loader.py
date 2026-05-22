import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class mydataset(Dataset):
    def __init__(self, root, df):
        self.df = df
        self.root = root
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        real_img_path = os.path.join(self.root, row["real"])
        fake_img_path = os.path.join(self.root, row["fake"])
        mask_path = os.path.join(self.root, row["mask"])
        real_img = Image.open(real_img_path).convert("RGB")
        fake_img = Image.open(fake_img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        real_img = self.transform(real_img)
        fake_img = self.transform(fake_img)
        mask = self.transform(mask)
        black_mask = torch.zeros_like(mask)
        return [(real_img, black_mask, 1), (fake_img, mask, 0)]


class loader:
    def __init__(self, root, batch_size, num_workers):
        df = pd.read_csv(os.path.join(root, "table.csv"))
        full_dataset = mydataset(root, df)
        dataset_size = len(full_dataset)
        eval_num = int(dataset_size * 0.15)
        test_num = int(dataset_size * 0.15)
        train_num = dataset_size - eval_num - test_num
        torch.manual_seed(0)
        train_dataset, eval_dataset, test_dataset = random_split(
            full_dataset, [train_num, eval_num, test_num]
        )
        persistent = False if num_workers == 0 else True
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent,
        )
