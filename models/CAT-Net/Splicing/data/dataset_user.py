import project_config
from Splicing.data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch
from tqdm import tqdm
import glob

class UserDataset(AbstractDataset):
    def __init__(self, crop_size, grid_crop, blocks: list, DCT_channels: int, data_root: str, mode: str, read_from_jpeg=False):
        """
        :param crop_size: (H,W) or None
        :param blocks:
        :param data_root: path to the data root directory
        :param mode: 'train' or 'test'
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        """
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        self.read_from_jpeg = read_from_jpeg
        self.tamp_list = []
        
        tampered_dir = os.path.join(data_root, mode, 'tampered')
        masks_dir = os.path.join(data_root, mode, 'masks')
        
        tampered_files = []
        tampered_files.extend(sorted(glob.glob(os.path.join(tampered_dir, '*.jpg'))))
        tampered_files.extend(sorted(glob.glob(os.path.join(tampered_dir, '*.jpeg'))))
        tampered_files.extend(sorted(glob.glob(os.path.join(tampered_dir, '*.png'))))
        tampered_files.extend(sorted(glob.glob(os.path.join(tampered_dir, '*.tif'))))
        
        for tampered_path in tampered_files:
            base_name = os.path.basename(tampered_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            mask_path_jpg = os.path.join(masks_dir, f'{name_without_ext}_mask.jpg')
            mask_path_jpeg = os.path.join(masks_dir, f'{name_without_ext}_mask.jpeg')
            mask_path_png = os.path.join(masks_dir, f'{name_without_ext}_mask.png')
            mask_path_tif = os.path.join(masks_dir, f'{name_without_ext}_mask.tif')
            
            if os.path.exists(mask_path_jpg):
                self.tamp_list.append([tampered_path, mask_path_jpg])
            elif os.path.exists(mask_path_jpeg):
                self.tamp_list.append([tampered_path, mask_path_jpeg])
            elif os.path.exists(mask_path_png):
                self.tamp_list.append([tampered_path, mask_path_png])
            else:
                print(f"Warning: Mask not found for {tampered_path}")

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path, mask_path = self.tamp_list[index]
        
        # Load mask
        mask = np.array(Image.open(mask_path).convert('L'))
        # Convert mask to binary (0 for background, 1 for tampered)
        mask = (mask > 127).astype(np.float32)
        
        if not self.read_from_jpeg and os.path.splitext(tamp_path)[1].lower() != '.jpg':
            temp_jpg = f"____temp_{index:04d}.jpg"
            Image.open(tamp_path).convert('RGB').save(temp_jpg, quality=100, subsampling=0)
            tensors = self._create_tensor(temp_jpg, mask)
            os.remove(temp_jpg)
        else:
            tensors = self._create_tensor(tamp_path, mask)
        return tensors
    
    def __getitem__(self, index):
        """Return data for DataLoader"""
        return self.get_tamp(index)
