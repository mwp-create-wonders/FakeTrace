import os
import sys
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_dataset(source_dir, output_dir, train_ratio=0.85, seed=42):
    random.seed(seed)
    
    tampered_dir = os.path.join(source_dir, 'tampered')
    masks_dir = os.path.join(source_dir, 'masks')
    
    if not os.path.exists(tampered_dir) or not os.path.exists(masks_dir):
        print(f"Error: Directory not found - {tampered_dir} or {masks_dir}")
        return
    
    train_tampered_dir = os.path.join(output_dir, 'train', 'tampered')
    train_masks_dir = os.path.join(output_dir, 'train', 'masks')
    test_tampered_dir = os.path.join(output_dir, 'test', 'tampered')
    test_masks_dir = os.path.join(output_dir, 'test', 'masks')
    
    os.makedirs(train_tampered_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(test_tampered_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)
    
    tampered_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        tampered_files.extend(sorted(Path(tampered_dir).glob(ext)))
    
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        mask_files.extend(sorted(Path(masks_dir).glob(ext)))
    
    tampered_files = sorted(tampered_files)
    mask_files = sorted(mask_files)
    
    print(f"Found {len(tampered_files)} tampered images and {len(mask_files)} masks")
    
    paired_files = []
    for tampered_path in tampered_files:
        name_without_ext = os.path.splitext(os.path.basename(tampered_path))[0]
        mask_pattern = f"{name_without_ext}_mask.*"
        found_mask = None
        for mask_path in mask_files:
            mask_name = os.path.basename(mask_path)
            if mask_name.startswith(f"{name_without_ext}_mask"):
                found_mask = mask_path
                break
        if found_mask:
            paired_files.append((tampered_path, found_mask))
        else:
            print(f"Warning: Mask not found for {tampered_path}")
    
    print(f"Found {len(paired_files)} paired image-mask files")
    
    random.shuffle(paired_files)
    
    split_idx = int(len(paired_files) * train_ratio)
    train_pairs = paired_files[:split_idx]
    test_pairs = paired_files[split_idx:]
    
    print(f"Training set: {len(train_pairs)} samples")
    print(f"Test set: {len(test_pairs)} samples")
    
    for tampered_path, mask_path in tqdm(train_pairs, desc="Copying training files"):
        shutil.copy(str(tampered_path), train_tampered_dir)
        shutil.copy(str(mask_path), train_masks_dir)
    
    for tampered_path, mask_path in tqdm(test_pairs, desc="Copying test files"):
        shutil.copy(str(tampered_path), test_tampered_dir)
        shutil.copy(str(mask_path), test_masks_dir)
    
    print(f"Dataset split completed successfully!")
    print(f"Training data saved to: {os.path.join(output_dir, 'train')}")
    print(f"Test data saved to: {os.path.join(output_dir, 'test')}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python split_dataset.py <source_dataset_dir> <output_dataset_dir>")
        print("Example: python split_dataset.py /path/to/SID /path/to/SID_split")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    split_dataset(source_dir, output_dir, train_ratio=0.85, seed=42)