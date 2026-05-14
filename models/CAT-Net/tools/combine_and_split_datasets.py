import os
import sys
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def collect_paired_files(dataset_dir):
    tampered_dir = os.path.join(dataset_dir, 'tampered')
    masks_dir = os.path.join(dataset_dir, 'masks')
    
    if not os.path.exists(tampered_dir) or not os.path.exists(masks_dir):
        print(f"Warning: Directory not found - {tampered_dir} or {masks_dir}")
        return []
    
    tampered_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        tampered_files.extend(sorted(Path(tampered_dir).glob(ext)))
    
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        mask_files.extend(sorted(Path(masks_dir).glob(ext)))
    
    paired_files = []
    for tampered_path in tampered_files:
        name_without_ext = os.path.splitext(os.path.basename(tampered_path))[0]
        found_mask = None
        for mask_path in mask_files:
            mask_name = os.path.basename(mask_path)
            if mask_name.startswith(f"{name_without_ext}_mask"):
                found_mask = mask_path
                break
        if found_mask:
            paired_files.append((str(tampered_path), str(found_mask)))
        else:
            print(f"Warning: Mask not found for {tampered_path}")
    
    return paired_files

def combine_and_split(sid_dir, cocoglide_dir, output_dir, train_ratio=0.85, seed=42):
    random.seed(seed)
    
    print("Collecting SID dataset files...")
    sid_pairs = collect_paired_files(sid_dir)
    print(f"Found {len(sid_pairs)} paired files in SID")
    
    print("Collecting CocoGlide dataset files...")
    cocoglide_pairs = collect_paired_files(cocoglide_dir)
    print(f"Found {len(cocoglide_pairs)} paired files in CocoGlide")
    
    all_pairs = sid_pairs + cocoglide_pairs
    print(f"Total paired files: {len(all_pairs)}")
    
    random.shuffle(all_pairs)
    
    split_idx = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    train_tampered_dir = os.path.join(output_dir, 'train', 'tampered')
    train_masks_dir = os.path.join(output_dir, 'train', 'masks')
    test_tampered_dir = os.path.join(output_dir, 'test', 'tampered')
    test_masks_dir = os.path.join(output_dir, 'test', 'masks')
    
    os.makedirs(train_tampered_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(test_tampered_dir, exist_ok=True)
    os.makedirs(test_masks_dir, exist_ok=True)
    
    print(f"\nTraining set: {len(train_pairs)} samples")
    print(f"Test set: {len(test_pairs)} samples")
    
    for tampered_path, mask_path in tqdm(train_pairs, desc="Copying training files"):
        filename = os.path.basename(tampered_path)
        shutil.copy(tampered_path, os.path.join(train_tampered_dir, filename))
        
        mask_filename = os.path.basename(mask_path)
        shutil.copy(mask_path, os.path.join(train_masks_dir, mask_filename))
    
    for tampered_path, mask_path in tqdm(test_pairs, desc="Copying test files"):
        filename = os.path.basename(tampered_path)
        shutil.copy(tampered_path, os.path.join(test_tampered_dir, filename))
        
        mask_filename = os.path.basename(mask_path)
        shutil.copy(mask_path, os.path.join(test_masks_dir, mask_filename))
    
    print(f"\nDataset combination and split completed successfully!")
    print(f"Training data saved to: {os.path.join(output_dir, 'train')}")
    print(f"Test data saved to: {os.path.join(output_dir, 'test')}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python combine_and_split_datasets.py <SID_dir> <CocoGlide_dir> <output_dir>")
        print("Example: python combine_and_split_datasets.py /path/to/SID /path/to/CocoGlide /path/to/combined_dataset")
        sys.exit(1)
    
    sid_dir = sys.argv[1]
    cocoglide_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    combine_and_split(sid_dir, cocoglide_dir, output_dir, train_ratio=0.85, seed=42)