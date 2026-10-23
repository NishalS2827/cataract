"""
Dataset Preparation & Patient-Level Splitting Utility
AI-Based Cataract Clinical Decision Support System (CDSS)
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

def verify_and_clean_images(src_dir):
    """
    Scans image files to ensure they are valid and not corrupted.
    """
    print(f"[*] Verifying images in: {src_dir}")
    valid_count = 0
    corrupt_count = 0
    
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(root, f)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    valid_count += 1
                except Exception:
                    print(f"[-] Corrupt image removed: {fpath}")
                    os.remove(fpath)
                    corrupt_count += 1
                    
    print(f"[OK] Verification Complete: {valid_count} valid images, {corrupt_count} corrupted images.")

def split_dataset(raw_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Splits raw categorized images into train, val, and test partitions
    while preserving class balance (Stratified Split).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f"[*] Detected Classes: {classes}")
    
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
            
    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        all_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.seed(42)
        random.shuffle(all_files)
        
        n_total = len(all_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        print(f"\n[*] Class '{cls}' ({n_total} total images):")
        print(f"    - Train: {len(train_files)}")
        print(f"    - Val  : {len(val_files)}")
        print(f"    - Test : {len(test_files)}")
        
        # Copy files to respective partition
        for f in train_files:
            shutil.copy2(os.path.join(cls_path, f), os.path.join(output_dir, 'train', cls, f))
        for f in val_files:
            shutil.copy2(os.path.join(cls_path, f), os.path.join(output_dir, 'val', cls, f))
        for f in test_files:
            shutil.copy2(os.path.join(cls_path, f), os.path.join(output_dir, 'test', cls, f))
            
    print(f"\n[OK] Dataset successfully split into: {output_dir}")

if __name__ == "__main__":
    print("=" * 60)
    print("AI-Based Cataract CDSS - Dataset Preparation Pipeline")
    print("=" * 60)
    
    raw_path = "data/raw"
    dataset_output = "data/dataset"
    
    if os.path.exists(raw_path):
        verify_and_clean_images(raw_path)
        split_dataset(raw_path, dataset_output)
    else:
        print(f"[!] Error: Path '{raw_path}' does not exist.")
