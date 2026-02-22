"""
Script to organize duck egg images into fertile/infertile categories
for training, validation, and test sets.
"""

import os
import random
import shutil
from pathlib import Path

def create_category_folders(base_dir: str):
    """Create folders for fertile and infertile images in train/val/test directories"""
    folders = [
        'train/fertile', 'train/infertile',
        'val/fertile', 'val/infertile',
        'test/fertile', 'test/infertile'
    ]
    
    for folder in folders:
        folder_path = Path(base_dir) / folder
        folder_path.mkdir(parents=True, exist_ok=True)

def get_image_files(directory: str):
    """Get list of image files in a directory"""
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(directory).glob(f'*{ext}'))
        image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted(image_files)

def split_dataset(image_files, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split dataset into train/val/test sets"""
    random.shuffle(image_files)
    
    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    return train_files, val_files, test_files

def copy_images(files, destination_dir):
    """Copy images to destination directory"""
    destination_path = Path(destination_dir)
    destination_path.mkdir(exist_ok=True)
    
    for file_path in files:
        destination_file = destination_path / file_path.name
        shutil.copy2(file_path, destination_file)
        print(f"Copied: {file_path.name} to {destination_dir}")

def main():
    # Configuration
    raw_data_dir = "data/raw/Dataset1_JPG-20260128T021855Z-3-001/Dataset1_JPG"
    output_base_dir = "data"
    random_seed = 42
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    print("Organizing duck egg fertility dataset...")
    
    # Step 1: Create category folders
    create_category_folders(output_base_dir)
    
    # Step 2: Get all image files
    image_files = get_image_files(raw_data_dir)
    print(f"Found {len(image_files)} images in raw dataset")
    
    if len(image_files) == 0:
        print("Error: No images found in raw dataset")
        return
    
    # Step 3: Split dataset (assuming we have annotations or use random split)
    # For this example, we'll use random split since we don't have annotations yet
    train_files, val_files, test_files = split_dataset(image_files)
    
    print(f"\nSplit distribution:")
    print(f"Training: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")
    
    # Step 4: Distribute images into fertile/infertile folders (random assignment for demonstration)
    # NOTE: In real scenario, you should use actual annotations or labels
    
    # Randomly assign 50% as fertile, 50% as infertile for each set
    def split_into_categories(files):
        random.shuffle(files)
        n = len(files)
        n_fertile = n // 2
        n_infertile = n - n_fertile
        return files[:n_fertile], files[n_fertile:]
    
    print("\nAssigning categories (random for demonstration):")
    
    # Training set
    train_fertile, train_infertile = split_into_categories(train_files)
    print(f"Training fertile: {len(train_fertile)}, infertile: {len(train_infertile)}")
    copy_images(train_fertile, f"{output_base_dir}/train/fertile")
    copy_images(train_infertile, f"{output_base_dir}/train/infertile")
    
    # Validation set
    val_fertile, val_infertile = split_into_categories(val_files)
    print(f"Validation fertile: {len(val_fertile)}, infertile: {len(val_infertile)}")
    copy_images(val_fertile, f"{output_base_dir}/val/fertile")
    copy_images(val_infertile, f"{output_base_dir}/val/infertile")
    
    # Test set
    test_fertile, test_infertile = split_into_categories(test_files)
    print(f"Test fertile: {len(test_fertile)}, infertile: {len(test_infertile)}")
    copy_images(test_fertile, f"{output_base_dir}/test/fertile")
    copy_images(test_infertile, f"{output_base_dir}/test/infertile")
    
    print("\n✅ Dataset organization complete!")
    print(f"\nOutput structure:")
    for root, dirs, files in os.walk(output_base_dir):
        level = root.replace(output_base_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        folder_name = os.path.basename(root)
        print(f"{indent}{folder_name}/")
        for file in files[:3]:  # Show first 3 files as sample
            print(f"{' ' * 2 * (level + 1)}{file}")
        if len(files) > 3:
            print(f"{' ' * 2 * (level + 1)}... and {len(files) - 3} more")

if __name__ == "__main__":
    main()
