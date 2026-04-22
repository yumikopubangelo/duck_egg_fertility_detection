import os
import shutil
import random
from pathlib import Path

def setup_splits(base_dir: str):
    base_path = Path(base_dir)
    splits = ['train', 'val', 'test']
    categories = ['fertile', 'infertile']
    
    all_files = {'fertile': set(), 'infertile': set()}
    for split in splits:
        for category in categories:
            cat_path = base_path / split / category
            if cat_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for f in cat_path.glob(ext):
                        all_files[category].add(f.resolve())
                        
    print(f\"Total unique fertile: {len(all_files['fertile'])}\")
    print(f\"Total unique infertile: {len(all_files['infertile'])}\")

    for split in splits:
        split_path = base_path / split
        if split_path.exists():
            shutil.rmtree(split_path)
            
    for split in splits:
        for category in categories:
            (base_path / split / category).mkdir(parents=True, exist_ok=True)
            
    return all_files

def distribute_files(all_files: dict, base_dir: str):
    random.seed(42)
    base_path = Path(base_dir)
    
    for category, files in all_files.items():
        file_list = sorted(list(files))
        random.shuffle(file_list)
        
        n = len(file_list)
        n_train = int(n * 0.7)
        n_val = int(n * 0.2)
        
        train_files = file_list[:n_train]
        val_files = file_list[n_train:n_train+n_val]
        test_files = file_list[n_train+n_val:]
        
        for f in train_files:
            shutil.copy2(f, base_path / 'train' / category / f.name)
        for f in val_files:
            shutil.copy2(f, base_path / 'val' / category / f.name)
        for f in test_files:
            shutil.copy2(f, base_path / 'test' / category / f.name)
            
        print(f\"\n{category.capitalize()} Distribution:\")
        print(f\"Train: {len(train_files)}\")
        print(f\"Val: {len(val_files)}\")
        print(f\"Test: {len(test_files)}\")

if __name__ == '__main__':
    base_data_dir = 'data'
    print('Memulai pembersihan dan split data...')
    files_dict = setup_splits(base_data_dir)
    distribute_files(files_dict, base_data_dir)
    print('\nSplit 70-20-10 Selesai!')
