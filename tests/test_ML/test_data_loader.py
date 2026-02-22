"""
Test script to verify the new data loader and dataset structure
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.segmentation.data_loader import EggDataset
from src.utils.config import load_config
from torchvision import transforms

def test_data_loader():
    """Test the EggDataset class with new folder structure"""
    
    try:
        # Load configuration
        config = load_config('configs/unet_config.yaml')
        
        print("Configuration loaded successfully")
        print(f"Train fertile dir: {config['data']['train_fertile_dir']}")
        print(f"Train infertile dir: {config['data']['train_infertile_dir']}")
        print(f"Val fertile dir: {config['data']['val_fertile_dir']}")
        print(f"Val infertile dir: {config['data']['val_infertile_dir']}")
        print(f"Test fertile dir: {config['data']['test_fertile_dir']}")
        print(f"Test infertile dir: {config['data']['test_infertile_dir']}")
        
        # Create transforms
        transform = transforms.Compose([
            transforms.Resize(config['data']['image_size']),
            transforms.ToTensor()
        ])
        
        # Test training dataset
        print("\nTesting training dataset:")
        train_dataset = EggDataset(
            config['data']['train_fertile_dir'],
            config['data']['train_infertile_dir'],
            config['data']['image_size'],
            transform
        )
        print(f"Training dataset size: {len(train_dataset)}")
        
        # Test validation dataset
        print("\nTesting validation dataset:")
        val_dataset = EggDataset(
            config['data']['val_fertile_dir'],
            config['data']['val_infertile_dir'],
            config['data']['image_size'],
            transform
        )
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Test test dataset
        print("\nTesting test dataset:")
        test_dataset = EggDataset(
            config['data']['test_fertile_dir'],
            config['data']['test_infertile_dir'],
            config['data']['image_size'],
            transform
        )
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Test loading some samples
        print("\nTesting sample loading:")
        if len(train_dataset) > 0:
            image, label = train_dataset[0]
            print(f"First training sample - Image shape: {image.shape}, Label: {label}")
            
        if len(val_dataset) > 0:
            image, label = val_dataset[0]
            print(f"First validation sample - Image shape: {image.shape}, Label: {label}")
            
        if len(test_dataset) > 0:
            image, label = test_dataset[0]
            print(f"First test sample - Image shape: {image.shape}, Label: {label}")
        
        # Check label distribution
        print("\nChecking label distribution:")
        
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        train_fertile = sum(1 for label in train_labels if label == 1)
        train_infertile = sum(1 for label in train_labels if label == 0)
        print(f"Training - Fertile: {train_fertile}, Infertile: {train_infertile}")
        
        val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
        val_fertile = sum(1 for label in val_labels if label == 1)
        val_infertile = sum(1 for label in val_labels if label == 0)
        print(f"Validation - Fertile: {val_fertile}, Infertile: {val_infertile}")
        
        test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
        test_fertile = sum(1 for label in test_labels if label == 1)
        test_infertile = sum(1 for label in test_labels if label == 0)
        print(f"Test - Fertile: {test_fertile}, Infertile: {test_infertile}")
        
        print("\n✅ Data loader test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print(f"\nStack trace: {traceback.format_exc()}")
        return False
        
    return True

if __name__ == "__main__":
    print("Testing EggDataset with new folder structure...")
    test_data_loader()
