#!/usr/bin/env python3
"""
Download and prepare a small deepfake dataset for quick training.
This downloads a manageable dataset to get started quickly.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import shutil

def download_file(url, dest_path):
    """Download file with progress bar."""
    print(f"Downloading from {url}...")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end='')
    
    urllib.request.urlretrieve(url, dest_path, reporthook)
    print("\n✅ Download complete!")

def prepare_dataset_structure(base_dir):
    """Create the expected directory structure."""
    base_path = Path(base_dir)
    
    # Create directories
    (base_path / 'images' / 'real').mkdir(parents=True, exist_ok=True)
    (base_path / 'images' / 'fake').mkdir(parents=True, exist_ok=True)
    (base_path / 'videos' / 'real').mkdir(parents=True, exist_ok=True)
    (base_path / 'videos' / 'fake').mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created dataset structure in {base_dir}")

def download_sample_dataset():
    """
    Download a small sample dataset for testing.
    Note: For production, use larger datasets from Kaggle.
    """
    print("=" * 70)
    print("📥 Download Sample Deepfake Dataset")
    print("=" * 70)
    print()
    print("Options:")
    print("1. Create synthetic test dataset (instant, for testing pipeline)")
    print("2. Download from Kaggle (requires API setup)")
    print("3. Manual download instructions")
    print()
    
    choice = input("Choose an option (1-3): ").strip()
    
    if choice == "1":
        create_synthetic_dataset()
    elif choice == "2":
        download_from_kaggle()
    else:
        show_manual_instructions()

def create_synthetic_dataset():
    """Create a tiny synthetic dataset for testing the training pipeline."""
    import numpy as np
    from PIL import Image
    
    print("\n🔧 Creating synthetic test dataset...")
    
    base_dir = Path('datasets')
    prepare_dataset_structure(base_dir)
    
    # Create synthetic images
    img_size = 224
    num_samples = 100  # Small for quick testing
    
    print(f"Generating {num_samples} synthetic images per class...")
    
    for label, label_name in [(0, 'real'), (1, 'fake')]:
        out_dir = base_dir / 'images' / label_name
        
        for i in range(num_samples):
            # Create random image
            if label == 0:
                # "Real" - more natural colors
                img_array = np.random.randint(50, 200, (img_size, img_size, 3), dtype=np.uint8)
            else:
                # "Fake" - add some artifacts
                img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
                # Add some structured noise to simulate artifacts
                noise_rows = len(img_array[::10, :])
                img_array[::10, :] = np.random.randint(0, 50, (noise_rows, img_size, 3))
            
            img = Image.fromarray(img_array)
            img.save(out_dir / f'{label_name}_{i:04d}.jpg')
        
        print(f"  ✅ Created {num_samples} {label_name} images")
    
    print("\n✅ Synthetic dataset created!")
    print(f"Location: {base_dir}/images/")
    print("\nNote: This is for testing the pipeline only.")
    print("For real training, download actual deepfake datasets from Kaggle.")
    print("\nYou can now run:")
    print("  python3 train_vit.py --data_dir datasets/images --epochs 5")

def download_from_kaggle():
    """Download dataset from Kaggle."""
    print("\n" + "=" * 70)
    print("📥 Kaggle Dataset Download")
    print("=" * 70)
    print()
    
    # Check if Kaggle is set up
    kaggle_dir = Path.home() / '.kaggle'
    if not (kaggle_dir / 'kaggle.json').exists():
        print("❌ Kaggle API not configured!")
        print("\nPlease run: ./setup_kaggle.sh")
        print("Or use: python3 download_datasets.py")
        return
    
    print("Recommended datasets:")
    print()
    print("1. Deepfake and Real Images (600MB, quick)")
    print("   ID: manjilkarki/deepfake-and-real-images")
    print()
    print("2. 140k Real and Fake Faces (5GB, better quality)")
    print("   ID: xhlulu/140k-real-and-fake-faces")
    print()
    
    choice = input("Choose dataset (1 or 2): ").strip()
    
    if choice == "1":
        dataset_id = "manjilkarki/deepfake-and-real-images"
    elif choice == "2":
        dataset_id = "xhlulu/140k-real-and-fake-faces"
    else:
        print("Invalid choice!")
        return
    
    print(f"\nDownloading {dataset_id}...")
    os.system(f'kaggle datasets download -d {dataset_id} -p datasets --unzip')
    print("\n✅ Download complete!")

def show_manual_instructions():
    """Show manual download instructions."""
    print("\n" + "=" * 70)
    print("📖 Manual Download Instructions")
    print("=" * 70)
    print()
    print("1. Go to Kaggle and download one of these datasets:")
    print()
    print("   • Deepfake and Real Images (600MB)")
    print("     https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images")
    print()
    print("   • 140k Real and Fake Faces (5GB)")
    print("     https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
    print()
    print("2. Extract the dataset")
    print()
    print("3. Organize into this structure:")
    print("   datasets/")
    print("   └── images/")
    print("       ├── real/")
    print("       │   ├── img001.jpg")
    print("       │   └── ...")
    print("       └── fake/")
    print("           ├── img001.jpg")
    print("           └── ...")
    print()
    print("4. Run training:")
    print("   python3 train_vit.py --data_dir datasets/images --epochs 20")
    print()

if __name__ == '__main__':
    download_sample_dataset()
