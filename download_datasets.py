#!/usr/bin/env python3
"""
Download popular deepfake datasets from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Get API credentials from: https://www.kaggle.com/settings
3. Place kaggle.json in ~/.kaggle/

Popular Datasets:
- Deepfake Detection Challenge (DFDC) - 470GB
- FaceForensics++ - Widely used benchmark
- Celeb-DF - Celebrity deepfakes
- DFFD - Diverse Fake Face Dataset
"""

import os
import subprocess

# Popular deepfake datasets
DATASETS = {
    "1": {
        "name": "DFDC Preview Dataset (Small)",
        "kaggle_id": "c/deepfake-detection-challenge",
        "size": "~4GB",
        "description": "Preview dataset from DFDC competition with real and fake videos"
    },
    "2": {
        "name": "140k Real and Fake Faces",
        "kaggle_id": "xhlulu/140k-real-and-fake-faces",
        "size": "~5GB",
        "description": "140k images of real and fake faces for training"
    },
    "3": {
        "name": "Deepfake and Real Images",
        "kaggle_id": "manjilkarki/deepfake-and-real-images",
        "size": "~600MB",
        "description": "Curated dataset of deepfake and real images"
    },
    "4": {
        "name": "FaceForensics (Subset)",
        "kaggle_id": "sorokin/faceforensics",
        "size": "~2GB",
        "description": "Subset of FaceForensics++ dataset"
    },
    "5": {
        "name": "Deepfake Detection Dataset",
        "kaggle_id": "borhanitrash/deepfake-detection-dataset",
        "size": "~800MB",
        "description": "Balanced dataset for deepfake detection"
    }
}

def check_kaggle_setup():
    """Check if Kaggle API is set up correctly."""
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        print("❌ Kaggle API not configured!")
        print("\n📋 Setup Instructions:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Save kaggle.json to: ~/.kaggle/")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        print(f"✅ Kaggle CLI installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Kaggle CLI not installed!")
        print("Install with: pip install kaggle")
        return False

def download_dataset(kaggle_id, output_dir="datasets"):
    """Download dataset from Kaggle."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📥 Downloading: {kaggle_id}")
    print(f"📁 Output directory: {output_dir}")
    
    if "/c/" in kaggle_id or kaggle_id.startswith("c/"):
        # Competition dataset
        cmd = f"kaggle competitions download -c {kaggle_id.replace('c/', '')} -p {output_dir}"
    else:
        # Regular dataset
        cmd = f"kaggle datasets download -d {kaggle_id} -p {output_dir} --unzip"
    
    print(f"🔧 Running: {cmd}")
    os.system(cmd)
    print("✅ Download complete!")

def main():
    print("=" * 70)
    print("📥 Kaggle Deepfake Dataset Downloader")
    print("=" * 70)
    print()
    
    # Check setup
    if not check_kaggle_setup():
        return
    
    print("\n" + "=" * 70)
    print("📊 Available Datasets:")
    print("=" * 70)
    
    for key, dataset in DATASETS.items():
        print(f"\n{key}. {dataset['name']}")
        print(f"   ID: {dataset['kaggle_id']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Description: {dataset['description']}")
    
    print("\n" + "=" * 70)
    choice = input("\nEnter dataset number to download (or 'all' for all datasets): ").strip()
    
    if choice.lower() == "all":
        for dataset in DATASETS.values():
            download_dataset(dataset['kaggle_id'])
    elif choice in DATASETS:
        download_dataset(DATASETS[choice]['kaggle_id'])
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
