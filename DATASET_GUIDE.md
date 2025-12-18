# 📊 Deepfake Dataset Guide

## 🎯 What You Need

For a complete deepfake detection system, you need:

1. **Gating Classifier** (✅ Already trained with synthetic data)
   - Routes videos to appropriate models
   - Can be retrained with real video features

2. **ViT Model** (❌ Not yet built)
   - For high-resolution, static videos
   - Needs training on deepfake dataset

3. **CNN Model** (❌ Not yet built)
   - For dynamic, motion-heavy videos
   - Needs training on deepfake dataset

## 📥 Recommended Datasets from Kaggle

### 🥇 Best for Beginners

#### 1. **140k Real and Fake Faces** (Recommended!)
- **Kaggle ID**: `xhlulu/140k-real-and-fake-faces`
- **Size**: ~5GB
- **Type**: Images
- **Why**: Perfect for training ViT, balanced dataset, easy to work with
- **Download**: 
  ```bash
  kaggle datasets download -d xhlulu/140k-real-and-fake-faces --unzip
  ```

#### 2. **Deepfake and Real Images**
- **Kaggle ID**: `manjilkarki/deepfake-and-real-images`
- **Size**: ~600MB
- **Type**: Images
- **Why**: Small, curated, great for quick experiments

### 🎥 Video Datasets

#### 3. **DFDC Preview Dataset**
- **Kaggle ID**: `c/deepfake-detection-challenge`
- **Size**: ~4GB (preview), full dataset is 470GB!
- **Type**: Videos
- **Why**: Industry standard, realistic scenarios
- **Note**: Requires Kaggle competition acceptance

#### 4. **FaceForensics (Subset)**
- **Kaggle ID**: `sorokin/faceforensics`
- **Size**: ~2GB
- **Type**: Videos
- **Why**: Academic benchmark, multiple manipulation types

## 🚀 Quick Start Guide

### Step 1: Install Kaggle CLI

```bash
pip3 install kaggle
```

### Step 2: Get API Credentials

1. Go to: https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`
5. Move to proper location:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 3: Download Dataset

**Option A: Using our script**
```bash
python3 download_datasets.py
```

**Option B: Manual download**
```bash
# Create datasets directory
mkdir -p datasets

# Download 140k faces dataset
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p datasets --unzip

# Or download deepfake images
kaggle datasets download -d manjilkarki/deepfake-and-real-images -p datasets --unzip
```

### Step 4: Organize Data

Expected structure:
```
datasets/
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── fake/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

Or for videos:
```
datasets/
├── real/
│   ├── video1.mp4
│   └── ...
└── fake/
    ├── video1.mp4
    └── ...
```

## 🔄 What to Do After Downloading

### For Image Datasets:
1. **Train ViT model** on the images
2. **Train CNN model** on the images
3. **Retrain gating classifier** if you have videos

### For Video Datasets:
1. **Extract features** using `feature_extraction.py`
2. **Train gating classifier** on real video features
3. **Extract frames** for ViT/CNN training

## 📋 Dataset Comparison

| Dataset | Size | Type | Real/Fake Ratio | Best For |
|---------|------|------|-----------------|----------|
| 140k Faces | 5GB | Images | 50/50 | ViT training, balanced |
| Deepfake Images | 600MB | Images | Balanced | Quick experiments |
| DFDC | 4GB-470GB | Videos | Varied | Production systems |
| FaceForensics | 2GB | Videos | Balanced | Gating classifier |

## 🎓 Training Pipeline

```
1. Download Dataset
   ↓
2. Preprocess (resize, normalize)
   ↓
3. Train ViT Model (for image classification)
   ↓
4. Train CNN Model (for temporal analysis)
   ↓
5. Train/Retrain Gating Classifier (for routing)
   ↓
6. Build Ensemble System
   ↓
7. Deploy! 🚀
```

## 💡 Pro Tips

1. **Start small**: Use the 600MB dataset first to validate your pipeline
2. **GPU required**: Deepfake detection models need GPU for training
3. **Data augmentation**: Use rotation, flipping, color jitter for better generalization
4. **Mixed training**: Combine multiple datasets for robustness
5. **Validation split**: Keep 20% for validation, never train on it

## 🔗 Additional Resources

- **FaceForensics++ (Official)**: https://github.com/ondyari/FaceForensics
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics
- **DFDC on Kaggle**: https://www.kaggle.com/c/deepfake-detection-challenge

## ❓ Need Help?

Common issues:
- **"401 Unauthorized"**: Check your kaggle.json credentials
- **"403 Forbidden"**: Accept competition rules on Kaggle website
- **Slow download**: Use `--unzip` flag or manually unzip after
- **Disk space**: Check available space before downloading large datasets

---

Ready to download? Run:
```bash
python3 download_datasets.py
```
